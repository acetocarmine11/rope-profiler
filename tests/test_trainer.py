from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest

from rope_profiler.integrations.trainer import attach_rope_monitor
from rope_profiler.monitor import MonitorConfig


LoopOutput = namedtuple("LoopOutput", "predictions label_ids metrics num_samples")


class FakeMonitor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.sessions = []
        self.batches = []
        self.last_metrics = {}
        self.active = False

    @contextmanager
    def evaluation(self, **kwargs):
        assert not self.active
        self.active = True
        self.sessions.append(kwargs)
        try:
            yield self
        finally:
            self.active = False
            self.last_metrics = {"rope_semantic_score": 0.9, "rope_examples": len(self.batches)}

    @contextmanager
    def batch(self, **kwargs):
        assert self.active
        self.batches.append(kwargs)
        yield


class FakeTrainer:
    def __init__(self):
        self.model = object()
        self.state = SimpleNamespace(global_step=12, epoch=1.5)
        self.steps = 0
        self.raise_in_step = False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        self.steps += 1
        if self.raise_in_step:
            raise RuntimeError("original prediction failure")
        return inputs["input_ids"]

    def prediction_loop(self, dataloader, description, metric_key_prefix="eval", **kwargs):
        predictions = [self.prediction_step(self.model, inputs, False) for inputs in dataloader]
        return LoopOutput(predictions, None, {f"{metric_key_prefix}_loss": 1.2}, len(predictions))

    def evaluation_loop(self, dataloader, description, metric_key_prefix="eval", **kwargs):
        return self.prediction_loop(dataloader, description, metric_key_prefix=metric_key_prefix)


@pytest.fixture
def fake_monitor(monkeypatch):
    monkeypatch.setattr("rope_profiler.integrations.trainer.RoPEMonitor", FakeMonitor)


def test_attachment_reuses_original_step_handles_nested_loops_and_prefix(fake_monitor):
    trainer = FakeTrainer()
    attachment = attach_rope_monitor(trainer)
    inputs = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    output = trainer.evaluation_loop([inputs, inputs], "Evaluation", "eval_dataset_a")
    assert output.predictions == [inputs["input_ids"], inputs["input_ids"]]
    assert trainer.steps == 2
    assert output.metrics == {
        "eval_dataset_a_loss": 1.2,
        "eval_dataset_a_rope_semantic_score": 0.9,
        "eval_dataset_a_rope_examples": 2,
    }
    assert attachment.monitor.sessions == [{"step": 12, "epoch": 1.5, "prefix": "eval_dataset_a"}]
    assert len(attachment.monitor.batches) == 2
    # Outside evaluation, the original prediction step runs without monitoring.
    trainer.prediction_step(trainer.model, inputs, False)
    assert len(attachment.monitor.batches) == 2
    attachment.detach()
    attachment.detach()
    assert "prediction_step" not in vars(trainer)
    assert "evaluation_loop" not in vars(trainer)
    assert "prediction_loop" not in vars(trainer)


def test_exception_cleanup_preserves_original_failure_and_allows_next_eval(fake_monitor):
    trainer = FakeTrainer()
    attachment = attach_rope_monitor(trainer)
    trainer.raise_in_step = True
    with pytest.raises(RuntimeError, match="original prediction failure"):
        trainer.evaluation_loop([{"input_ids": [1, 2, 3]}], "Evaluation")
    assert not attachment.monitor.active
    assert not attachment._active
    assert attachment._loop_depth == 0
    trainer.raise_in_step = False
    trainer.evaluation_loop([{"input_ids": [1, 2, 3]}], "Evaluation")
    attachment.detach()


def test_detach_restores_instance_override_and_preserves_later_replacement(fake_monitor):
    trainer = FakeTrainer()

    def custom_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return "custom"

    original = MethodType(custom_step, trainer)
    trainer.prediction_step = original
    attachment = attach_rope_monitor(trainer)
    assert trainer.evaluation_loop([{"input_ids": [1]}], "Eval").predictions == ["custom"]
    later_loop = lambda *args, **kwargs: "user replacement"
    trainer.evaluation_loop = later_loop
    attachment.detach()
    assert trainer.prediction_step is original
    assert trainer.evaluation_loop is later_loop


def test_duplicate_attach_rejected(fake_monitor):
    trainer = FakeTrainer()
    attachment = attach_rope_monitor(trainer)
    with pytest.raises(ValueError, match="already"):
        attach_rope_monitor(trainer)
    attachment.detach()
    attach_rope_monitor(trainer).detach()


def test_batch_metadata_is_monitor_only_and_model_inputs_are_unchanged(fake_monitor):
    trainer = FakeTrainer()
    ids, attention, positions = [1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]
    inputs = {"input_ids": ids, "attention_mask": attention, "position_ids": positions}
    calls = []

    def metadata(view):
        calls.append(view)
        with pytest.raises(TypeError):
            view["input_ids"] = []
        assert view["input_ids"] is ids
        return {"query_mask": [False, False, False, True],
                "key_mask": [True, True, False, False], "sample_ids": ["fixed-id"]}

    attachment = attach_rope_monitor(trainer, batch_metadata=metadata)
    output = trainer.evaluation_loop([inputs], "Evaluation")
    assert output.predictions == [ids]
    assert trainer.steps == 1
    assert len(calls) == 1
    assert inputs == {"input_ids": ids, "attention_mask": attention, "position_ids": positions}
    captured = attachment.monitor.batches[0]
    assert captured["sample_ids"] == ["fixed-id"]
    assert captured["query_mask"] == [False, False, False, True]
    assert captured["attention_mask"] is attention
    assert captured["position_ids"] is positions
    attachment.detach()


@pytest.mark.parametrize("returned,exception,match", [
    (None, TypeError, "return a Mapping"),
    ([1, 2], TypeError, "return a Mapping"),
    ({"unknown": 1}, ValueError, "unexpected fields"),
    ({"attention_mask": [1, 0]}, ValueError, "attention_mask"),
    ({"position_ids": [1, 2]}, ValueError, "position_ids"),
])
def test_bad_batch_metadata_cleans_up_session_without_forward(fake_monitor, returned, exception, match):
    trainer = FakeTrainer()
    attachment = attach_rope_monitor(trainer, batch_metadata=lambda inputs: returned)
    with pytest.raises(exception, match=match):
        trainer.evaluation_loop([{"input_ids": [1, 2, 3]}], "Evaluation")
    assert trainer.steps == 0
    assert not attachment.monitor.active
    assert not attachment._active
    assert attachment._loop_depth == 0
    assert attachment.monitor.batches == []
    attachment.detach()


def test_batch_metadata_exception_preserved_and_session_cleaned(fake_monitor):
    trainer = FakeTrainer()

    def metadata(inputs):
        raise RuntimeError("metadata failure")

    attachment = attach_rope_monitor(trainer, batch_metadata=metadata)
    with pytest.raises(RuntimeError, match="metadata failure"):
        trainer.evaluation_loop([{"input_ids": [1, 2, 3]}], "Evaluation")
    assert not attachment.monitor.active
    assert not attachment._active
    assert trainer.steps == 0
    attachment.detach()


def test_batch_metadata_requires_callable_before_wrapping(fake_monitor):
    trainer = FakeTrainer()
    with pytest.raises(TypeError, match="callable"):
        attach_rope_monitor(trainer, batch_metadata={})
    assert "prediction_step" not in vars(trainer)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_real_trainer_same_pass_tiny_models(tmp_path, family, padding_side):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    torch.manual_seed(42)
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    if family == "llama":
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**common))
    else:
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common))
    examples = [torch.arange(length, dtype=torch.long) % 60 + 3 for length in (8, 11, 7, 10)]

    def collate(rows):
        length = max(row.shape[0] for row in rows)
        ids = torch.zeros((len(rows), length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for index, row in enumerate(rows):
            start = length - len(row) if padding_side == "left" else 0
            ids[index, start : start + len(row)] = row
            mask[index, start : start + len(row)] = 1
            labels[index, start : start + len(row)] = row
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
            "labels": labels,
        }

    seen_predictions = []

    def compute_metrics(prediction):
        seen_predictions.append(prediction.predictions.copy())
        return {"custom_accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean())}

    class ExistingTrainerSubclass(transformers.Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_prediction_steps = 0

        def prediction_step(self, *args, **kwargs):
            self.custom_prediction_steps += 1
            return super().prediction_step(*args, **kwargs)

    trainer = ExistingTrainerSubclass(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        ),
        eval_dataset=examples,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    forward_calls = []
    counter = model.register_forward_hook(lambda *args: forward_calls.append(1))
    baseline = trainer.evaluate()
    baseline_predictions = seen_predictions[-1].copy()
    baseline_calls = len(forward_calls)
    forward_calls.clear()
    attachment = attach_rope_monitor(
        trainer,
        config=MonitorConfig(queries=2, keys=4, distance_samples=32, max_examples=4, seed=123),
    )
    result = trainer.evaluate(metric_key_prefix="validation")
    assert len(forward_calls) == baseline_calls == 2
    assert trainer.custom_prediction_steps == 4
    assert result["validation_loss"] == pytest.approx(baseline["eval_loss"], abs=1e-7)
    assert result["validation_custom_accuracy"] == baseline["eval_custom_accuracy"]
    assert result["validation_rope_examples"] == 4
    assert result["validation_rope_semantic_valid_triplets"] > 0
    assert result["validation_rope_positional_valid_pairs"] > 0
    assert len(trainer.state.log_history) == 2
    assert "validation_loss" in trainer.state.log_history[-1]
    assert "validation_rope_semantic_score" in trainer.state.log_history[-1]
    torch.testing.assert_close(torch.from_numpy(seen_predictions[-1]), torch.from_numpy(baseline_predictions), rtol=0, atol=0)
    assert not any(layer.self_attn.q_proj._forward_hooks for layer in model.model.layers)
    assert not any(layer.self_attn.k_proj._forward_hooks for layer in model.model.layers)
    attachment.detach()
    assert trainer.compute_metrics is compute_metrics
    assert "prediction_step" not in vars(trainer)
    counter.remove()


def test_real_trainer_multiple_eval_datasets(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32,
        )
    )
    example = {"input_ids": torch.tensor([2, 3, 4, 5, 6, 7]), "labels": torch.tensor([2, 3, 4, 5, 6, 7])}
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(output_dir=str(tmp_path), report_to=[], disable_tqdm=True, use_cpu=True),
        eval_dataset={"first": [example], "second": [example]},
    )
    attachment = attach_rope_monitor(trainer, config=MonitorConfig(queries=2, keys=4, distance_samples=16))
    metrics = trainer.evaluate()
    assert metrics["eval_first_rope_examples"] == 1
    assert metrics["eval_second_rope_examples"] == 1
    assert sum("rope_semantic_score" in key for key in metrics) == 2
    attachment.detach()


@pytest.mark.parametrize("with_metadata", [True, False])
def test_real_trainer_region_metadata_no_leak_or_extra_forward(tmp_path, with_metadata):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    torch.manual_seed(37)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32,
        )
    )
    rows = []
    for index in range(4):
        ids = torch.arange(12, dtype=torch.long) + index + 2
        rows.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path), report_to=[], disable_tqdm=True,
            use_cpu=True, per_device_eval_batch_size=2, dataloader_pin_memory=False,
        ),
        eval_dataset=rows,
    )
    model_kwargs = []
    counter = model.register_forward_pre_hook(
        lambda module, args, kwargs: model_kwargs.append(set(kwargs)), with_kwargs=True,
    )
    baseline = trainer.evaluate()
    assert len(model_kwargs) == 2
    model_kwargs.clear()
    metadata_calls = []

    def metadata(inputs):
        metadata_calls.append(inputs)
        queries = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        keys = torch.zeros_like(queries)
        queries[:, 6:] = True
        keys[:, :6] = True
        return {"query_mask": queries, "key_mask": keys,
                "sample_ids": [f"sample-{int(row[0])}" for row in inputs["input_ids"]]}

    attachment = attach_rope_monitor(
        trainer,
        config=MonitorConfig(
            sampling_policy="regions", queries=2, keys=4,
            distance_samples=16, max_examples=4, seed=17,
        ),
        batch_metadata=metadata if with_metadata else None,
    )
    if with_metadata:
        metrics = trainer.evaluate()
        assert len(model_kwargs) == len(metadata_calls) == 2
        assert metrics["eval_loss"] == pytest.approx(baseline["eval_loss"], abs=1e-7)
        assert metrics["eval_rope_examples"] == 4
        assert metrics["eval_rope_semantic_valid_triplets"] > 0
        assert metrics["eval_rope_positional_valid_pairs"] > 0
        assert all(not ({"query_mask", "key_mask", "sample_ids"} & names) for names in model_kwargs)
        for plan in attachment.monitor.last_plans:
            assert all(position >= 6 for position in plan.query_positions)
            assert all(position < 6 for position in plan.key_positions)
            assert plan.sample_id.startswith("sample-")
    else:
        with pytest.raises(ValueError, match="regions.*query_mask.*key_mask"):
            trainer.evaluate()
        assert model_kwargs == []
    assert not attachment.monitor.active
    assert not attachment._active
    assert attachment._loop_depth == 0
    assert not model.model.layers[0].self_attn.q_proj._forward_hooks
    assert not model.model.layers[0].self_attn.k_proj._forward_hooks
    attachment.detach()
    counter.remove()
