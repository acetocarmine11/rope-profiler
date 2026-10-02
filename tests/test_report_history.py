"""Optional report history and evaluation/Trainer lifecycle regressions."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from rope_profiler import MonitorConfig, RoPEMonitor
from rope_profiler.integrations.trainer import attach_rope_monitor


def tiny_model():
    torch.manual_seed(53)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8,
                         max_position_embeddings=64, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval()


def inputs():
    ids = torch.arange(3, 15).unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def evaluate(monitor, *, step=7, prefix="eval"):
    data = inputs()
    with monitor.evaluation(step=step, epoch=0.5, prefix=prefix):
        with monitor.batch(**data, sample_ids=["item-1"]):
            with torch.no_grad():
                monitor.model(**data)


def test_repeated_step_evaluations_keep_independent_complete_reports(tmp_path):
    directory = tmp_path / "history"
    config = MonitorConfig(layers=[0], heads={0: [1]}, distance_samples=4)
    monitor = RoPEMonitor(tiny_model(), config, report_dir=directory)
    assert not directory.exists() and monitor.last_report_path is None
    evaluate(monitor, prefix="eval/dataset")
    first_path = monitor.last_report_path
    first_bytes = first_path.read_bytes()
    evaluate(monitor, prefix="eval/dataset")
    second_path = monitor.last_report_path
    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert set(directory.glob("*.json")) == {first_path, second_path}
    for path in (first_path, second_path):
        saved = json.loads(path.read_text())
        assert MonitorConfig.from_dict(saved["config"]) == config
        assert saved["evaluation"]["step"] == 7
        assert saved["evaluation"]["prefix"] == "eval/dataset"
        assert saved["metrics"]["rope_examples"] == 1
        assert saved["frequency_grids"] and saved["sampling_trace"]
        assert "input_ids" not in path.read_text()
        assert json.dumps(inputs()["input_ids"][0].tolist()) not in path.read_text()
    assert not list(directory.glob("*.tmp"))


def test_new_evaluation_clears_report_path_and_budget(tmp_path):
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(max_examples=1, distance_samples=4),
                          report_dir=tmp_path)
    evaluate(monitor)
    previous = monitor.last_report_path
    data = inputs()
    with monitor.evaluation(step=8):
        assert monitor.last_report_path is None and monitor.last_report == {}
        with monitor.batch(**data):
            with torch.no_grad():
                monitor.model(**data)
    assert monitor.last_report_path != previous
    assert monitor.last_metrics["rope_examples"] == 1


def test_default_report_history_does_not_write(monkeypatch):
    def unexpected_write(*args, **kwargs):
        raise AssertionError("default report history attempted file I/O")
    monkeypatch.setattr("rope_profiler.monitor._atomic_report_write", unexpected_write)
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4))
    evaluate(monitor)
    assert monitor.last_report_path is None
    assert monitor.last_report and monitor.last_metrics["rope_examples"] == 1


@pytest.mark.parametrize("prefix", ["../../escape", "/absolute/path", "..\\..\\escape",
                                   "\x00../escape", "", "..", "\u8bc4\u4f30/\u6570\u636e"])
def test_prefix_cannot_escape_report_directory(tmp_path, prefix):
    directory = tmp_path / "history"
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4), report_dir=directory)
    evaluate(monitor, prefix=prefix)
    path = monitor.last_report_path
    assert path.parent == directory
    assert path.resolve().parent == directory.resolve()
    assert all(character.isascii() and (character.isalnum() or character in "-_.")
               for character in path.name)
    assert json.loads(path.read_text())["evaluation"]["prefix"] == prefix
    assert list(tmp_path.rglob("*.json")) == [path]


def test_report_filename_contains_distributed_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: None)
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4), report_dir=tmp_path)
    evaluate(monitor)
    assert "-rank-3-" in monitor.last_report_path.name
    assert json.loads(monitor.last_report_path.read_text())["evaluation"]["rank"] == 3


def test_history_collision_never_overwrites_a_previous_report(tmp_path, monkeypatch):
    identifiers = iter(["same", "same", "different"])
    monkeypatch.setattr("rope_profiler.monitor.uuid4", lambda: SimpleNamespace(hex=next(identifiers)))
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4), report_dir=tmp_path)
    evaluate(monitor)
    first_path, first_content = monitor.last_report_path, monitor.last_report_path.read_bytes()
    evaluate(monitor)
    assert monitor.last_report_path != first_path
    assert first_path.read_bytes() == first_content
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_logger_failure_does_not_create_report_history(tmp_path):
    directory = tmp_path / "history"
    class BrokenLogger:
        def log(self, *args, **kwargs):
            assert not directory.exists()
            raise RuntimeError("logger failure")
    model = tiny_model()
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=4), logger=BrokenLogger(),
                          report_dir=directory)
    with pytest.raises(RuntimeError, match="logger failure"):
        evaluate(monitor)
    assert not directory.exists()
    assert monitor.last_report_path is None and monitor.last_report == {} and monitor.last_metrics == {}
    assert not monitor.active and not any(module._forward_hooks for module in model.modules())


def test_save_failure_cleans_results_and_hooks_and_allows_next_eval(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("keep")
    model = tiny_model()
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=4), report_dir=blocked / "history")
    with pytest.raises(RuntimeError, match="failed to save RoPE evaluation report"):
        evaluate(monitor)
    assert blocked.read_text() == "keep"
    assert monitor.last_report_path is None and monitor.last_report == {} and monitor.last_metrics == {}
    assert monitor.last_plans == () and not monitor.active
    assert not any(module._forward_hooks for module in model.modules())
    monitor.report_dir = tmp_path / "valid"
    evaluate(monitor)
    assert monitor.last_report_path.is_file()


def test_forward_failure_does_not_save_or_reuse_previous_path(tmp_path):
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4), report_dir=tmp_path)
    evaluate(monitor)
    previous = monitor.last_report_path
    with pytest.raises(RuntimeError, match="forward failure"):
        with monitor.evaluation(step=8):
            with monitor.batch(**inputs()):
                raise RuntimeError("forward failure")
    assert monitor.last_report_path is None and monitor.last_report == {}
    assert set(tmp_path.glob("*.json")) == {previous}
    assert not monitor.active and not any(module._forward_hooks for module in monitor.model.modules())


def test_manual_save_report_keeps_interface_and_atomic_replace(tmp_path, monkeypatch):
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4))
    evaluate(monitor)
    target = monitor.save_report(tmp_path / "manual.json")
    assert target == tmp_path / "manual.json"
    before = target.read_bytes()
    def fail_replace(*args, **kwargs):
        raise PermissionError("replacement denied")
    monkeypatch.setattr("rope_profiler.monitor.os.replace", fail_replace)
    with pytest.raises(PermissionError, match="replacement denied"):
        monitor.save_report(target)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_evaluation_context_is_single_use_without_corrupting_history(tmp_path):
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(max_examples=1, distance_samples=4),
                          report_dir=tmp_path)
    context = monitor.evaluation(step=7)
    with context:
        with monitor.batch(**inputs()):
            with torch.no_grad():
                monitor.model(**inputs())
    expected = deepcopy(monitor.last_report)
    path = monitor.last_report_path
    with pytest.raises(RuntimeError, match="evaluation context is single-use"):
        with context:
            pass
    assert not monitor.active and monitor.last_report == expected and monitor.last_report_path == path
    evaluate(monitor)
    assert monitor.last_metrics["rope_examples"] == 1


def test_batch_context_is_single_use():
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4))
    with monitor.evaluation():
        batch = monitor.batch(**inputs())
        with batch:
            with torch.no_grad():
                monitor.model(**inputs())
        with pytest.raises(RuntimeError, match="batch context is single-use"):
            with batch:
                pass
    assert monitor.last_metrics["rope_examples"] == 1
    assert len(monitor.last_report["sampling_trace"]) == 1


def test_batch_context_cannot_cross_evaluation_sessions():
    monitor = RoPEMonitor(tiny_model(), MonitorConfig(distance_samples=4))
    with monitor.evaluation():
        pending = monitor.batch(**inputs())
    with monitor.evaluation():
        with pytest.raises(RuntimeError, match="evaluation that created it"):
            with pending:
                pass
    assert not monitor.active


def test_real_trainer_model_init_rebind_keeps_history_subclass_and_forward_count(tmp_path):
    created, forward_modes, metric_calls = [], [], []
    def model_init():
        model = tiny_model()
        model.register_forward_hook(lambda current, *args: forward_modes.append(current.training))
        created.append(model)
        return model
    def compute_metrics(prediction):
        metric_calls.append(prediction)
        return {"custom_accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean())}
    class ExistingTrainer(Trainer):
        prediction_calls = 0
        def prediction_step(self, *args, **kwargs):
            self.prediction_calls += 1
            return super().prediction_step(*args, **kwargs)
    row = {name: value[0] for name, value in inputs().items()}
    row["labels"] = row["input_ids"].clone()
    trainer = ExistingTrainer(model_init=model_init,
        args=TrainingArguments(output_dir=str(tmp_path / "trainer"), max_steps=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            eval_strategy="steps", eval_steps=1, save_strategy="no", report_to=[],
            use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False),
        train_dataset=[row], eval_dataset=[row], compute_metrics=compute_metrics)
    config = MonitorConfig(layers=[0], heads={0: [1]}, max_examples=1, distance_samples=4)
    directory = tmp_path / "history"
    attachment = attach_rope_monitor(trainer, config=config, report_dir=directory)
    before = trainer.evaluate()
    old_monitor = attachment.monitor
    old_path, old_content = old_monitor.last_report_path, old_monitor.last_report_path.read_bytes()
    trainer.train()
    assert len(created) == 2 and attachment.monitor is not old_monitor
    assert attachment.monitor.model is trainer.model
    assert attachment.monitor.config == config and attachment.monitor.report_dir == directory
    after = trainer.evaluate()
    assert "eval_custom_accuracy" in before and "eval_custom_accuracy" in after
    assert after["eval_rope_examples"] == 1
    assert trainer.prediction_calls == 3 and len(metric_calls) == 3
    assert forward_modes == [False, True, False, False]
    assert old_path.read_bytes() == old_content
    assert len(list(directory.glob("*.json"))) == 3
    assert attachment.monitor.last_report_path != old_path
    attachment.detach()
    assert "prediction_step" not in vars(trainer) and "evaluation_loop" not in vars(trainer)
