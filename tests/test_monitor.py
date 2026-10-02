"""Integration checks against real, randomly initialized Transformers models.

These fixtures require no checkpoint download and exercise the actual attention
implementation, including Qwen3's Q/K normalization and grouped-query heads.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from rope_profiler import MonitorConfig, RoPEMonitor
from rope_profiler.sampling import sample_tokens


def tiny_model(family: str = "llama"):
    torch.manual_seed(71)
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_type(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return model_type(config).eval()


def inputs(batch_size=2, length=32):
    generator = torch.Generator().manual_seed(91)
    token_ids = torch.randint(3, 97, (batch_size, length), generator=generator)
    return {"input_ids": token_ids, "attention_mask": torch.ones_like(token_ids)}


def monitored(model, data, config=None, *, step=0, sample_ids=None, logger=None):
    monitor = RoPEMonitor(model, config=config or MonitorConfig(), logger=logger)
    with monitor.evaluation(step=step):
        with monitor.batch(**data, sample_ids=sample_ids):
            with torch.no_grad():
                result = model(**data)
    return result, monitor


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_monitor_preserves_logits_and_forward_count(family):
    model = tiny_model(family)
    data = inputs()
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        baseline = model(**data).logits
    assert len(calls) == 1
    result, monitor = monitored(model, data, MonitorConfig(distance_samples=16))
    assert len(calls) == 2  # Exactly the one eval call made by the caller.
    torch.testing.assert_close(result.logits, baseline, rtol=0, atol=0)
    assert math.isfinite(monitor.last_metrics["rope_semantic_score"])
    assert math.isfinite(monitor.last_metrics["rope_positional_score"])
    handle.remove()


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_padding_and_batch_partition_preserve_scores(family):
    model = tiny_model(family)
    tokens = inputs(batch_size=1, length=24)["input_ids"]
    mask = torch.ones_like(tokens)
    pad = torch.zeros((1, 8), dtype=torch.long)
    config = MonitorConfig(distance_samples=16)
    plain, plain_monitor = monitored(model, {"input_ids": tokens, "attention_mask": mask}, config)
    for left in (False, True):
        padded = torch.cat((pad, tokens) if left else (tokens, pad), dim=1)
        attention = torch.cat((pad, mask) if left else (mask, pad), dim=1)
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        _, monitor = monitored(model, {"input_ids": padded, "attention_mask": attention,
                                       "position_ids": positions}, config)
        for key in ("rope_semantic_score", "rope_positional_score"):
            assert monitor.last_metrics[key] == pytest.approx(plain_monitor.last_metrics[key], abs=1e-7)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_capture_is_post_norm_pre_rope_and_detached(family, monkeypatch):
    import rope_profiler.monitor as monitor_module

    model = tiny_model(family)
    data = inputs(batch_size=1)
    full = {}
    handles = []
    for layer, decoder in enumerate(model.model.layers):
        attention = decoder.self_attn
        for kind in ("q", "k"):
            module = getattr(attention, f"{kind}_norm", None) or getattr(attention, f"{kind}_proj")
            def save(_module, _args, output, *, layer=layer, kind=kind):
                full[layer, kind] = output.detach().clone()
            handles.append(module.register_forward_hook(save))
    original_score = monitor_module.score_activations
    captured = []
    def spy(q, k, *args, **kwargs):
        assert q.grad_fn is None and k.grad_fn is None
        assert not q.requires_grad and not k.requires_grad
        captured.append((q.detach().clone(), k.detach().clone()))
        return original_score(q, k, *args, **kwargs)
    monkeypatch.setattr(monitor_module, "score_activations", spy)
    _, monitor = monitored(model, data, MonitorConfig(distance_samples=16))
    plan = sample_tokens(data["input_ids"][0].tolist(), [True] * 32)
    assert plan is not None
    assert len(captured) == 2
    for layer, (q, k) in enumerate(captured):
        for kind, sparse, selected, heads in (
            ("q", q, plan.query_positions, 4),
            ("k", k, plan.key_positions, 2),
        ):
            full_tensor = full[layer, kind].reshape(1, 32, heads, 8)
            expected = full_tensor[0, list(selected)].transpose(0, 1)
            torch.testing.assert_close(sparse, expected, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_fixed_sampling_survives_checkpoint_and_batch_changes():
    model = tiny_model()
    data = inputs(batch_size=2)
    monitor = RoPEMonitor(model, config=MonitorConfig(distance_samples=16))
    with monitor.evaluation(step=0):
        with monitor.batch(**data):
            with torch.no_grad():
                model(**data)
    first = dict(monitor.last_metrics)
    first_plans = monitor.last_plans
    with monitor.evaluation(step=1):
        for row in range(2):
            one = {key: value[row:row + 1] for key, value in data.items()}
            with monitor.batch(**one):
                with torch.no_grad():
                    model(**one)
    for key in ("rope_semantic_score", "rope_positional_score"):
        assert monitor.last_metrics[key] == pytest.approx(first[key], abs=1e-7)
    assert monitor.last_plans == first_plans
    # Simulate a new training checkpoint without changing eval examples.
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(0.01)
    with monitor.evaluation(step=2):
        with monitor.batch(**data):
            with torch.no_grad():
                model(**data)
    assert monitor.last_plans == first_plans


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_generate_scores_prefill_only(family):
    model = tiny_model(family)
    data = inputs(batch_size=1)
    _, reference = monitored(model, data, MonitorConfig(distance_samples=16))
    monitor = RoPEMonitor(model, config=MonitorConfig(distance_samples=16))
    with monitor.evaluation(step=0):
        with monitor.batch(**data):
            with torch.no_grad():
                generated = model.generate(**data, max_new_tokens=3, do_sample=False,
                                           use_model_defaults=False, eos_token_id=None)
    assert generated.shape[1] == 35
    for key in ("rope_semantic_score", "rope_positional_score"):
        assert monitor.last_metrics[key] == pytest.approx(reference.last_metrics[key], abs=1e-9)


def test_exception_removes_hooks_and_next_evaluation_works():
    model = tiny_model()
    data = inputs(batch_size=1)
    monitor = RoPEMonitor(model, config=MonitorConfig(distance_samples=16))
    hooks_before = sum(len(module._forward_hooks) for module in model.modules())
    with pytest.raises(RuntimeError, match="caller failure"):
        with monitor.evaluation(step=0):
            with monitor.batch(**data):
                with torch.no_grad():
                    model(**data)
                raise RuntimeError("caller failure")
    assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before
    with monitor.evaluation(step=1):
        with monitor.batch(**data):
            with torch.no_grad():
                model(**data)
    assert math.isfinite(monitor.last_metrics["rope_semantic_score"])


def test_max_examples_limits_monitor_without_changing_predictions():
    model = tiny_model()
    data = inputs(batch_size=3)
    _, limited = monitored(model, data, MonitorConfig(max_examples=1, distance_samples=16))
    first = {key: value[:1] for key, value in data.items()}
    _, reference = monitored(model, first, MonitorConfig(distance_samples=16))
    for key in ("rope_semantic_score", "rope_positional_score"):
        assert limited.last_metrics[key] == pytest.approx(reference.last_metrics[key], abs=1e-7)


def test_monitor_does_not_cut_off_model_gradient():
    # eval() controls dropout, while autograd remains enabled independently.
    model = tiny_model()
    data = inputs(batch_size=1)
    monitor = RoPEMonitor(model, config=MonitorConfig(distance_samples=16))
    with monitor.evaluation(step=0):
        with monitor.batch(**data):
            output = model(**data, labels=data["input_ids"])
            output.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()


def test_rejects_packed_positions_before_forward():
    model = tiny_model()
    data = inputs(batch_size=1)
    data["position_ids"] = torch.arange(32).unsqueeze(0) % 16
    monitor = RoPEMonitor(model)
    with pytest.raises(ValueError, match="packed|position"):
        with monitor.evaluation(step=0):
            with monitor.batch(**data):
                model(**data)


def test_dynamic_rope_uses_frequency_grid_of_actual_eval_forward(monkeypatch):
    import rope_profiler.monitor as monitor_module

    config = tiny_model().config
    config.max_position_embeddings = 32
    config.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
    model = LlamaForCausalLM(config).eval()
    original_grid = model.model.rotary_emb.inv_freq.clone()
    seen_grids = []
    original_score = monitor_module.score_activations
    def spy(q, k, mapping, pairs, triplets, omega, window, **kwargs):
        seen_grids.append(omega.detach().clone())
        return original_score(q, k, mapping, pairs, triplets, omega, window, **kwargs)
    monkeypatch.setattr(monitor_module, "score_activations", spy)
    _, monitor = monitored(model, inputs(batch_size=1, length=64), MonitorConfig(distance_samples=16))
    runtime_grid = model.model.rotary_emb.inv_freq
    assert not torch.equal(runtime_grid, original_grid)
    assert len(seen_grids) == 2
    for grid in seen_grids:
        torch.testing.assert_close(grid, runtime_grid, rtol=0, atol=0)
    expected = runtime_grid * 64 >= 2 * math.pi
    for mask in monitor.frequency_bands(64, threshold=2 * math.pi).values():
        assert torch.equal(mask, expected)


def _distributed_worker(rank, init_path, output_path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://" + init_path,
                                         rank=rank, world_size=2)
    try:
        model = tiny_model()
        combined = inputs(batch_size=3)
        selected = slice(0, 1) if rank == 0 else slice(1, 3)
        data = {key: value[selected] for key, value in combined.items()}
        _, monitor = monitored(model, data, MonitorConfig(distance_samples=16))
        Path(output_path, f"rank-{rank}.json").write_text(json.dumps(monitor.last_metrics))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="CPU distributed Gloo is unavailable")
def test_distributed_aggregation_weights_examples_not_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,
                                args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    _, reference = monitored(tiny_model(), inputs(batch_size=3), MonitorConfig(distance_samples=16))
    for rank in range(2):
        actual = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert actual["rope_examples"] == 3
        for key in ("rope_semantic_score", "rope_positional_score"):
            assert actual[key] == pytest.approx(reference.last_metrics[key], abs=1e-7)
