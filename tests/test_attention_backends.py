"""Native eager/SDPA capture against the actual inputs to RoPE, without downloads."""
from __future__ import annotations

import importlib

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from rope_profiler import MonitorConfig, RoPEMonitor


def model_fixture(family, backend, **config_overrides):
    torch.manual_seed(293)
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_type(
        vocab_size=83, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        **config_overrides,
    )
    config._attn_implementation = backend
    return model_type(config).eval()


def data_fixture(length):
    generator = torch.Generator().manual_seed(379)
    ids = torch.randint(3, 83, (2, length), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def evaluate(model, data):
    monitor = RoPEMonitor(model, MonitorConfig(
        queries=3, keys=5, distance_samples=16, heads={0: [3, 0], 1: [2, 3]},
    ))
    with monitor.evaluation():
        with monitor.batch(**data):
            with torch.no_grad():
                logits = model(**data).logits
    return logits, monitor


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("length", [4, 24])
def test_native_backend_preserves_forward_and_captures_actual_pre_rope_gqa(
    family, backend, length, monkeypatch,
):
    import rope_profiler.monitor as monitor_module

    native = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    model = model_fixture(family, backend)
    data = data_fixture(length)
    rotary_original = native.apply_rotary_pos_emb
    pre_rope = []

    def observe_rope(q, k, *args, **kwargs):
        # These are the tensors the model will actually rotate, after native
        # normalization and head reshape. No assumed norm/module layout is used.
        pre_rope.append((q.detach().clone(), k.detach().clone()))
        return rotary_original(q, k, *args, **kwargs)

    monkeypatch.setattr(native, "apply_rotary_pos_emb", observe_rope)
    backend_calls = []
    backend_original = (native.eager_attention_forward if backend == "eager"
                        else native.ALL_ATTENTION_FUNCTIONS["sdpa"])

    def observe_backend(*args, **kwargs):
        backend_calls.append(backend)
        return backend_original(*args, **kwargs)

    if backend == "eager":
        monkeypatch.setattr(native, "eager_attention_forward", observe_backend)
    else:
        monkeypatch.setitem(native.ALL_ATTENTION_FUNCTIONS, "sdpa", observe_backend)

    forwards = []
    handle = model.register_forward_hook(lambda *_: forwards.append(1))
    try:
        with torch.no_grad():
            baseline = model(**data).logits
        assert len(forwards) == 1 and len(backend_calls) == 2
        pre_rope.clear()
        original_score = monitor_module.score_activations
        captures = []

        def observe_score(q, k, mapping, pairs, triplets, omega, window, **kwargs):
            assert q.grad_fn is None and k.grad_fn is None
            assert not q.requires_grad and not k.requires_grad
            result = original_score(q, k, mapping, pairs, triplets, omega, window, **kwargs)
            captures.append((q.clone(), k.clone(), tuple(mapping), result,
                             omega.clone(), window, dict(kwargs)))
            return result

        monkeypatch.setattr(monitor_module, "score_activations", observe_score)
        logits, monitor = evaluate(model, data)
        assert len(forwards) == 2
        assert len(backend_calls) == 4 and all(call == backend for call in backend_calls)
        assert model.config._attn_implementation == backend
        torch.testing.assert_close(logits, baseline, rtol=0, atol=0)
        assert len(pre_rope) == 2 and len(captures) == 4
        assert len(monitor.last_plans) == 2
        for layer, (native_q, native_k) in enumerate(pre_rope):
            heads = [3, 0] if layer == 0 else [2, 3]
            required_kv = sorted({head // 2 for head in heads})
            for row, plan in enumerate(monitor.last_plans):
                q, k, mapping, actual, omega, window, kwargs = captures[layer * 2 + row]
                expected_q = native_q[row, heads][:, list(plan.query_positions)]
                expected_k = native_k[row, required_kv][:, list(plan.key_positions)]
                torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
                torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
                assert mapping == tuple(required_kv.index(head // 2) for head in heads)
                # Independent GQA reference retains every native KV head and
                # uses global head indices instead of the monitor's local map.
                reference = original_score(
                    expected_q, native_k[row, :, list(plan.key_positions)],
                    [head // 2 for head in heads], plan.pair_indices,
                    plan.triplet_indices, omega, window, **kwargs,
                )
                torch.testing.assert_close(actual.semantic_by_head, reference.semantic_by_head, rtol=0, atol=0)
                torch.testing.assert_close(actual.positional_by_head, reference.positional_by_head, rtol=0, atol=0)
                assert window == length
        assert torch.isfinite(torch.tensor([
            monitor.last_metrics["rope_semantic_score"],
            monitor.last_metrics["rope_positional_score"],
        ])).all()
        # length=4 also exercises native Qwen3's valid [B,T,H,D] normalization
        # output when the token and query-head dimensions happen to be equal.
        if family == "qwen3":
            for spec in monitor.adapter.layers:
                assert spec.query_module is spec.attention.q_norm
                assert spec.key_module is spec.attention.k_norm
    finally:
        handle.remove()


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_same_weights_eager_and_sdpa_keep_logits_and_raw_scores_close(family):
    sdpa = model_fixture(family, "sdpa")
    eager = model_fixture(family, "eager")
    eager.load_state_dict(sdpa.state_dict())
    data = data_fixture(24)
    sdpa_logits, sdpa_monitor = evaluate(sdpa, data)
    eager_logits, eager_monitor = evaluate(eager, data)
    torch.testing.assert_close(eager_logits, sdpa_logits, rtol=2e-5, atol=1e-6)
    assert eager_monitor.last_plans == sdpa_monitor.last_plans
    for name in ("rope_semantic_score", "rope_positional_score"):
        assert eager_monitor.last_metrics[name] == pytest.approx(sdpa_monitor.last_metrics[name], abs=3e-6)
    for name in ("rope_semantic_valid_triplets", "rope_positional_valid_pairs", "rope_invalid_cells"):
        assert eager_monitor.last_metrics[name] == sdpa_monitor.last_metrics[name]


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("first_sliding_layer", [0, 1])
def test_native_qwen3_sliding_attention_requires_a_window_sampling_adapter(
    backend, first_sliding_layer,
):
    # These public Qwen3 configuration fields enable real sliding attention in
    # both Transformers 4.51 and 4.56. The latter derives layer_types, whereas
    # the former sets each attention module's window from its layer index.
    model = model_fixture(
        "qwen3", backend, use_sliding_window=True,
        sliding_window=4, max_window_layers=first_sliding_layer,
    )
    active = [index for index, layer in enumerate(model.model.layers)
              if layer.self_attn.sliding_window is not None]
    assert active == list(range(first_sliding_layer, 2))
    assert all(model.model.layers[index].self_attn.sliding_window == 4 for index in active)
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    try:
        hooks_before = sum(len(module._forward_hooks) for module in model.modules())
        with pytest.raises(ValueError, match="full causal attention.*sliding-window.*window-aware sampling adapter"):
            RoPEMonitor(model)
        assert calls == []
        assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before
    finally:
        handle.remove()
