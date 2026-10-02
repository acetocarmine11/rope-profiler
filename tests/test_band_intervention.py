"""Native-model and independent coefficient checks for repository-only scaling."""
from __future__ import annotations

from dataclasses import replace
import importlib
import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

import research_tools.intervention as intervention_module
from research_tools.intervention import BandScaleIntervention, _scale_query
from rope_profiler import MonitorConfig, RoPEMonitor
from rope_profiler.adapters import ModelAdapter, projection_view


def tiny_model(family="qwen3", backend="sdpa", *, layers=1, **overrides):
    torch.manual_seed(823)
    config_type, model_type = {"llama": (LlamaConfig, LlamaForCausalLM),
                               "qwen3": (Qwen3Config, Qwen3ForCausalLM)}[family]
    options = dict(vocab_size=83, hidden_size=32, intermediate_size=48,
                   num_hidden_layers=layers, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8,
                   max_position_embeddings=128, attention_dropout=0.,
                   pad_token_id=0, bos_token_id=1, eos_token_id=None)
    options.update(overrides)
    config = config_type(**options)
    config._attn_implementation = backend
    return model_type(config).eval()


def data(length=24):
    ids = torch.randint(3, 83, (2, length), generator=torch.Generator().manual_seed(919))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def hook_count(model):
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
               for module in model.modules())


def observe_rotary(monkeypatch, family):
    native = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    original = native.apply_rotary_pos_emb
    captured = []

    def capture(q, k, *args, **kwargs):
        rotated = original(q, k, *args, **kwargs)
        captured.append(tuple(x.detach().clone() for x in (q, k, *rotated)))
        return rotated

    monkeypatch.setattr(native, "apply_rotary_pos_emb", capture)
    return captured


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("length", [4, 24])
def test_native_selected_query_band_coefficients_and_gqa_isolation(family, backend, length, monkeypatch):
    model = tiny_model(family, backend)
    captured = observe_rotary(monkeypatch, family)
    batch = data(length)
    calls = []
    call_hook = model.register_forward_hook(lambda *_: calls.append(1))
    intervention = BandScaleIntervention(model, window=24, threshold=2,
        high_scale=.25, low_scale=1.5, heads={0: [0, 3]})
    before = hook_count(model)
    try:
        with torch.no_grad():
            baseline = model(**batch, use_cache=True)
            with intervention:
                changed = model(**batch, use_cache=True)
        assert calls == [1, 1]  # no additional model forward
        assert hook_count(model) == before
        q, k, qr, kr = captured[0]
        cq, ck, cqr, ckr = captured[1]
        omega = model.model.rotary_emb.inv_freq
        high = omega * 24 >= 2
        assert high.any() and (~high).any()
        coordinate_high = torch.cat((high, high))
        for head in (0, 3):
            expected = q[:, head].clone()
            expected[..., coordinate_high] *= .25
            expected[..., ~coordinate_high] *= 1.5
            torch.testing.assert_close(cq[:, head], expected, rtol=0, atol=0)
        # Both selected heads have an unselected sibling sharing the same K.
        torch.testing.assert_close(cq[:, [1, 2]], q[:, [1, 2]], rtol=0, atol=0)
        torch.testing.assert_close(ck, k, rtol=0, atol=0)
        torch.testing.assert_close(ckr, kr, rtol=0, atol=0)
        # Independent post-RoPE dot products, including cross-token distances:
        # every selected head's score decomposes into scaled high + scaled low.
        for head in (0, 3):
            kv = head // 2
            high_score = qr[:, head, :, coordinate_high] @ kr[:, kv, :, coordinate_high].transpose(-1, -2)
            low_score = qr[:, head, :, ~coordinate_high] @ kr[:, kv, :, ~coordinate_high].transpose(-1, -2)
            actual = cqr[:, head] @ ckr[:, kv].transpose(-1, -2)
            torch.testing.assert_close(actual, .25 * high_score + 1.5 * low_score, rtol=2e-6, atol=2e-6)
            # Symmetric sqrt(alpha) has the same coefficients in real arithmetic.
            factor = torch.where(coordinate_high, math.sqrt(.25), math.sqrt(1.5))
            symmetric = (qr[:, head] * factor) @ (kr[:, kv] * factor).transpose(-1, -2)
            torch.testing.assert_close(actual, symmetric, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(baseline.past_key_values[0][0], changed.past_key_values[0][0], rtol=0, atol=0)
        torch.testing.assert_close(baseline.past_key_values[0][1], changed.past_key_values[0][1], rtol=0, atol=0)
        assert not torch.equal(changed.logits, baseline.logits)
        report = intervention.metadata()
        assert report["layers"]["0"]["high_mask"] == high.tolist()
        assert report["layers"]["0"]["query_forwards"] == 1
        assert report["coefficient_multiplier"] and not report["norm_compensation"]
    finally:
        call_hook.remove()


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_identity_is_hook_free_and_strictly_preserves_logits_generation_and_cache(family, backend):
    model = tiny_model(family, backend, layers=2)
    batch = data(4)
    before = hook_count(model)
    intervention = BandScaleIntervention(model, window=4, threshold=2, heads={1: [2]})
    with torch.inference_mode():
        baseline = model(**batch, use_cache=True)
        baseline_ids = model.generate(**batch, max_new_tokens=3, do_sample=False, use_cache=True)
        with intervention:
            assert hook_count(model) == before
            output = model(**batch, use_cache=True)
            output_ids = model.generate(**batch, max_new_tokens=3, do_sample=False, use_cache=True)
        torch.testing.assert_close(output.logits, baseline.logits, rtol=0, atol=0)
        torch.testing.assert_close(output_ids, baseline_ids, rtol=0, atol=0)
        for layer in range(2):
            for kind in (0, 1):
                torch.testing.assert_close(output.past_key_values[layer][kind], baseline.past_key_values[layer][kind], rtol=0, atol=0)
    assert hook_count(model) == before
    assert intervention.metadata()["layers"]["1"]["native_omega"] is None


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_decode_scales_each_new_query_with_native_cache_and_restores_model(family, backend, monkeypatch):
    model = tiny_model(family, backend)
    captured = observe_rotary(monkeypatch, family)
    batch = data(4)
    intervention = BandScaleIntervention(model, window=24, threshold=2,
                                        high_scale=0, heads={0: [0]})
    with torch.no_grad():
        baseline = model(**batch, use_cache=True)
        with intervention:
            prefill = model(**batch, use_cache=True)
            next_ids = torch.full((2, 1), 3)
            decode = model(input_ids=next_ids, attention_mask=torch.ones((2, 5), dtype=torch.long),
                           past_key_values=prefill.past_key_values, use_cache=True)
        restored = model(**batch, use_cache=True)
    assert [entry[0].shape[-2] for entry in captured] == [4, 4, 1, 4]
    mask = torch.tensor(intervention.metadata()["layers"]["0"]["high_mask"])
    coordinate_high = torch.cat((mask, mask))
    assert torch.count_nonzero(captured[1][0][:, 0, :, coordinate_high]) == 0
    assert torch.count_nonzero(captured[2][0][:, 0, :, coordinate_high]) == 0
    assert decode.past_key_values.get_seq_length() == 5
    assert intervention.metadata()["layers"]["0"]["query_forwards"] == 2
    torch.testing.assert_close(prefill.past_key_values[0][0][..., :4, :], baseline.past_key_values[0][0], rtol=0, atol=0)
    torch.testing.assert_close(prefill.past_key_values[0][1][..., :4, :], baseline.past_key_values[0][1], rtol=0, atol=0)
    torch.testing.assert_close(restored.logits, baseline.logits, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_dynamic_grid_comes_from_first_actual_rotary_forward_then_mask_freezes(family, monkeypatch):
    model = tiny_model(family, max_position_embeddings=8,
                       rope_scaling={"rope_type": "dynamic", "factor": 2.0})
    initial = model.model.rotary_emb.inv_freq.clone()
    # At dimension 1, extending native dynamic RoPE drops below this boundary.
    threshold = float(initial[1]) * 16 * .8
    captured = observe_rotary(monkeypatch, family)
    actual_grids = []
    handle = model.model.rotary_emb.register_forward_hook(
        lambda module, *_: actual_grids.append(module.inv_freq.detach().clone()))
    intervention = BandScaleIntervention(model, window=16, threshold=threshold,
                                        high_scale=.25, heads={0: [0]})
    raw_queries = []
    original_scale = intervention_module._scale_query

    def observe_scale(output, spec, indices, factors, **kwargs):
        raw_queries.append(projection_view(output, spec, "q").transpose(1, 2).detach().clone())
        return original_scale(output, spec, indices, factors, **kwargs)

    monkeypatch.setattr(intervention_module, "_scale_query", observe_scale)
    before = hook_count(model)
    try:
        with torch.no_grad():
            with intervention:
                prefill = model(**data(16), use_cache=True)
                frozen = intervention.metadata()
                model(input_ids=torch.full((2, 1), 3),
                      past_key_values=prefill.past_key_values, use_cache=True)
                # A shorter native forward resets dynamic omega to its original
                # grid. The context still uses its first prompt's band mask.
                model(**data(4))
                after_decode = intervention.metadata()
        assert not torch.equal(actual_grids[0], initial)
        assert not torch.equal(actual_grids[1], actual_grids[0])
        torch.testing.assert_close(actual_grids[2], initial, rtol=0, atol=0)
        first_mask = actual_grids[0] * 16 >= threshold
        assert not torch.equal(first_mask, initial * 16 >= threshold)
        assert frozen["layers"]["0"]["native_omega"] == actual_grids[0].tolist()
        assert after_decode["layers"]["0"]["native_omega"] == actual_grids[0].tolist()
        assert frozen["layers"]["0"]["high_mask"] == first_mask.tolist()
        assert after_decode["layers"]["0"]["high_mask"] == first_mask.tolist()
        coordinates = torch.cat((first_mask, first_mask))
        assert len(captured) == 3
        # Native post-RoPE scaling is still computed by the model for its latest
        # omega; only which planes receive .25 remains frozen at the first grid.
        assert after_decode["layers"]["0"]["query_forwards"] == 3
        assert coordinates.shape[0] == captured[0][0].shape[-1]
        for index in (0, 1, 2):
            expected = raw_queries[index].clone()
            expected[:, 0, :, coordinates] *= .25
            torch.testing.assert_close(captured[index][0], expected, rtol=0, atol=0)
        assert hook_count(model) == before
        with torch.no_grad():
            with intervention:
                model(**data(4))
        assert intervention.metadata()["layers"]["0"]["native_omega"] == initial.tolist()
        assert intervention.metadata()["layers"]["0"]["query_forwards"] == 1
    finally:
        handle.remove()


def test_partial_rotary_tail_zero_and_double_precision_scaling_are_preserved():
    spec = replace(ModelAdapter(tiny_model()).layers[0], rotary_dim=4)
    q = torch.randn(2, 4, 4, 8, dtype=torch.float64)
    original = q.clone()
    alpha = 1.00000000001
    factors = torch.tensor([0, alpha, 0, alpha], dtype=torch.float64)
    result = _scale_query(q, spec, torch.tensor([0, 3]), factors)
    expected = original.clone()
    for head in (0, 3):
        expected[:, :, head, 0] = 0
        expected[:, :, head, 2] = 0
        expected[:, :, head, 1] *= alpha
        expected[:, :, head, 3] *= alpha
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(q, original, rtol=0, atol=0)
    torch.testing.assert_close(result[..., 4:], original[..., 4:], rtol=0, atol=0)
    # Flattened Llama q_proj output takes the same rotary planes.
    flat = q.reshape(2, 4, -1)
    torch.testing.assert_close(_scale_query(flat, spec, torch.tensor([0, 3]), factors),
                               expected.reshape_as(flat), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"window": 0}, {"window": True}, {"window": 2.5},
    {"threshold": 0}, {"threshold": float("nan")}, {"threshold": True},
    {"high_scale": -.1}, {"low_scale": float("inf")}, {"high_scale": True},
    {"heads": {2: [0]}}, {"heads": {0: [4]}}, {"heads": {0: []}},
    {"heads": {0: [0, 0]}}, {"heads": {False: [0]}}, {"heads": {0: [True]}},
    {"heads": {0: "0"}}, {"heads": [0]},
    {"validate_finite": 1},
])
def test_invalid_settings_fail_before_registering_hooks(kwargs):
    model = tiny_model()
    options = {"window": 24, "threshold": 2, "high_scale": .5}
    options.update(kwargs)
    with pytest.raises(ValueError):
        BandScaleIntervention(model, **options)
    assert hook_count(model) == 0


def test_inference_boundary_nested_context_and_exception_cleanup():
    model = tiny_model()
    intervention = BandScaleIntervention(model, window=24, threshold=2, high_scale=.5)
    with pytest.raises(RuntimeError, match="eval.*no_grad"):
        with intervention:
            pass
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match="eval.*no_grad"):
            with intervention:
                pass
        model.eval()
        with pytest.raises(ValueError, match="deliberate inference failure"):
            with intervention:
                with pytest.raises(RuntimeError, match="already active"):
                    with intervention:
                        pass
                with pytest.raises(RuntimeError, match="overlapping"):
                    with BandScaleIntervention(model, window=24, threshold=2, high_scale=.5):
                        pass
                raise ValueError("deliberate inference failure")
        assert hook_count(model) == 0
        with intervention:
            model(**data(4))
        with intervention:
            model.train()
            with pytest.raises(RuntimeError, match="eval.*no_grad"):
                model(**data(4))
            model.eval()
            model(**data(4))
    assert hook_count(model) == 0


def test_cached_old_grid_is_never_used_when_native_rotary_is_bypassed(monkeypatch):
    model = tiny_model()
    intervention = BandScaleIntervention(model, window=24, threshold=2, high_scale=.5)
    original = model.model.rotary_emb.forward
    with torch.no_grad():
        with intervention:
            model(**data(4))
            # Mimic an unsupported architecture that constructs cos/sin outside
            # the module, bypassing its runtime update and all module hooks.
            monkeypatch.setattr(model.model.rotary_emb, "__class__", type(
                "BypassRotary", (type(model.model.rotary_emb),),
                {"__call__": lambda self, *args, **kwargs: original(*args, **kwargs)}))
            with pytest.raises(RuntimeError, match="unsupported forward order"):
                model(**data(4))
    assert hook_count(model) == 0


def test_empty_selection_is_noop_and_user_mapping_is_copied():
    model = tiny_model()
    heads = {0: [3, 0]}
    selected = BandScaleIntervention(model, window=24, threshold=2, high_scale=.5, heads=heads)
    heads[0].clear()
    assert selected.heads == {0: (3, 0)}
    noop = BandScaleIntervention(model, window=24, threshold=2, high_scale=0, heads={})
    with torch.no_grad():
        baseline = model(**data(4)).logits
        with noop:
            assert hook_count(model) == 0
            actual = model(**data(4)).logits
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert noop.metadata()["identity"]


@pytest.mark.parametrize("field", ["high_scale", "low_scale"])
def test_finite_python_multiplier_outside_q_dtype_range_fails_and_restores_hooks(field):
    model = tiny_model()
    options = {"high_scale": .5, field: 1e100}
    intervention = BandScaleIntervention(model, window=24, threshold=2, **options)
    with torch.no_grad():
        baseline = model(**data(4)).logits
        with pytest.raises(ValueError, match="finite range of Q dtype torch.float32"):
            with intervention:
                model(**data(4))
        assert hook_count(model) == 0
        restored = model(**data(4)).logits
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("validate_finite", [False, True])
def test_cpu_detects_actual_product_overflow_with_representable_factor_and_restores_model(validate_finite):
    model = tiny_model("qwen3")
    scale = 2e38
    assert scale < torch.finfo(torch.float32).max
    intervention = BandScaleIntervention(model, window=24, threshold=2,
        high_scale=scale, heads={0: [0]}, validate_finite=validate_finite)
    with torch.no_grad():
        baseline = model(**data(24), use_cache=True)
        original_keys = baseline.past_key_values[0][0].clone()
        original_values = baseline.past_key_values[0][1].clone()
        with pytest.raises(ValueError, match="band scaling produced nonfinite selected Q"):
            with intervention:
                model(input_ids=data(24)["input_ids"],
                      attention_mask=torch.ones((2, 48), dtype=torch.long),
                      past_key_values=baseline.past_key_values, use_cache=True)
        assert hook_count(model) == 0
        # Failure happens before native K/V cache update, and scaling never
        # mutates the original Q projection output or unselected heads.
        assert baseline.past_key_values.get_seq_length() == 24
        torch.testing.assert_close(baseline.past_key_values[0][0], original_keys, rtol=0, atol=0)
        torch.testing.assert_close(baseline.past_key_values[0][1], original_values, rtol=0, atol=0)
        restored = model(**data(24)).logits
    torch.testing.assert_close(restored, baseline.logits, rtol=0, atol=0)
    report = intervention.metadata()
    assert report["validate_finite"] is validate_finite
    assert report["finite_validation"]["cpu"] == "always_for_nonidentity"


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_monitor_observes_effective_q_in_both_context_nesting_orders(family, backend, monkeypatch):
    import rope_profiler.monitor as monitor_module

    model = tiny_model(family, backend)
    batch = data(24)
    config = MonitorConfig(queries=3, keys=5, distance_samples=16,
                           heads={0: [0]}, head_preset=None)
    captured = []
    original_score = monitor_module.score_activations
    native_q = observe_rotary(monkeypatch, family)

    def observe_score(q, k, *args, **kwargs):
        captured.append((q.clone(), k.clone()))
        return original_score(q, k, *args, **kwargs)

    monkeypatch.setattr(monitor_module, "score_activations", observe_score)

    def evaluate(intervention_first):
        monitor = RoPEMonitor(model, config)
        intervention = BandScaleIntervention(model, window=24, threshold=2,
                                            high_scale=.25, heads={0: [0]})
        with torch.no_grad():
            with monitor.evaluation():
                if intervention_first:
                    with intervention:
                        with monitor.batch(**batch):
                            output = model(**batch).logits
                else:
                    with monitor.batch(**batch):
                        with intervention:
                            output = model(**batch).logits
        assert hook_count(model) == 0
        return output, monitor

    first_logits, first = evaluate(True)
    second_logits, second = evaluate(False)
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
    assert first.last_plans == second.last_plans
    # Elapsed score time measures each run and is expected to differ.
    assert {key: value for key, value in first.last_metrics.items() if key != "rope_score_seconds"} == {
        key: value for key, value in second.last_metrics.items() if key != "rope_score_seconds"}
    assert len(captured) == 4 and len(native_q) == 2
    for row, plan in enumerate(first.last_plans):
        q1, k1 = captured[row]
        q2, k2 = captured[row + 2]
        torch.testing.assert_close(q1, q2, rtol=0, atol=0)
        torch.testing.assert_close(k1, k2, rtol=0, atol=0)
        # Verify against the actual tensors entering native RoPE, independently
        # of monitor registration order and its scalar aggregation.
        expected_q = native_q[0][0][row, [0]][:, list(plan.query_positions)]
        torch.testing.assert_close(q1, expected_q, rtol=0, atol=0)
    assert math.isfinite(first.last_metrics["rope_positional_score"])
