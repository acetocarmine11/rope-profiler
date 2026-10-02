"""Report reproducibility and native sparse-head capture checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from rope_profiler import MonitorConfig, RoPEMonitor
from rope_profiler.sampling import input_fingerprint


def model_fixture(family="llama", *, theta=10000, scaling=None):
    torch.manual_seed(317)
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_type(vocab_size=97, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=2, num_attention_heads=8,
                         num_key_value_heads=4, head_dim=8,
                         max_position_embeddings=4096, rope_theta=theta,
                         rope_scaling=scaling, attention_dropout=0.0,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "sdpa"
    return model_type(config).eval()


def data_fixture():
    ids = torch.arange(3, 35).unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def evaluate(monitor, data, **metadata):
    with monitor.evaluation(step=17, epoch=1.25, prefix="validation"):
        with monitor.batch(**data, **metadata):
            with torch.no_grad():
                return monitor.model(**data).logits


def recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from recursive_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from recursive_keys(item)


def test_report_contains_recoverable_config_runtime_grid_and_sampling_hashes(tmp_path):
    model = model_fixture()
    data = data_fixture()
    query_mask = torch.zeros_like(data["input_ids"], dtype=torch.bool)
    query_mask[:, 24:] = True
    key_mask = torch.zeros_like(query_mask)
    key_mask[:, :24] = True
    config = MonitorConfig(sampling_policy="regions", seed=7, layers=[1],
                           heads={1: [5, 0, 7]}, distance_samples=16)
    monitor = RoPEMonitor(model, config)
    evaluate(monitor, data, sample_ids=["example-1"], query_mask=query_mask, key_mask=key_mask)
    target = monitor.save_report(tmp_path / "nested" / "report.json")
    saved = json.loads(target.read_text())
    assert MonitorConfig.from_dict(saved["config"]) == config
    assert saved["runtime"]["torch"] == torch.__version__
    assert saved["runtime"]["python"]
    assert saved["runtime"]["transformers"]
    assert saved["evaluation"] == {"step": 17, "epoch": 1.25, "prefix": "validation", "rank": 0, "world_size": 1}
    assert saved["metrics"]["rope_examples"] == 1
    assert saved["metadata"]["sampling_policy"] == "regions"
    assert saved["frequency_grids"]
    torch.testing.assert_close(torch.tensor(saved["frequency_grids"][0]["inv_freq"]),
                               model.model.rotary_emb.inv_freq, rtol=0, atol=0)
    assert saved["frequency_uses"][0]["layer"] == 1
    assert saved["frequency_uses"][0]["sample_indices"] == [0]
    trace = saved["sampling_trace"][0]
    assert trace["sample_id"] == "example-1" and trace["window"] == 32
    assert trace["query_positions"] and trace["key_positions"]
    assert trace["pair_indices"] and trace["triplet_indices"]
    assert all(query_mask[0, index] for index in trace["query_positions"])
    assert all(key_mask[0, index] for index in trace["key_positions"])
    assert trace["input_sha256"] == input_fingerprint(data["input_ids"][0].tolist())
    expected_metadata = {"query": query_mask[0].tolist(), "key": key_mask[0].tolist(), "positions": None}
    expected_hash = hashlib.sha256(json.dumps(expected_metadata, sort_keys=True).encode()).hexdigest()
    assert trace["sampling_metadata_sha256"] == expected_hash
    assert not {"input_ids", "input_text", "text", "tokens"}.intersection(recursive_keys(saved))
    assert json.dumps(data["input_ids"][0].tolist()) not in target.read_text()


def test_report_hashes_track_input_and_region_masks():
    model = model_fixture()
    data = data_fixture()
    config = MonitorConfig(sampling_policy="regions", queries=2, keys=4, distance_samples=16)
    monitor = RoPEMonitor(model, config)
    q_mask = torch.zeros_like(data["input_ids"], dtype=torch.bool)
    q_mask[:, 24:] = True
    k_mask = ~q_mask
    evaluate(monitor, data, sample_ids=["same-id"], query_mask=q_mask, key_mask=k_mask)
    first = deepcopy(monitor.last_report["sampling_trace"][0])
    k_mask[:, 0] = False
    evaluate(monitor, data, sample_ids=["same-id"], query_mask=q_mask, key_mask=k_mask)
    second = deepcopy(monitor.last_report["sampling_trace"][0])
    assert first["input_sha256"] == second["input_sha256"]
    assert first["sampling_metadata_sha256"] != second["sampling_metadata_sha256"]
    changed = {name: value.clone() for name, value in data.items()}
    changed["input_ids"][0, 1] += 1
    evaluate(monitor, changed, sample_ids=["same-id"], query_mask=q_mask, key_mask=k_mask)
    assert monitor.last_report["sampling_trace"][0]["input_sha256"] != second["input_sha256"]


def test_report_frequency_grid_is_a_historical_snapshot(tmp_path):
    model = model_fixture()
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=16))
    evaluate(monitor, data_fixture())
    expected = deepcopy(monitor.last_report)
    with torch.no_grad():
        model.model.rotary_emb.inv_freq.mul_(2)
    assert monitor.last_report == expected
    assert json.loads(monitor.save_report(tmp_path / "report.json").read_text()) == json.loads(json.dumps(expected))


def test_report_traces_frequency_changes_between_batches():
    model = model_fixture()
    monitor = RoPEMonitor(model, MonitorConfig(layers=[0], distance_samples=16))
    data = data_fixture()
    original = model.model.rotary_emb.inv_freq.detach().clone()
    with monitor.evaluation(step=19):
        with monitor.batch(**data, sample_ids=["first"]):
            with torch.no_grad():
                model(**data)
        with torch.no_grad():
            model.model.rotary_emb.inv_freq.mul_(1.5)
        changed = model.model.rotary_emb.inv_freq.detach().clone()
        with monitor.batch(**data, sample_ids=["second"]):
            with torch.no_grad():
                model(**data)
    report = monitor.last_report
    assert len(report["frequency_grids"]) == 2
    for grid, expected in zip(report["frequency_grids"], [original, changed]):
        torch.testing.assert_close(torch.tensor(grid["inv_freq"]), expected, rtol=0, atol=0)
    assert report["frequency_uses"] == [
        {"layer": 0, "grid_id": 0, "sample_indices": [0]},
        {"layer": 0, "grid_id": 1, "sample_indices": [1]},
    ]


def test_report_model_config_is_a_historical_snapshot():
    model = model_fixture(scaling={"rope_type": "linear", "factor": 2.0})
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=16))
    evaluate(monitor, data_fixture())
    expected = deepcopy(monitor.last_report)
    model.config.rope_scaling["factor"] = 3.0
    assert monitor.last_report == expected


def test_metadata_and_report_distinguish_native_frequency_config():
    monitors = [RoPEMonitor(model_fixture(theta=theta), MonitorConfig(distance_samples=16))
                for theta in (10000, 100000)]
    assert monitors[0].metadata() != monitors[1].metadata()
    for monitor in monitors:
        evaluate(monitor, data_fixture())
    assert monitors[0].last_report["frequency_grids"] != monitors[1].last_report["frequency_grids"]


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("heads", [[5, 0, 7], [3, 2]])
def test_sparse_head_order_and_gqa_match_all_head_reference_reduce_capture(family, heads, monkeypatch):
    import rope_profiler.monitor as monitor_module

    model = model_fixture(family)
    data = data_fixture()
    original_score = monitor_module.score_activations
    captures = []
    def spy(q, k, mapping, *args, **kwargs):
        result = original_score(q, k, mapping, *args, **kwargs)
        captures.append((q.clone(), k.clone(), tuple(mapping), result))
        return result
    monkeypatch.setattr(monitor_module, "score_activations", spy)
    full = RoPEMonitor(model, MonitorConfig(layers=[0], distance_samples=16))
    logits_full = evaluate(full, data)
    full_q, full_k, _, full_result = captures.pop()
    sparse = RoPEMonitor(model, MonitorConfig(layers=[0], heads={0: heads}, distance_samples=16))
    logits_sparse = evaluate(sparse, data)
    q, k, mapping, result = captures.pop()
    native_keys = sorted({head // 2 for head in heads})
    expected_mapping = tuple(native_keys.index(head // 2) for head in heads)
    assert q.shape[0] == len(heads) and k.shape[0] == len(native_keys)
    assert q.dtype == model.dtype and k.dtype == model.dtype
    assert mapping == expected_mapping
    torch.testing.assert_close(q, full_q[heads], atol=0, rtol=0)
    torch.testing.assert_close(k, full_k[native_keys], atol=0, rtol=0)
    torch.testing.assert_close(result.semantic_by_head, full_result.semantic_by_head[heads], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.positional_by_head, full_result.positional_by_head[heads], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(logits_sparse, logits_full, atol=0, rtol=0)
    expected_bytes = (q.numel() + k.numel()) * q.element_size()
    assert sparse.last_metrics["rope_retained_activation_bytes"] == expected_bytes
    assert sparse.last_metrics["rope_peak_activation_bytes"] == expected_bytes
    assert expected_bytes < full.last_metrics["rope_retained_activation_bytes"]
    assert sparse.last_report["metadata"]["selected_heads"][0] == heads


@pytest.mark.parametrize("case", ["internal_gap", "fractional_positions"])
def test_invalid_mask_and_positions_fail_before_forward_and_cleanup(case):
    model = model_fixture()
    data = data_fixture()
    if case == "internal_gap":
        data["attention_mask"][0, 4] = 0
        expected = "contiguous|internal"
    else:
        data["position_ids"] = torch.arange(32).unsqueeze(0) + 0.5
        expected = "integer positions"
    monitor = RoPEMonitor(model)
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    hooks_before = sum(len(module._forward_hooks) for module in model.modules())
    with pytest.raises(ValueError, match=expected):
        evaluate(monitor, data)
    assert calls == [] and monitor.last_report == {} and not monitor.active
    assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before
    handle.remove()


def test_failure_clears_previous_report_and_missing_capture_cannot_be_saved(tmp_path):
    model = model_fixture()
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=16))
    data = data_fixture()
    evaluate(monitor, data)
    assert monitor.last_report
    hooks_before = sum(len(module._forward_hooks) for module in model.modules())
    with pytest.raises(RuntimeError, match="did not capture"):
        with monitor.evaluation(step=18):
            with monitor.batch(**data):
                pass
    assert not monitor.active and monitor.last_report == {} and monitor.last_plans == ()
    assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before
    with pytest.raises(RuntimeError, match="complete an evaluation"):
        monitor.save_report(tmp_path / "failed.json")
    assert not (tmp_path / "failed.json").exists()


def test_save_report_requires_completed_evaluation(tmp_path):
    monitor = RoPEMonitor(model_fixture())
    with pytest.raises(RuntimeError, match="complete an evaluation"):
        monitor.save_report(tmp_path / "before.json")
    with monitor.evaluation():
        with pytest.raises(RuntimeError, match="complete an evaluation"):
            monitor.save_report(tmp_path / "during.json")
    assert not (tmp_path / "before.json").exists() and not (tmp_path / "during.json").exists()


def test_logger_failure_does_not_leave_completed_report(tmp_path):
    class BrokenLogger:
        def log(self, *args, **kwargs):
            raise RuntimeError("logging failure")
    model = model_fixture()
    monitor = RoPEMonitor(model, MonitorConfig(distance_samples=16), logger=BrokenLogger())
    with pytest.raises(RuntimeError, match="logging failure"):
        evaluate(monitor, data_fixture())
    assert not monitor.active and monitor.last_report == {}
    assert not any(module._forward_hooks for module in model.modules())
    with pytest.raises(RuntimeError, match="complete an evaluation"):
        monitor.save_report(tmp_path / "failed.json")
