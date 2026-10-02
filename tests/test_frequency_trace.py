"""Historical native frequency tracing survives eviction and buffer replacement."""
from __future__ import annotations

import weakref

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rope_profiler import MonitorConfig, RoPEMonitor


def test_frequency_uses_match_capture_grids_and_windows_with_one_cache_entry(monkeypatch):
    import rope_profiler.monitor as monitor_module

    torch.manual_seed(209)
    config = LlamaConfig(
        vocab_size=53, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    monitor = RoPEMonitor(model, MonitorConfig(
        queries=2, keys=4, distance_samples=2, max_examples=None, cache_entries=1,
    ))
    original_score = monitor_module.score_activations
    captured = []

    def observe_score(q, k, mapping, pairs, triplets, omega, window, **kwargs):
        # Clone values only: the observer must not keep the original source
        # buffers alive and accidentally prevent the identity-reuse regression.
        captured.append((omega.detach().clone(), window, kwargs["rope_scale"]))
        return original_score(q, k, mapping, pairs, triplets, omega, window, **kwargs)

    monkeypatch.setattr(monitor_module, "score_activations", observe_score)
    source_refs = []
    expected_grids = []
    expected_windows = []
    with monitor.evaluation(step=40):
        for index in range(40):
            # Changing the native buffer mirrors dynamic RoPE replacing its
            # tensor. Every replacement has mutation version zero. A tiny
            # score cache evicts older anchors; the report needs its own
            # lifetime guarantee to keep Python identities from being reused.
            model.model.rotary_emb.inv_freq = torch.tensor(
                [0.2 + index * 0.01, 0.13, 0.04, 0.003], dtype=torch.float32,
            )
            source_refs.append(weakref.ref(model.model.rotary_emb.inv_freq))
            expected_grids.append(model.model.rotary_emb.inv_freq.clone())
            length = 12 + index % 3
            expected_windows.append(length)
            ids = (torch.arange(length).unsqueeze(0) % 50) + 1
            data = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            with monitor.batch(**data, sample_ids=[f"sample-{index}"]):
                with torch.no_grad():
                    model(**data)
            # This makes the failure deterministic even on an allocator that
            # happens not to recycle a tensor identity during forty forwards.
            assert all(reference() is not None for reference in source_refs)

    report = monitor.last_report
    assert len(report["frequency_grids"]) == 40
    assert len(report["sampling_trace"]) == 40
    assert len(report["frequency_uses"]) == len(captured) == 80
    grids = {grid["id"]: grid for grid in report["frequency_grids"]}
    for call_index, (use, observation) in enumerate(zip(report["frequency_uses"], captured)):
        sample_index = call_index // 2
        observed_grid, observed_window, observed_scale = observation
        assert use["layer"] == call_index % 2
        assert use["sample_indices"] == [sample_index]
        trace = report["sampling_trace"][sample_index]
        assert trace["sample_id"] == f"sample-{sample_index}"
        assert trace["window"] == observed_window == expected_windows[sample_index]
        stored = grids[use["grid_id"]]
        torch.testing.assert_close(torch.tensor(stored["inv_freq"]), observed_grid, rtol=0, atol=0)
        torch.testing.assert_close(observed_grid, expected_grids[sample_index], rtol=0, atol=0)
        assert stored["attention_scaling"] == observed_scale

    assert report["cache"]["entries"] == report["cache"]["max_entries"] == 1
    # Historical anchors are needed only while constructing the report. The
    # final buffer stays alive in the model/cache; previous sources are freed.
    assert all(reference() is None for reference in source_refs[:-1])
    assert source_refs[-1]() is model.model.rotary_emb.inv_freq
