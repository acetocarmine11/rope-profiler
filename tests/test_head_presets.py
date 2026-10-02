"""Check the verified paper lists and their resolution in the monitor."""
import hashlib
from importlib.resources import files
import json
from types import SimpleNamespace

import pytest

from rope_profiler import MonitorConfig, RoPEMonitor, get_head_preset, list_head_presets
from rope_profiler.heads import find_head_preset


def fake_adapter(monkeypatch, name, *, family="qwen3", layers=36, heads=32):
    adapter = SimpleNamespace(
        config=SimpleNamespace(_name_or_path=name), architecture=family,
        layers=tuple(SimpleNamespace(index=index, query_heads=heads, kv_heads=8, head_dim=128)
                     for index in range(layers)),
        metadata=lambda: {"architecture": family})
    monkeypatch.setattr("rope_profiler.monitor.ModelAdapter", lambda model: adapter)
    return object()


def test_paper_resource_matches_independently_audited_bytes():
    content = files("rope_profiler").joinpath("data/paper_heads.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == "d02c729feb5188362dd6e7523757bbeed8f43fee8a51df2f4b7976a0262f726d"
    assert set(list_head_presets()) == {"qwen3-8b", "llama-3.1-8b-instruct"}


@pytest.mark.parametrize("name,count,layers", [("qwen3-8b", 58, 36), ("llama-3.1-8b-instruct", 52, 32)])
def test_paper_presets_keep_ranking_and_independent_selection(name, count, layers):
    preset = get_head_preset(name)
    assert len(preset.ranked_heads) == len(set(preset.ranked_heads)) == count
    assert preset.num_layers == layers and preset.num_query_heads == 32
    assert sum(len(heads) for heads in preset.heads.values()) == count
    assert {(layer, head) for layer, heads in preset.heads.items() for head in heads} == set(preset.ranked_heads)
    mutable = preset.heads
    mutable.clear()
    assert sum(len(heads) for heads in preset.heads.values()) == count
    metadata = json.loads(json.dumps(preset.to_dict()))
    assert metadata["ddof"] == 1 and metadata["num_settings"] == 49
    assert metadata["pair_protocol"] == "common_pair"
    assert metadata["source_sha256"] == "c57e4424cf21f9f7c57f6d2e9aa979d67a6ee17daf2e66df080ed5e472035a9d"


@pytest.mark.parametrize("identifier,preset", [
    ("Qwen/Qwen3-8B", "qwen3-8b"), ("Qwen3-8B", "qwen3-8b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct"),
])
def test_auto_resolves_verified_identity_and_gathers_only_paper_heads(monkeypatch, identifier, preset):
    expected = get_head_preset(preset)
    model = fake_adapter(monkeypatch, identifier, family=expected.model_type, layers=expected.num_layers)
    monitor = RoPEMonitor(model)
    assert monitor.config.head_preset == "auto"
    assert monitor._query_heads == expected.heads
    assert {spec.index for spec in monitor.layers} == set(expected.heads)
    assert monitor.metadata()["head_selection"]["preset"]["name"] == preset
    assert monitor.metadata()["head_selection"]["selected_count"] == len(expected.ranked_heads)
    for spec in monitor.layers:
        groups = [head // 4 for head in expected.heads[spec.index]]
        assert monitor._kv_heads[spec.index] == tuple(sorted(set(groups)))
        assert monitor._query_to_kv[spec.index] == tuple(monitor._kv_heads[spec.index].index(kv) for kv in groups)


@pytest.mark.parametrize("identifier", ["", "another-org/Qwen3-8B", "/tmp/Qwen3-8B", "Qwen/Qwen3-4B", None])
def test_auto_never_guesses_model_identity_from_geometry(monkeypatch, identifier):
    assert find_head_preset(identifier) is None
    monitor = RoPEMonitor(fake_adapter(monkeypatch, identifier))
    assert monitor.head_preset is None
    assert monitor.metadata()["head_selection"] == {"mode": "all", "selected_count": 1152, "preset": None}


def test_explicit_preset_supports_local_checkpoint(monkeypatch):
    monitor = RoPEMonitor(fake_adapter(monkeypatch, "/my/checkpoint"), MonitorConfig(head_preset="qwen3-8b"))
    assert monitor.metadata()["head_selection"]["selected_count"] == 58


def test_all_or_manual_heads_override_auto_preset(monkeypatch):
    model = fake_adapter(monkeypatch, "Qwen/Qwen3-8B")
    monitor = RoPEMonitor(model, MonitorConfig(head_preset=None))
    assert monitor.metadata()["head_selection"]["selected_count"] == 1152
    monitor = RoPEMonitor(model, MonitorConfig(layers=[0], heads={0: [1, 4]}))
    assert monitor._query_heads == {0: (1, 4)} and monitor.head_preset is None
    assert monitor.metadata()["head_selection"]["mode"] == "custom"
    monitor = RoPEMonitor(model, MonitorConfig(heads={1: [2, 7]}))
    assert monitor._query_heads == {1: (2, 7)}
    assert [spec.index for spec in monitor.layers] == [1]


@pytest.mark.parametrize("family,layers,heads", [("llama", 36, 32), ("qwen3", 35, 32), ("qwen3", 36, 16)])
def test_preset_rejects_different_model_layout(monkeypatch, family, layers, heads):
    model = fake_adapter(monkeypatch, "Qwen/Qwen3-8B", family=family, layers=layers, heads=heads)
    with pytest.raises(ValueError, match="does not match"):
        RoPEMonitor(model)


@pytest.mark.parametrize("value", [False, 5, ""])
def test_invalid_preset_config_is_rejected(value):
    with pytest.raises(ValueError, match="head_preset"):
        MonitorConfig(head_preset=value)


def test_explicit_preset_and_custom_selection_cannot_silently_conflict():
    with pytest.raises(ValueError, match="cannot be combined"):
        MonitorConfig(head_preset="qwen3-8b", heads={0: [0]})
    config = MonitorConfig(head_preset="llama-3.1-8b-instruct")
    assert MonitorConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="unknown paper head preset"):
        get_head_preset("qwen3-4b")


def test_all_head_setting_is_explicit_for_legacy_config():
    assert MonitorConfig.from_dict({"head_preset": None}).head_preset is None
