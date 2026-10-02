from dataclasses import FrozenInstanceError, asdict
import json
import pickle

import pytest

from rope_profiler.config import MonitorConfig


@pytest.mark.parametrize("field,value", [
    ("queries", True), ("keys", 2.5), ("seed", False),
    ("cache_entries", 0), ("distance_samples", 2.5), ("max_examples", True),
    ("layers", [False]), ("heads", {0: [1.5]}),
])
def test_config_rejects_malformed_sample_budgets(field, value):
    with pytest.raises(ValueError):
        MonitorConfig(**{field: value})


def test_config_copies_sample_selection():
    layers, heads = [0], {0: [1, 2]}
    config = MonitorConfig(layers=layers, heads=heads)
    layers.append(1)
    heads[0].append(3)
    assert config.layers == (0,)
    assert config.heads[0] == (1, 2)
    with pytest.raises(TypeError):
        config.heads[0] = (3,)


def test_config_exhaustive_and_unlimited_are_explicit():
    config = MonitorConfig(distance_samples=None, max_examples=None)
    assert config.distance_samples is None
    assert config.max_examples is None
    assert config.cache_entries == 64


@pytest.mark.parametrize("config", [
    MonitorConfig(),
    MonitorConfig.full_distances(max_examples=None),
    MonitorConfig(layers=[0, -1], heads={0: [1, 2], 7: [0]}, sampling_policy="regions"),
])
def test_config_json_asdict_pickle_roundtrips(config):
    exported = config.to_dict()
    assert MonitorConfig.from_dict(json.loads(json.dumps(exported))) == config
    assert MonitorConfig.from_dict(json.loads(json.dumps(asdict(config)))) == config
    assert pickle.loads(pickle.dumps(config)) == config


def test_config_export_is_an_independent_copy():
    config = MonitorConfig(layers=[0], heads={0: [1, 2]})
    exported = config.to_dict()
    exported["layers"].append(3)
    exported["heads"]["0"].append(3)
    assert config.layers == (0,)
    assert config.heads == {0: (1, 2)}


@pytest.mark.parametrize("mutate", [
    lambda heads: heads.__setitem__(0, (3,)),
    lambda heads: heads.__delitem__(0),
    lambda heads: heads.update({0: (3,)}),
    lambda heads: heads.clear(),
    lambda heads: heads.pop(0),
    lambda heads: heads.popitem(),
    lambda heads: heads.setdefault(0, (3,)),
    lambda heads: heads.__ior__({0: (3,)}),
])
def test_config_all_ordinary_mapping_mutators_are_rejected(mutate):
    config = MonitorConfig(heads={0: [1, 2]})
    with pytest.raises(TypeError, match="immutable"):
        mutate(config.heads)
    assert config.heads == {0: (1, 2)}


def test_config_field_assignment_and_tuple_mutation_are_rejected():
    config = MonitorConfig(heads={0: [1, 2]})
    with pytest.raises(FrozenInstanceError):
        config.queries = 12
    with pytest.raises(TypeError):
        config.heads[0][0] = 4
    # Re-initializing a mapping cannot alter an existing frozen instance.
    config.heads.__init__({0: (3,)})
    assert config.heads == {0: (1, 2)}


@pytest.mark.parametrize("value,match", [
    ({"unknown": 1}, "unknown configuration"),
    ({"sampling_policy": "paper"}, "sampling_policy"),
    ({"heads": {"x": [0]}}, "integer strings"),
    ({"heads": {"01": [0]}}, "canonical"),
    ({"heads": {0: [0], "0": [1]}}, "duplicate"),
    ({"heads": {True: [0]}}, "integers"),
    ({"heads": []}, "mapping"),
    ({"layers": [0.5]}, "integers"),
    ({"keys": 1}, "queries must"),
])
def test_config_loader_rejects_invalid_or_ambiguous_configuration(value, match):
    with pytest.raises(ValueError, match=match):
        MonitorConfig.from_dict(value)


def test_config_loader_accepts_int_and_canonical_json_head_keys():
    assert MonitorConfig.from_dict({"heads": {"0": [1], "-1": [2]}}).heads == {0: (1,), -1: (2,)}
    assert MonitorConfig.from_dict({"heads": {0: [1]}}).heads == {0: (1,)}


def test_config_consumes_generator_selections_once():
    config = MonitorConfig(layers=(index for index in [0, 1]),
                           heads={0: (head for head in [1, 2])})
    assert config.layers == (0, 1)
    assert config.heads == {0: (1, 2)}


def test_full_distances_factory_changes_only_distance_budget():
    config = MonitorConfig.full_distances(queries=2, keys=8, sampling_policy="regions")
    assert config.distance_samples is None
    assert config.queries == 2 and config.keys == 8
    assert config.sampling_policy == "regions"
    assert config.max_examples == 32
    assert config.score_dtype == "float64"
    with pytest.raises(ValueError, match="distance_samples=None"):
        MonitorConfig.full_distances(distance_samples=128)
