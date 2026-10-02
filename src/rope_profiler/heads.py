"""Fixed Common-pair head selections distributed with the paper's monitor."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any


@dataclass(frozen=True)
class PaperHeadPreset:
    """A model-specific, zero-based head list in descending variance order.

    The same fixed selection supplies both scores across tasks/checkpoints.
    Access ``heads`` for the layer-to-query-head mapping accepted by the monitor.
    Each access returns an independent mapping; the stored ranking is immutable.
    """

    name: str
    model_id: str
    model_type: str
    num_layers: int
    num_query_heads: int
    ranked_heads: tuple[tuple[int, int], ...]
    source_sha256: str
    num_settings: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None

    @property
    def heads(self) -> dict[int, tuple[int, ...]]:
        grouped: dict[int, list[int]] = {}
        for layer, head in self.ranked_heads:
            grouped.setdefault(layer, []).append(head)
        return {layer: tuple(sorted(grouped[layer])) for layer in sorted(grouped)}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id,
                "model_type": self.model_type, "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads, "head_dim": self.head_dim,
                "selected_count": len(self.ranked_heads), "fraction": 0.05,
                "count_rule": "ceil(fraction * total_query_heads)",
                "statistic": "sample_variance_of_setting_mean_r_H", "ddof": 1,
                "num_settings": self.num_settings, "pair_protocol": "common_pair",
                "historical_cutoff": {"frame_constant": 0.05, "epsilon": 0.10},
                "indexing": "zero-based layer/head", "source_sha256": self.source_sha256,
                "ranked_heads": [list(pair) for pair in self.ranked_heads]}


@lru_cache(maxsize=1)
def _presets() -> tuple[PaperHeadPreset, ...]:
    data = json.loads(files("rope_profiler").joinpath("data/paper_heads.json").read_text(encoding="utf-8"))
    selection = data["selection"]
    if (data["schema_version"] != 1 or selection["fraction"] != 0.05
            or selection["statistic"] != "sample_variance" or selection["ddof"] != 1
            or selection["settings"] != 49 or selection["pair_protocol"] != "common_pair"):
        raise ValueError("unrecognized bundled paper head-selection protocol")
    result = []
    for name, values in data["presets"].items():
        preset = PaperHeadPreset(
            name=name, model_id=values["model_id"], model_type=values["model_type"],
            num_layers=values["num_layers"], num_query_heads=values["num_query_heads"],
            ranked_heads=tuple(tuple(pair) for pair in values["ranked_heads"]),
            source_sha256=selection["source_sha256"], num_settings=selection["settings"],
            num_key_value_heads=values.get("num_key_value_heads"), head_dim=values.get("head_dim"))
        expected_count = (preset.num_layers * preset.num_query_heads + 19) // 20
        if (len(preset.ranked_heads) != expected_count
                or len(set(preset.ranked_heads)) != expected_count
                or any(len(pair) != 2 or any(type(index) is not int for index in pair)
                       or not 0 <= pair[0] < preset.num_layers
                       or not 0 <= pair[1] < preset.num_query_heads for pair in preset.ranked_heads)):
            raise ValueError(f"invalid bundled paper head selection: {name}")
        result.append(preset)
    return tuple(result)


def list_head_presets() -> tuple[str, ...]:
    """Return names of the bundled, model-specific paper selections."""
    return tuple(preset.name for preset in _presets())


def find_head_preset(model_id: Any) -> PaperHeadPreset | None:
    """Recognize an exact public ID or short model name, without guessing paths.

    Local checkpoint paths and other publishers' fine-tuned models require an
    explicit preset choice. A matching model geometry alone is insufficient.
    """
    if not isinstance(model_id, str):
        return None
    identifier = model_id.casefold()
    return next((preset for preset in _presets()
                 if identifier in {preset.name.casefold(), preset.model_id.casefold(),
                                   preset.model_id.split("/")[-1].casefold()}), None)


def get_head_preset(name: str) -> PaperHeadPreset:
    """Load a bundled paper preset by name or canonical model identifier."""
    preset = find_head_preset(name)
    if preset is None:
        raise ValueError(f"unknown paper head preset {name!r}; choose from {list_head_presets()}")
    return preset


def validate_head_preset(preset: PaperHeadPreset, adapter: Any) -> None:
    """Reject a preset when the loaded model's native head layout differs."""
    if (adapter.architecture != preset.model_type or len(adapter.layers) != preset.num_layers
            or any(spec.query_heads != preset.num_query_heads
                   or (preset.num_key_value_heads is not None and spec.kv_heads != preset.num_key_value_heads)
                   or (preset.head_dim is not None and spec.head_dim != preset.head_dim)
                   for spec in adapter.layers)):
        raise ValueError(f"paper preset {preset.name!r} does not match the loaded model layout; "
                         "use head_preset=None for all heads or supply a custom selection")


__all__ = ["PaperHeadPreset", "get_head_preset", "list_head_presets"]
