"""Immutable, JSON-recoverable configuration for a RoPE monitor."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence


class _FrozenHeads(dict):
    """An immutable dict with tuple values and ordinary dict serialization.

    Keeping dict's read interface lets dataclasses.asdict and JSON handle this
    mapping directly. Every ordinary mutator fails; pickle reconstructs it
    through its constructor. As with frozen dataclasses, deliberately calling
    low-level base-class mutation methods is outside the public contract.
    """

    __slots__ = ()

    def __new__(cls, values=()):
        instance = super().__new__(cls)
        dict.update(instance, ((key, tuple(value)) for key, value in dict(values).items()))
        return instance

    def __init__(self, values=()):
        # Construction happens once in __new__; re-invoking __init__ is inert.
        pass

    def _immutable(self, *args, **kwargs):
        raise TypeError("head selection is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __reduce_ex__(self, protocol):
        return type(self), (dict(self),)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class MonitorConfig:
    """Specify sampling and computation without changing model inference.

    ``causal`` uses the monitor's generic causal token sampler. ``regions``
    requires both query_mask and key_mask at each monitored batch. The caller
    supplies masks that already exclude special tokens, punctuation, invalid
    offsets, and padding as appropriate for its task. Region masks define token
    eligibility; they do not reproduce the paper's SHA-256 token-ordering rule.

    ``distance_samples=None`` enumerates every virtual adjacent distance for
    the positional score. Tokens remain sampled and semantic reversal remains
    the Gaussian approximation.

    ``head_preset="auto"`` uses the paper's fixed top 5% for recognized model
    IDs; other models include all heads. ``head_preset=None`` selects all heads.
    Explicit ``heads``/``layers`` override auto. With ``heads`` and no ``layers``,
    only the mapping's layers are selected. With explicit ``layers``, layers
    omitted from ``heads`` include all their query heads.
    """

    queries: int = 4
    keys: int = 16
    distance_samples: int | None = 256
    seed: int = 0
    max_examples: int | None = 32
    layers: Sequence[int] | None = None
    heads: Mapping[int, Sequence[int]] | None = None
    score_dtype: str = "float64"
    cache_entries: int = 64
    sampling_policy: str = "causal"
    head_preset: str | None = "auto"

    def __post_init__(self) -> None:
        for name in ("queries", "keys", "seed", "cache_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.queries < 1 or self.keys < 2:
            raise ValueError("queries must be >=1 and keys must be >=2")
        if self.cache_entries < 1:
            raise ValueError("cache_entries must be positive")
        for name in ("distance_samples", "max_examples"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer or None")
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        if self.score_dtype not in ("float32", "float64"):
            raise ValueError("score_dtype must be float32 or float64")
        if self.sampling_policy not in ("causal", "regions"):
            raise ValueError("sampling_policy must be causal or regions")
        if self.head_preset is not None and (not isinstance(self.head_preset, str) or not self.head_preset):
            raise ValueError("head_preset must be 'auto', a paper preset name, or None")
        if self.head_preset not in (None, "auto") and (self.heads is not None or self.layers is not None):
            raise ValueError("an explicit head_preset cannot be combined with custom heads/layers")
        if self.layers is not None:
            try:
                layers = tuple(self.layers)
            except TypeError as exc:
                raise ValueError("layers must be a sequence of integers or None") from exc
            if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
                raise ValueError("layer indices must be integers")
            object.__setattr__(self, "layers", layers)
        if self.heads is not None:
            if not isinstance(self.heads, Mapping):
                raise ValueError("heads must map integer layer indices to head-index sequences")
            copied = {}
            for layer, selection in self.heads.items():
                if isinstance(layer, bool) or not isinstance(layer, int):
                    raise ValueError("head-selection layer indices must be integers")
                try:
                    heads = tuple(selection)
                except TypeError as exc:
                    raise ValueError("head selection must be a sequence of integers") from exc
                if any(isinstance(head, bool) or not isinstance(head, int) for head in heads):
                    raise ValueError("head indices must be integers")
                copied[layer] = heads
            object.__setattr__(self, "heads", _FrozenHeads(copied))

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-ready configuration, including head keys."""
        return {
            "queries": self.queries,
            "keys": self.keys,
            "distance_samples": self.distance_samples,
            "seed": self.seed,
            "max_examples": self.max_examples,
            "layers": None if self.layers is None else list(self.layers),
            "heads": None if self.heads is None else {
                str(layer): list(self.heads[layer]) for layer in sorted(self.heads)},
            "score_dtype": self.score_dtype,
            "cache_entries": self.cache_entries,
            "sampling_policy": self.sampling_policy,
            "head_preset": self.head_preset,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MonitorConfig":
        """Restore exported/JSON configuration and reject unrecognized fields.

        Direct construction requires integer head-selection keys. This loader
        also accepts canonical JSON integer strings, e.g. ``"0"`` and ``"-1"``.
        """
        if not isinstance(value, Mapping):
            raise ValueError("configuration must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(map(str, unknown))}")
        restored = dict(value)
        heads = restored.get("heads")
        if heads is not None:
            if not isinstance(heads, Mapping):
                raise ValueError("heads must be a mapping")
            normalized = {}
            for key, selection in heads.items():
                if isinstance(key, bool):
                    raise ValueError("head-selection layer indices must be integers")
                if isinstance(key, str):
                    try:
                        layer = int(key)
                    except ValueError as exc:
                        raise ValueError("JSON head-selection keys must be integer strings") from exc
                    if str(layer) != key:
                        raise ValueError("JSON head-selection keys must be canonical integer strings")
                elif isinstance(key, int):
                    layer = key
                else:
                    raise ValueError("head-selection layer indices must be integers")
                if layer in normalized:
                    raise ValueError(f"duplicate head-selection layer index after normalization: {layer}")
                normalized[layer] = selection
            restored["heads"] = normalized
        return cls(**restored)

    @classmethod
    def full_distances(cls, **kwargs: Any) -> "MonitorConfig":
        """Enumerate positional distances while retaining token/Gaussian sampling.

        Other configuration fields may be overridden. A non-None distance
        budget conflicts with this factory and is rejected.
        """
        if kwargs.get("distance_samples") is not None:
            raise ValueError("full_distances requires distance_samples=None")
        return cls(**{**kwargs, "distance_samples": None})


__all__ = ["MonitorConfig"]
