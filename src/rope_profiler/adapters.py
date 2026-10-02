"""Native Hugging Face Llama/Qwen3 projection and RoPE metadata adapters."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any
import torch


@dataclass(frozen=True)
class LayerSpec:
    index: int
    name: str
    attention: Any
    query_module: Any
    key_module: Any
    query_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    layout: str
    rotary_module: Any
    attention_scale: float

    def active_rope(self) -> tuple[torch.Tensor, float]:
        # Read after position embeddings have been prepared for this forward.
        omega = self.rotary_module.inv_freq
        if not isinstance(omega, torch.Tensor) or omega.ndim != 1:
            raise ValueError("native RoPE must expose a one-dimensional inv_freq tensor")
        if omega.numel() * 2 != self.rotary_dim:
            raise ValueError("rotary dimension changed; recreate the monitor adapter")
        scaling = float(getattr(self.rotary_module, "attention_scaling", 1.0))
        return omega, scaling


class ModelAdapter:
    """Inspect an already-loaded model; no weights are loaded or recomputed."""

    def __init__(self, model: Any) -> None:
        self.model = model
        base = model
        while hasattr(base, "module"):
            base = base.module
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        config = getattr(base, "config", None)
        architecture = getattr(config, "model_type", None)
        if architecture not in {"llama", "qwen3"}:
            raise ValueError(f"initial release supports native Llama/Qwen3; got {architecture!r}")
        self.architecture = architecture
        self.config = config
        modules = list(base.named_modules())
        global_rotaries = [module for name, module in modules
                           if name.endswith("rotary_emb") and hasattr(module, "inv_freq")]
        specs = []
        for name, attention in modules:
            if not hasattr(attention, "q_proj") or not hasattr(attention, "k_proj"):
                continue
            if getattr(attention, "sliding_window", None) is not None:
                raise ValueError(
                    "current capture supports full causal attention only; "
                    f"sliding-window attention at {name} requires a window-aware sampling adapter"
                )
            rotary = getattr(attention, "rotary_emb", None)
            if rotary is None:
                if len(global_rotaries) != 1:
                    raise ValueError("cannot unambiguously associate attention with native RoPE")
                rotary = global_rotaries[0]
            hq = int(getattr(attention, "num_heads", config.num_attention_heads))
            hkv = int(getattr(attention, "num_key_value_heads", config.num_key_value_heads))
            dim = int(getattr(attention, "head_dim", getattr(config, "head_dim", None)
                              or config.hidden_size // hq))
            if hq < 1 or hkv < 1 or hq % hkv:
                raise ValueError("invalid native grouped-query head layout")
            rotary_dim = int(rotary.inv_freq.numel()) * 2
            if not 0 < rotary_dim <= dim or rotary_dim % 2:
                raise ValueError("invalid native rotary dimension")
            q_module = getattr(attention, "q_norm", None) or attention.q_proj
            k_module = getattr(attention, "k_norm", None) or attention.k_proj
            specs.append(LayerSpec(len(specs), name, attention, q_module, k_module,
                                   hq, hkv, dim, rotary_dim, "split_half", rotary,
                                   float(getattr(attention, "scaling", dim ** -0.5))))
        if not specs:
            raise ValueError("model has no exposed native Q/K projection modules")
        self.layers = tuple(specs)

    def metadata(self) -> dict[str, Any]:
        return {"architecture": self.architecture,
                "model_class": type(self.model).__name__,
                "rope_config": {name: deepcopy(getattr(self.config, name, None))
                                for name in ("rope_theta", "rope_scaling", "max_position_embeddings")},
                "layers": [
            {"index": spec.index, "name": spec.name, "query_heads": spec.query_heads,
             "kv_heads": spec.kv_heads, "head_dim": spec.head_dim,
             "rotary_dim": spec.rotary_dim, "layout": spec.layout,
             "capture_point": "post_qk_norm_pre_rope" if spec.query_module is not spec.attention.q_proj
             else "post_projection_pre_rope",
             "attention_scale": spec.attention_scale}
            for spec in self.layers]}

    def frequency_bands(self, window: int, threshold: float) -> dict[int, torch.Tensor]:
        """Operational mask M*omega >= threshold using the latest native grid.

        The threshold is an explicit analysis setting; scores use all rotary
        coordinates and do not depend on this optional partition.
        """
        if window < 2 or threshold <= 0:
            raise ValueError("window must be >=2 and threshold must be positive")
        return {spec.index: spec.active_rope()[0] * window >= threshold for spec in self.layers}


def projection_view(output: Any, spec: LayerSpec, kind: str) -> torch.Tensor:
    """Return a transient [B,T,H,D] view without cloning the full sequence."""
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError("projection output must be a tensor")
    heads = spec.query_heads if kind == "q" else spec.kv_heads
    if value.ndim == 3 and value.shape[-1] == heads * spec.head_dim:
        return value.view(value.shape[0], value.shape[1], heads, spec.head_dim)
    if value.ndim == 4 and value.shape[-1] == spec.head_dim:
        if value.shape[2] == heads:
            return value
        if value.shape[1] == heads:
            return value.transpose(1, 2)
    raise ValueError(f"unsupported {kind} projection shape {tuple(value.shape)}")
