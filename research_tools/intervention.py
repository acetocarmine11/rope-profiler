"""Inference-only frequency-band scaling on an already-loaded native model.

This source-checkout research helper is separate from the monitoring package.
For each selected logical query head, it scales the two Q coordinates of a
rotary plane after native Q normalization and before rotation. A scale alpha
therefore multiplies that plane's QK score coefficient by alpha. The paper uses
sqrt(alpha) on both Q and K; the coefficient operation is mathematically the
same, while the internal tensors and floating-point rounding differ. Q-only
scaling keeps the shared GQA keys intact. No coefficient-norm compensation is
applied, and the nonrotary tail is preserved.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from threading import RLock
from typing import Any, Mapping, Sequence
import weakref

import torch

from rope_profiler.adapters import LayerSpec, ModelAdapter, projection_view


_ACTIVE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ACTIVE_LOCK = RLock()


def _positive_index(value: Any, name: str, *, zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


def _finite_scale(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def _scale_query(output: torch.Tensor, spec: LayerSpec, indices: torch.Tensor,
                 coordinate_scales: torch.Tensor, *, validate_finite: bool = False) -> torch.Tensor:
    """Return a Q replacement with the native shape, leaving the input intact."""
    view = projection_view(output, spec, "q")
    result = view.clone()
    chosen = view.index_select(2, indices)
    chosen[..., :spec.rotary_dim].mul_(coordinate_scales)
    # CPU validation is cheap and does not synchronize another device. GPU
    # validation is opt-in because reading this result synchronizes its stream.
    # A CUDA device assert would instead invalidate that CUDA context on error.
    if chosen.device.type == "cpu" or validate_finite:
        if not torch.isfinite(chosen).all().item():
            raise ValueError("band scaling produced nonfinite selected Q; reduce the scales or inspect input Q")
    result.index_copy_(2, indices, chosen)
    if output.ndim == 3:
        return result.reshape_as(output)
    if output.shape[2] == spec.query_heads:
        return result
    return result.transpose(1, 2)


class BandScaleIntervention:
    """Temporarily multiply high/low-band score coefficients during inference.

    ``heads`` maps zero-based layer indices to logical query-head indices;
    ``None`` selects every head, and ``{}`` is an explicit no-op. High-frequency
    planes satisfy ``window * omega >= threshold`` using the actual native
    frequency grid produced by the first rotary forward in this context. That
    per-layer mask stays fixed throughout subsequent forwards and decode.
    ``window`` is a fixed reference length; it does not grow during generation.

    Enter under ``torch.no_grad()`` or ``torch.inference_mode()`` with an eval
    model. Native Hugging Face Llama/Qwen3 eager and SDPA paths are supported.
    Each forward must go through the supplied model and run its native rotary
    module before Q. Bypassing that execution order raises an error instead of
    using a stale dynamic-RoPE grid. Identity settings register no hooks.
    Query replacement runs before ordinary module hooks so RoPEMonitor observes
    the effective Q regardless of the two contexts' nesting order.

    Scales must fit the Q dtype. CPU also checks each scaled selected Q block
    for nonfinite values. On other devices, ``validate_finite=True`` enables
    the same check with a tensor-value synchronization per selected layer and
    forward. The default avoids that synchronization; finite factors alone
    do not guarantee finite GPU products or whole-model logits. Identity
    contexts do not add finite-value checks to the native model.

    Native KV caching and single-token decode remain available. Existing keys
    are never directly changed: only queries executed inside the context are
    scaled. Entering after a previous prefill has a different whole-model
    trajectory from intervening during that prefill. A selected head can also
    change later layers through the residual stream; the local GQA isolation
    guarantee concerns heads given the same incoming hidden states.

    One context is used by one inference stream. Nested/overlapping contexts
    on the same model or selected Q modules are rejected. Exiting removes all
    hooks, including when inference raises, and re-entering freezes a new grid.
    """

    def __init__(self, model: Any, *, window: int, threshold: float,
                 high_scale: float = 1.0, low_scale: float = 1.0,
                 heads: Mapping[int, Sequence[int]] | None = None,
                 validate_finite: bool = False) -> None:
        self.model = model
        self.window = _positive_index(window, "window", zero=False)
        self.threshold = _finite_scale(threshold, "threshold", positive=True)
        self.high_scale = _finite_scale(high_scale, "high_scale")
        self.low_scale = _finite_scale(low_scale, "low_scale")
        if not isinstance(validate_finite, bool):
            raise ValueError("validate_finite must be a boolean")
        self.validate_finite = validate_finite
        self.adapter = ModelAdapter(model)
        backend = getattr(self.adapter.config, "_attn_implementation", None)
        if backend not in {"eager", "sdpa"}:
            raise ValueError("band scaling currently supports native eager/SDPA attention")
        if heads is None:
            self.heads = {spec.index: tuple(range(spec.query_heads)) for spec in self.adapter.layers}
        elif isinstance(heads, Mapping):
            self.heads = {}
            for raw_layer, raw_heads in heads.items():
                layer = _positive_index(raw_layer, "layer")
                if layer >= len(self.adapter.layers):
                    raise ValueError(f"layer {layer} is outside the model")
                if isinstance(raw_heads, (str, bytes)):
                    raise ValueError("heads must be sequences of query-head indices")
                try:
                    chosen = tuple(_positive_index(head, "query head") for head in raw_heads)
                except TypeError as exc:
                    raise ValueError("heads must be sequences of query-head indices") from exc
                if not chosen or len(set(chosen)) != len(chosen):
                    raise ValueError("each selected layer needs distinct, nonempty query heads")
                if max(chosen) >= self.adapter.layers[layer].query_heads:
                    raise ValueError(f"query head is outside layer {layer}")
                self.heads[layer] = chosen
        else:
            raise ValueError("heads must be a layer-to-query-head mapping or None")
        self._specs = tuple(spec for spec in self.adapter.layers if spec.index in self.heads)
        # A tied Q module cannot distinguish which logical layer invoked it.
        for spec in self._specs:
            if sum(other.query_module is spec.query_module for other in self.adapter.layers) != 1:
                raise ValueError("shared Q modules need a layer-aware intervention adapter")
            if spec.layout != "split_half":
                raise ValueError("band scaling requires native split-half rotary layout")
        self._identity = not self._specs or (self.high_scale == 1.0 and self.low_scale == 1.0)
        self._handles: list[Any] = []
        self._claims: list[Any] = []
        self._entered = False
        self._inside_forward = False
        self._prepared: set[int] = set()
        self._omega: dict[int, torch.Tensor] = {}
        self._masks: dict[int, torch.Tensor] = {}
        self._factors: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._query_forwards: dict[int, int] = {}

    def _check_inference(self) -> None:
        if any(module.training for module in self.model.modules()) or torch.is_grad_enabled():
            raise RuntimeError("band scaling requires model.eval() and no_grad()/inference_mode()")

    def __enter__(self) -> "BandScaleIntervention":
        self._check_inference()
        if self._entered:
            raise RuntimeError("band intervention context is already active")
        claims = [self.model] + [spec.query_module for spec in self._specs]
        with _ACTIVE_LOCK:
            if any(_ACTIVE.get(module, lambda: None)() is not None for module in claims):
                raise RuntimeError("overlapping band intervention on the same model/Q modules")
            for module in claims:
                _ACTIVE[module] = weakref.ref(self)
        self._claims = claims
        self._entered = True
        self._inside_forward = False
        self._prepared.clear()
        self._omega.clear()
        self._masks.clear()
        self._factors.clear()
        self._query_forwards = {spec.index: 0 for spec in self._specs}
        try:
            if not self._identity:
                self._handles.append(self.model.register_forward_pre_hook(self._model_begin))
                self._handles.append(self.model.register_forward_hook(self._model_done, always_call=True))
                rotaries: dict[Any, list[LayerSpec]] = {}
                for spec in self._specs:
                    rotaries.setdefault(spec.rotary_module, []).append(spec)
                for module, specs in rotaries.items():
                    self._handles.append(module.register_forward_hook(self._rotary_hook(tuple(specs))))
                for spec in self._specs:
                    self._handles.append(spec.query_module.register_forward_hook(self._query_hook(spec), prepend=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        with _ACTIVE_LOCK:
            for module in self._claims:
                owner = _ACTIVE.get(module)
                if owner is not None and owner() is self:
                    del _ACTIVE[module]
        self._claims.clear()
        self._entered = False
        self._inside_forward = False
        self._prepared.clear()
        # Keep the frozen grid and counters for an explicit post-run report.
        self._factors.clear()

    def _model_begin(self, _module: Any, _args: Any) -> None:
        self._check_inference()
        if self._inside_forward:
            raise RuntimeError("band intervention requires one sequential inference stream")
        self._inside_forward = True
        self._prepared.clear()

    def _model_done(self, _module: Any, _args: Any, _output: Any) -> None:
        self._inside_forward = False
        self._prepared.clear()

    def _rotary_hook(self, specs: tuple[LayerSpec, ...]):
        def after_rotary(_module: Any, _args: Any, _output: Any) -> None:
            if not self._inside_forward:
                return
            # forward has already updated dynamic inv_freq and generated the
            # actual cos/sin. Reading earlier at context entry can be stale.
            for spec in specs:
                omega = spec.rotary_module.inv_freq
                if not isinstance(omega, torch.Tensor) or omega.ndim != 1 or omega.numel() * 2 != spec.rotary_dim:
                    raise ValueError("native rotary frequency layout changed; recreate the intervention")
                if spec.index not in self._masks:
                    frozen = omega.detach().clone()
                    self._omega[spec.index] = frozen
                    self._masks[spec.index] = frozen * self.window >= self.threshold
                self._prepared.add(spec.index)
        return after_rotary

    def _query_hook(self, spec: LayerSpec):
        def replace_query(_module: Any, _args: Any, output: Any) -> torch.Tensor:
            if not self._inside_forward or spec.index not in self._prepared:
                raise RuntimeError("unsupported forward order: native rotary must run before Q through the supplied model")
            if not isinstance(output, torch.Tensor):
                raise TypeError("native Q projection/normalization output must be a tensor")
            key = (spec.index, output.device, output.dtype)
            if key not in self._factors:
                if max(self.high_scale, self.low_scale) > torch.finfo(output.dtype).max:
                    raise ValueError(f"band scale exceeds the finite range of Q dtype {output.dtype}")
                mask = self._masks[spec.index].to(device=output.device)
                plane = torch.full(mask.shape, self.low_scale, device=output.device, dtype=output.dtype)
                plane.masked_fill_(mask, self.high_scale)
                # split-half RoPE pairs coordinate n with n + rotary_dim/2.
                factors = torch.cat((plane, plane))
                indices = torch.tensor(self.heads[spec.index], device=output.device, dtype=torch.long)
                self._factors[key] = (indices, factors)
            indices, factors = self._factors[key]
            self._query_forwards[spec.index] += 1
            return _scale_query(output, spec, indices, factors, validate_finite=self.validate_finite)
        return replace_query

    def metadata(self) -> dict[str, Any]:
        """Return JSON-ready settings and frozen bands; explicit GPU sync here.

        Default non-CPU hooks avoid tensor-value synchronization;
        validate_finite=True adds a product-validation synchronization. Call
        this after inference when the frequency grid/mask is needed for an audit.
        Identity contexts have no captured grid because they register no hooks.
        """
        return {
            "operation": "q_only_band_score_scaling",
            "coefficient_multiplier": True,
            "norm_compensation": False,
            "validate_finite": self.validate_finite,
            "finite_validation": {"cpu": "always_for_nonidentity", "other_devices":
                                  "enabled_with_synchronization" if self.validate_finite else "disabled"},
            "band_rule": "window * native_omega >= threshold",
            "band_mask_frozen": "first_rotary_forward_per_layer",
            "window": self.window, "threshold": self.threshold,
            "high_scale": self.high_scale, "low_scale": self.low_scale,
            "identity": self._identity,
            "heads": {str(layer): list(chosen) for layer, chosen in self.heads.items()},
            "layers": {str(spec.index): {
                "query_forwards": self._query_forwards.get(spec.index, 0),
                "native_omega": self._omega[spec.index].detach().cpu().tolist()
                    if spec.index in self._omega else None,
                "high_mask": self._masks[spec.index].detach().cpu().tolist()
                    if spec.index in self._masks else None,
            } for spec in self._specs},
        }
