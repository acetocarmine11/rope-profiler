"""Finite-window RoPE diagnostics for a sparse collection of attention cells.

Semantic scores use analytic moments of the accepted Gaussian reversal
approximation. Positional scores average absolute adjacent changes, normalized
by each pair's rotary coefficient norm. Sampling distances estimates that same
finite-window average; it does not change the definition of the score.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScoreResult:
    """Per-head scores and counts, all on the activations' device.

    An undefined head is represented by NaN, accompanied by a zero valid count.
    Reference ties are counted separately from numerical/zero-norm failures.
    """

    semantic_by_head: Tensor
    positional_by_head: Tensor
    semantic_valid_triplets: Tensor
    semantic_ties: Tensor
    positional_valid_pairs: Tensor
    invalid_cells: Tensor
    reversal_by_head: Tensor
    semantic_valid_triplets_by_head: Tensor
    positional_valid_pairs_by_head: Tensor
    semantic_invalid_cells: Tensor
    positional_invalid_cells: Tensor
    negative_variance_cells: Tensor
    roundoff_variance_clamps: Tensor
    distance_sample_count: Tensor


class ScoreCache:
    """Bounded cache for frequency/window kernels and sampled distance phases.

    Tensor frequencies are keyed by their identity and PyTorch mutation version,
    so checking a GPU cache never copies frequencies to the host. Pass the same
    runtime frequency tensor on consecutive calls. Equal-valued new tensors are
    safe but incur a cache miss. Inference tensors without version tracking are
    not cached. Sequence frequencies are keyed by their numerical values.
    """

    def __init__(self, max_entries: int = 16) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def metadata(self) -> dict[str, int]:
        """Return counters and unique storage bytes retained by cached values.

        Storage metadata is read without copying tensor values to the host.
        Views and repeated tensors count their shared backing storage once,
        including across entries. Source anchors are excluded: retaining a
        reference to a model's frequency buffer does not allocate that buffer.
        This is a payload-storage count, not allocator or Python-object memory.
        """
        storages: set[tuple[str, int | None, int]] = set()
        containers: set[int] = set()

        def storage_bytes(value: Any) -> int:
            if isinstance(value, Tensor):
                storage = value.untyped_storage()
                identity = (value.device.type, value.device.index, storage.data_ptr())
                if identity in storages:
                    return 0
                storages.add(identity)
                return storage.nbytes()
            if isinstance(value, (dict, tuple, list)):
                if id(value) in containers:
                    return 0
                containers.add(id(value))
                children = value.values() if isinstance(value, dict) else value
                return sum(storage_bytes(child) for child in children)
            return 0

        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "cached_tensor_bytes": sum(storage_bytes(value) for value, _ in self._entries.values()),
        }

    def _get(
        self, key: tuple[Any, ...] | None, make: Callable[[], Any], source: Any,
    ) -> Any:
        if key is not None and key in self._entries:
            self.hits += 1
            value, anchor = self._entries.pop(key)
            self._entries[key] = (value, anchor)
            return value
        self.misses += 1
        value = make()
        if key is not None:
            # Holding the source prevents object-identity reuse before eviction.
            self._entries[key] = (value, source)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value


_DEFAULT_CACHE = ScoreCache()
_CELL_CHUNK = 1024
_DISTANCE_CHUNK = 2048
_MAX_CACHED_PHASE_ELEMENTS = 1_048_576


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if value in ("float64", torch.float64):
        return torch.float64
    if value in ("float32", torch.float32):
        return torch.float32
    raise ValueError("dtype must be float64 or float32")


def _frequency_key(omega: Tensor | Sequence[float]) -> tuple[Any, ...] | None:
    if isinstance(omega, Tensor):
        try:
            version = omega._version
        except RuntimeError:
            return None
        return ("tensor", id(omega), version, str(omega.device), omega.dtype, tuple(omega.shape))
    return ("values", tuple(float(value) for value in omega))


def _mean_phase(angle: Tensor, window: int) -> Tensor:
    # Preserve very small native frequencies: modulo arithmetic can erase them.
    wrapped = torch.where(
        angle.abs() <= math.pi,
        angle,
        torch.remainder(angle + math.pi, 2 * math.pi) - math.pi,
    )
    return (
        torch.exp(0.5j * (window - 1) * wrapped)
        * torch.sinc(window * wrapped / (2 * math.pi))
        / torch.sinc(wrapped / (2 * math.pi))
    )


def _moment_kernels(omega: Tensor, window: int) -> tuple[Tensor, Tensor, Tensor]:
    mean = _mean_phase(omega, window)
    minus = _mean_phase(omega[:, None] - omega[None, :], window)
    minus = minus - mean[:, None] * mean.conj()[None, :]
    plus = _mean_phase(omega[:, None] + omega[None, :], window)
    plus = plus - mean[:, None] * mean[None, :]
    return mean, minus, plus


def _indices(value: Any, width: int, device: torch.device, name: str) -> Tensor:
    result = torch.as_tensor(value, device=device)
    if result.numel() == 0:
        return result.to(dtype=torch.long).reshape(0, width)
    if result.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indices")
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape [cells,{width}]")
    return result.to(dtype=torch.long)


def _complex_rotary(value: Tensor, rotary_dim: int, layout: str) -> Tensor:
    rotary = value[..., :rotary_dim]
    if layout == "split_half":
        half = rotary_dim // 2
        return torch.complex(rotary[..., :half], rotary[..., half:])
    if layout == "interleaved":
        return torch.complex(rotary[..., 0::2], rotary[..., 1::2])
    raise ValueError("layout must be split_half or interleaved")


def _valid_mean(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    counts = valid.sum(dim=-1)
    totals = torch.where(valid, values, 0).sum(dim=-1)
    means = totals / counts.clamp_min(1)
    return torch.where(counts > 0, means, torch.nan), counts


def _distances(window: int, samples: int | None, seed: int, device: torch.device) -> Tensor:
    count = window - 1
    if samples is None or samples >= count:
        return torch.arange(count, device=device, dtype=torch.long)
    # A CPU generator makes fixed grids identical on CPU and CUDA. No device
    # results are copied to the host. Replacement gives O(samples) construction.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(count, (samples,), generator=generator).to(device=device)


@torch.inference_mode()
def score_activations(
    q: Tensor,
    k: Tensor,
    query_to_kv: Tensor | Sequence[int],
    pair_indices: Tensor | Sequence[Sequence[int]],
    triplet_indices: Tensor | Sequence[Sequence[int]],
    omega: Tensor | Sequence[float],
    window: int,
    *,
    layout: str = "split_half",
    rotary_dim: int | None = None,
    attention_scale: float | None = None,
    rope_scale: float = 1.0,
    distance_samples: int | None = 256,
    seed: int = 0,
    dtype: str | torch.dtype = "float64",
    cache: ScoreCache | None = None,
) -> ScoreResult:
    """Score pre-RoPE activations after the model's native Q/K normalization.

    ``q`` is [Hq,Q,D], ``k`` is [Hkv,K,D], and query_to_kv maps each query
    head to its native KV head. Pair rows address (query,key), and triplet rows
    address (query,key1,key2). ``omega`` is the actual runtime inverse frequency
    vector, including RoPE scaling. ``rope_scale`` is the amplitude applied to
    each rotary Q/K vector. It is squared in rotary score coefficients; the
    non-rotary tail retains only attention_scale.

    Semantic triplets are oriented by their current zero-distance ordering.
    Positional normalization uses rotary coefficients only. distance_samples
    selects a reproducible uniform sample with replacement; None enumerates
    the full window. At least window-1 samples also select full enumeration.
    No tensor value is read back to the host. Invalid numerical cells are
    excluded and counted, including materially negative variances in float32.
    """
    work_dtype = _dtype(dtype)
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if distance_samples is not None and (
        isinstance(distance_samples, bool) or not isinstance(distance_samples, int) or distance_samples < 1
    ):
        raise ValueError("distance_samples must be a positive integer or None")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(q, Tensor) or not isinstance(k, Tensor):
        raise TypeError("q and k must be torch tensors")
    if q.ndim != 3 or k.ndim != 3 or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have [heads,tokens,head_dim] shapes with equal head_dim")
    heads, _, head_dim = q.shape
    if heads < 1 or k.shape[0] < 1 or head_dim < 1:
        raise ValueError("q/k head counts and head_dim must be positive")
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int) or not 2 <= rotary_dim <= head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and lie between 2 and head_dim")
    if layout not in {"split_half", "interleaved"}:
        raise ValueError("layout must be split_half or interleaved")
    attention_scale = head_dim ** -0.5 if attention_scale is None else float(attention_scale)
    rope_scale = float(rope_scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("attention_scale must be finite and positive")
    if not math.isfinite(rope_scale):
        raise ValueError("rope_scale must be finite")

    device = q.device
    q = q.detach().to(dtype=work_dtype)
    k = k.detach().to(device=device, dtype=work_dtype)
    frequencies = torch.as_tensor(omega, device=device, dtype=work_dtype).detach()
    if frequencies.ndim != 1 or frequencies.shape[0] != rotary_dim // 2:
        raise ValueError("omega must have rotary_dim/2 elements")
    mapping = torch.as_tensor(query_to_kv, device=device)
    if mapping.shape != (heads,):
        raise ValueError("query_to_kv must contain one index per query head")
    if mapping.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("query_to_kv must contain integer indices")
    mapping = mapping.to(dtype=torch.long)
    pairs = _indices(pair_indices, 2, device, "pair_indices")
    triplets = _indices(triplet_indices, 3, device, "triplet_indices")
    q_rotary = _complex_rotary(q, rotary_dim, layout)
    k_rotary = _complex_rotary(k, rotary_dim, layout)
    scale = attention_scale * rope_scale ** 2
    finite_frequencies = torch.isfinite(frequencies).all()
    cache = _DEFAULT_CACHE if cache is None else cache
    signature = _frequency_key(omega)
    base_key = None if signature is None else (signature, window, str(device), work_dtype)
    zero = torch.zeros((), device=device, dtype=torch.long)
    zero_heads = torch.zeros(heads, device=device, dtype=torch.long)
    nan_heads = torch.full((heads,), torch.nan, device=device, dtype=work_dtype)

    semantic_counts = zero_heads
    semantic_ties = zero
    semantic_invalid = zero
    negative_variances = zero
    roundoff_clamps = zero
    reversal = nan_heads
    if triplets.shape[0]:
        ti, ta, tb = triplets.unbind(dim=1)
        selected_q = q_rotary.index_select(1, ti)
        first_k = k_rotary.index_select(1, ta).index_select(0, mapping)
        second_k = k_rotary.index_select(1, tb).index_select(0, mapping)
        difference = selected_q * (first_k - second_k).conj() * scale
        tail = torch.zeros(difference.shape[:-1], device=device, dtype=work_dtype)
        if rotary_dim < head_dim:
            tail_q = q[..., rotary_dim:].index_select(1, ti)
            tail_first = k[..., rotary_dim:].index_select(1, ta).index_select(0, mapping)
            tail_second = k[..., rotary_dim:].index_select(1, tb).index_select(0, mapping)
            tail = (tail_q * (tail_first - tail_second)).sum(-1) * attention_scale
        kernel_key = None if base_key is None else ("moments", *base_key)
        mean_kernel, covariance_minus, covariance_plus = cache._get(
            kernel_key, lambda: _moment_kernels(frequencies, window), omega,
        )
        flat = difference.reshape(-1, frequencies.shape[0])
        means, variances = [], []
        for block in flat.split(_CELL_CHUNK):
            means.append((block @ mean_kernel).real)
            variances.append(0.5 * (
                ((block @ covariance_minus) * block.conj()).sum(-1).real
                + ((block @ covariance_plus) * block).sum(-1).real
            ))
        mean = torch.cat(means).reshape(tail.shape) + tail
        variance = torch.cat(variances).reshape(tail.shape)
        reference = difference.real.sum(-1) + tail
        coefficient_mass = difference.abs().sum(-1)
        eps = torch.finfo(work_dtype).eps
        reference_tolerance = 64 * eps * (coefficient_mass + tail.abs())
        variance_tolerance = 128 * eps * coefficient_mass.square()
        finite = (
            torch.isfinite(difference).all(-1) & torch.isfinite(tail)
            & torch.isfinite(reference) & torch.isfinite(mean)
            & torch.isfinite(variance) & finite_frequencies
        )
        negative = variance < -variance_tolerance
        roundoff = (variance < 0) & ~negative & finite
        negative_variances = (negative & finite).sum()
        roundoff_clamps = roundoff.sum()
        variance = variance.clamp_min(0)
        oriented = mean * reference.sign()
        variable = variance > 0
        denominator = torch.where(variable, variance.sqrt() * math.sqrt(2), 1)
        probability = torch.where(
            variable,
            0.5 * torch.erfc(oriented / denominator),
            (oriented < 0).to(work_dtype),
        )
        tied = finite & (reference.abs() <= reference_tolerance)
        invalid = ~finite | negative | ~torch.isfinite(probability)
        valid = ~invalid & ~tied
        reversal, semantic_counts = _valid_mean(probability, valid)
        semantic_ties = tied.sum()
        semantic_invalid = invalid.sum()

    positional_counts = zero_heads
    positional_invalid = zero
    positional = nan_heads
    distance_count = zero
    if pairs.shape[0]:
        pi, pk = pairs.unbind(dim=1)
        coefficient = (
            q_rotary.index_select(1, pi)
            * k_rotary.index_select(1, pk).index_select(0, mapping).conj() * scale
        )
        norm = coefficient.abs().square().sum(-1).sqrt()
        derivative = coefficient * torch.expm1(1j * frequencies)
        sampled_count = min(window - 1, distance_samples or window - 1)
        effective_seed = seed if sampled_count < window - 1 else None
        phase_key = None if base_key is None else ("phases", *base_key, sampled_count, effective_seed)

        def make_phases() -> tuple[Tensor, Tensor | None, Tensor | None]:
            positions = _distances(window, distance_samples, seed, device)
            if frequencies.numel() * positions.numel() > _MAX_CACHED_PHASE_ELEMENTS:
                return positions, None, None
            phase = frequencies[:, None] * positions[None, :]
            return positions, phase.cos(), phase.sin()

        positions, cosine, sine = cache._get(phase_key, make_phases, omega)
        flat = derivative.reshape(-1, frequencies.shape[0])
        sums = []
        for block in flat.split(_CELL_CHUNK):
            total = torch.zeros(block.shape[0], device=device, dtype=work_dtype)
            real, imag = block.real.contiguous(), block.imag.contiguous()
            for start in range(0, positions.numel(), _DISTANCE_CHUNK):
                end = start + _DISTANCE_CHUNK
                if cosine is None:
                    phase = frequencies[:, None] * positions[None, start:end]
                    cos, sin = phase.cos(), phase.sin()
                else:
                    cos, sin = cosine[:, start:end], sine[:, start:end]
                gap = real @ cos - imag @ sin
                total += gap.abs().sum(-1)
            sums.append(total)
        raw = torch.cat(sums).reshape(norm.shape) / positions.numel()
        valid = (
            (norm > 0) & torch.isfinite(norm) & torch.isfinite(coefficient).all(-1)
            & torch.isfinite(raw) & finite_frequencies
        )
        # Substituting one only in an invalid denominator prevents spurious
        # division warnings; the mask retains these cells as undefined.
        normalized = raw / torch.where(valid, norm, 1)
        valid = valid & torch.isfinite(normalized)
        positional, positional_counts = _valid_mean(normalized, valid)
        positional_invalid = (~valid).sum()
        distance_count = torch.as_tensor(positions.numel(), device=device, dtype=torch.long)

    return ScoreResult(
        semantic_by_head=1 - reversal,
        positional_by_head=positional,
        semantic_valid_triplets=semantic_counts.sum(),
        semantic_ties=semantic_ties,
        positional_valid_pairs=positional_counts.sum(),
        invalid_cells=semantic_invalid + positional_invalid,
        reversal_by_head=reversal,
        semantic_valid_triplets_by_head=semantic_counts,
        positional_valid_pairs_by_head=positional_counts,
        semantic_invalid_cells=semantic_invalid,
        positional_invalid_cells=positional_invalid,
        negative_variance_cells=negative_variances,
        roundoff_variance_clamps=roundoff_clamps,
        distance_sample_count=distance_count,
    )


__all__ = ["ScoreCache", "ScoreResult", "score_activations"]
