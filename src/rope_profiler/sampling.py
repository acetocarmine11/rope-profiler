"""Deterministic, content-keyed token sampling for repeated evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
import hashlib
import random
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    query_positions: tuple[int, ...]
    key_positions: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    triplet_indices: tuple[tuple[int, int, int], ...]
    window: int


def input_fingerprint(tokens: Sequence[int]) -> str:
    # Logical, unpadded tokens: stable across left/right padding and batch sizes.
    payload = ",".join(map(str, tokens)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sample_tokens(
    tokens: Sequence[int],
    valid: Sequence[bool],
    *,
    queries: int = 4,
    keys: int = 16,
    seed: int = 0,
    sample_id: str | None = None,
    query_mask: Sequence[bool] | None = None,
    key_mask: Sequence[bool] | None = None,
    position_ids: Sequence[int] | None = None,
) -> SamplePlan | None:
    """Sample shared causal keys and queries; return padded-array column indices.

    Explicit query/key masks define question/context regions. Without masks,
    query candidates are valid tokens with enough preceding context to retain
    the key budget (or half the input for short sequences). Keys are uniformly
    sampled without replacement from the prefix before the first query.
    No tokenizer, target label, or generated answer is required.
    """
    if queries < 1 or keys < 2:
        raise ValueError("queries must be >=1 and keys must be >=2")
    if len(tokens) != len(valid):
        raise ValueError("token and validity lengths differ")
    for mask in (query_mask, key_mask, position_ids):
        if mask is not None and len(mask) != len(tokens):
            raise ValueError("sampling metadata must have the input sequence length")
    eligible = [i for i, active in enumerate(valid) if active]
    if len(eligible) < 3:
        return None
    if any(b != a + 1 for a, b in zip(eligible, eligible[1:])):
        raise ValueError("attention_mask must identify one contiguous input; packed/internal gaps need an adapter")
    if position_ids is not None:
        if any(isinstance(position_ids[i], bool) or not isinstance(position_ids[i], Integral) for i in eligible):
            raise ValueError("position_ids must contain integer positions")
        logical_positions = [int(position_ids[i]) for i in eligible]
        if any(b != a + 1 for a, b in zip(logical_positions, logical_positions[1:])):
            raise ValueError("packed/reset/noncontiguous position_ids need a dedicated adapter")
    identity = str(sample_id) if sample_id is not None else input_fingerprint([tokens[i] for i in eligible])
    digest = hashlib.sha256(f"rope-monitor-v1:{seed}:{identity}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:16], "little"))
    key_candidates = [i for i in eligible if key_mask is None or key_mask[i]]
    minimum_prefix = 2 if query_mask is not None else min(keys, max(2, len(eligible) // 2))
    query_candidates = [i for i in eligible
                        if (query_mask is None or query_mask[i])
                        and bisect_left(key_candidates, i) >= minimum_prefix]
    if not query_candidates:
        return None
    query_positions = tuple(sorted(rng.sample(query_candidates, min(queries, len(query_candidates)))))
    key_candidates = [i for i in key_candidates if i < min(query_positions)]
    key_positions = tuple(sorted(rng.sample(key_candidates, min(keys, len(key_candidates)))))
    if len(key_positions) < 2:
        return None
    pairs = tuple((q, k) for q in range(len(query_positions)) for k in range(len(key_positions)))
    key_order = list(range(len(key_positions)))
    rng.shuffle(key_order)
    triplets = tuple((q, key_order[j], key_order[j + 1])
                     for q in range(len(query_positions))
                     for j in range(0, len(key_order) - 1, 2))
    return SamplePlan(identity, query_positions, key_positions, pairs, triplets, len(eligible))
