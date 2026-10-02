"""Build question/context eligibility masks from a tokenizer's character offsets."""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch


def token_region_masks(
    texts: Sequence[str],
    offset_mapping: torch.Tensor | Sequence[Sequence[Sequence[int]]],
    *,
    question_spans: Sequence[tuple[int, int]],
    context_spans: Sequence[tuple[int, int]],
    special_tokens_mask: torch.Tensor | Sequence[Sequence[bool]],
) -> dict[str, torch.Tensor]:
    """Return Boolean CPU masks for question queries and context keys.

    Offsets and spans must refer to the exact texts tokenized for the model,
    including its chat template. Padding and special tokens are excluded.
    A token needs a nonempty overlap containing a letter or digit. These masks
    match the paper's eligibility rule; token selection retains the monitor's
    own seeded sampler and does not reproduce the historical SHA ordering.
    """
    offsets = torch.as_tensor(offset_mapping)
    special = torch.as_tensor(special_tokens_mask)
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError("offset_mapping must have shape [batch,sequence,2]")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offset_mapping must contain integer character offsets")
    if tuple(special.shape) != tuple(offsets.shape[:2]):
        raise ValueError("special_tokens_mask must match [batch,sequence]")
    count = offsets.shape[0]
    if isinstance(texts, str) or any(len(items) != count for items in (texts, question_spans, context_spans)):
        raise ValueError("texts and each span list must contain one entry per batch row")
    offset_rows, special_rows = offsets.detach().cpu().tolist(), special.detach().cpu().tolist()
    query = torch.zeros(offsets.shape[:2], dtype=torch.bool)
    key = torch.zeros_like(query)
    for row, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError("texts must contain strings")
        for span in (question_spans[row], context_spans[row]):
            if len(span) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in span):
                raise ValueError("each character span must contain two integer offsets")
            if not 0 <= span[0] <= span[1] <= len(text):
                raise ValueError("character spans must lie within their tokenized text")
        for column, (start, end) in enumerate(offset_rows[row]):
            if not 0 <= start <= end <= len(text):
                raise ValueError("token offsets must lie within their tokenized text")
            if special_rows[row][column] not in (0, 1):
                raise ValueError("special_tokens_mask must contain Boolean or binary values")
            if special_rows[row][column] or start == end:
                continue
            for output, span in ((query, question_spans[row]), (key, context_spans[row])):
                left, right = max(start, span[0]), min(end, span[1])
                output[row, column] = left < right and any(character.isalnum() for character in text[left:right])
    return {"query_mask": query, "key_mask": key}
