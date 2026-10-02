import pytest
import torch

from rope_profiler import token_region_masks


def test_regions_exclude_special_padding_and_punctuation_and_use_overlap():
    result = token_region_masks(
        ["abc ! Who?"], [[[0, 3], [4, 5], [6, 9], [9, 10], [0, 0], [0, 3]]],
        context_spans=[(0, 5)], question_spans=[(6, 10)],
        special_tokens_mask=[[0, 0, 0, 0, 0, 1]],
    )
    assert result["key_mask"].tolist() == [[True, False, False, False, False, False]]
    assert result["query_mask"].tolist() == [[False, False, True, False, False, False]]
    assert result["query_mask"].dtype == torch.bool
    assert result["query_mask"].device.type == "cpu"


def test_partial_overlap_uses_only_characters_inside_region():
    result = token_region_masks(["!!abc"], [[[0, 5]]], context_spans=[(0, 2)],
                               question_spans=[(2, 5)], special_tokens_mask=[[0]])
    assert not result["key_mask"].any()
    assert result["query_mask"].all()


@pytest.mark.parametrize("offsets,question,special", [
    ([[[0.5, 2.0]]], (0, 2), [[0]]),
    ([[[0, 8]]], (0, 2), [[0]]),
    ([[[0, 2]]], (0, 5), [[0]]),
    ([[[0, 2]]], (0, 2), [[2]]),
])
def test_regions_reject_misaligned_metadata(offsets, question, special):
    with pytest.raises(ValueError):
        token_region_masks(["abc"], offsets, question_spans=[question],
                           context_spans=[(0, 2)], special_tokens_mask=special)
