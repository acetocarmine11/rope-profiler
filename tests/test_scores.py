"""Independent finite-window references; these do not test Gaussian shape."""

import math

import pytest
import torch

import rope_profiler.scores as scores
from rope_profiler.scores import ScoreCache, score_activations


def _planes(vector, rotary_dim, layout):
    part = vector[:rotary_dim]
    if layout == "split_half":
        return part[:rotary_dim // 2], part[rotary_dim // 2:]
    return part[0::2], part[1::2]


def _direct_scores(query, key, omega, window, rotary_dim, layout, scale, rope_scale):
    """Rotate real coordinate pairs and take dot products at every distance."""
    qr, qi = _planes(query, rotary_dim, layout)
    kr, ki = _planes(key, rotary_dim, layout)
    phases = torch.arange(window, dtype=query.dtype)[:, None] * omega[None, :]
    rotated_real = qr * phases.cos() - qi * phases.sin()
    rotated_imag = qr * phases.sin() + qi * phases.cos()
    rotary = ((rotated_real * kr + rotated_imag * ki).sum(-1) * rope_scale ** 2)
    tail = (query[rotary_dim:] * key[rotary_dim:]).sum()
    return (rotary + tail) * scale


def _reference(q, k, mapping, pairs, triplets, omega, window, *, rotary_dim, layout, scale, rope_scale):
    semantic, positional = [], []
    for head, kv_head in enumerate(mapping):
        probabilities, normalized_gaps = [], []
        for query, first, second in triplets:
            a = _direct_scores(q[head, query], k[kv_head, first], omega, window, rotary_dim, layout, scale, rope_scale)
            b = _direct_scores(q[head, query], k[kv_head, second], omega, window, rotary_dim, layout, scale, rope_scale)
            margin = a - b
            if margin[0] == 0:
                continue
            oriented = margin * margin[0].sign()
            mean = oriented.mean()
            variance = oriented.var(correction=0)
            if variance > 0:
                probability = 0.5 * torch.erfc(mean / torch.sqrt(2 * variance))
            else:
                probability = (mean < 0).to(q.dtype)
            probabilities.append(probability)
        for query, key in pairs:
            series = _direct_scores(q[head, query], k[kv_head, key], omega, window, rotary_dim, layout, scale, rope_scale)
            qr, qi = _planes(q[head, query], rotary_dim, layout)
            kr, ki = _planes(k[kv_head, key], rotary_dim, layout)
            norm = torch.sqrt(((qr.square() + qi.square()) * (kr.square() + ki.square())).sum()) * scale * rope_scale ** 2
            if norm > 0:
                normalized_gaps.append((series[1:] - series[:-1]).abs().mean() / norm)
        semantic.append(1 - torch.stack(probabilities).mean() if probabilities else torch.tensor(torch.nan, dtype=q.dtype))
        positional.append(torch.stack(normalized_gaps).mean() if normalized_gaps else torch.tensor(torch.nan, dtype=q.dtype))
    return torch.stack(semantic), torch.stack(positional)


def _example(dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    q = torch.randn((4, 3, 8), generator=generator, dtype=dtype)
    k = torch.randn((2, 4, 8), generator=generator, dtype=dtype)
    return q, k, [0, 1, 0, 1], [[0, 0], [1, 2], [2, 3]], [[0, 0, 1], [1, 1, 2], [2, 2, 3]], torch.tensor([0.7, 0.13, 0.041, 0.003], dtype=dtype)


@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("work_dtype, tolerance", [("float64", 1e-11), ("float32", 3e-5)])
def test_analytic_gaussian_and_normalized_gap_match_direct_enumeration(layout, rotary_dim, work_dtype, tolerance):
    q, k, mapping, pairs, triplets, omega = _example(getattr(torch, work_dtype))
    omega = omega[:rotary_dim // 2]
    result = score_activations(q, k, mapping, pairs, triplets, omega, 23,
                               rotary_dim=rotary_dim, layout=layout, attention_scale=0.3,
                               rope_scale=1.7, distance_samples=None, dtype=work_dtype)
    semantic, positional = _reference(q, k, mapping, pairs, triplets, omega, 23,
                                      rotary_dim=rotary_dim, layout=layout, scale=0.3, rope_scale=1.7)
    torch.testing.assert_close(result.semantic_by_head, semantic, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(result.positional_by_head, positional, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(result.semantic_by_head, 1 - result.reversal_by_head)
    assert result.semantic_valid_triplets == 12
    assert result.positional_valid_pairs == 12
    assert result.invalid_cells == 0


def test_partial_rotary_tail_changes_semantic_but_not_positional_normalization():
    q = torch.tensor([[[1., 0., 4., 0.]]], dtype=torch.float64)
    k = torch.tensor([[[1., 0., 3., 0.], [-1., 0., -3., 0.]]], dtype=torch.float64)
    omega = torch.tensor([math.pi / 2], dtype=torch.float64)
    with_tail = score_activations(q, k, [0], [[0, 0]], [[0, 0, 1]], omega, 4,
                                  rotary_dim=2, attention_scale=1, distance_samples=None)
    without_tail = score_activations(q[..., :2], k[..., :2], [0], [[0, 0]], [[0, 0, 1]], omega, 4,
                                     attention_scale=1, distance_samples=None)
    torch.testing.assert_close(with_tail.positional_by_head, without_tail.positional_by_head)
    torch.testing.assert_close(with_tail.positional_by_head, torch.ones(1, dtype=torch.float64))
    assert with_tail.semantic_by_head > 0.99
    torch.testing.assert_close(without_tail.semantic_by_head, torch.full((1,), 0.5, dtype=torch.float64))


def test_ties_and_zero_norm_are_explicit_and_undefined_is_not_zero():
    q = torch.tensor([[[1., 0., 0., 0.]], [[0., 0., 0., 0.]]])
    k = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    result = score_activations(q, k, [0, 0], [[0, 0]], [[0, 0, 1]], [0.4, 0.1], 9)
    assert torch.isnan(result.semantic_by_head).all()
    assert result.semantic_ties == 2
    assert result.semantic_valid_triplets == 0
    assert result.positional_valid_pairs == 1
    assert torch.isnan(result.positional_by_head[1])
    assert result.invalid_cells == 1


def test_zero_frequency_is_deterministic_semantic_order_and_zero_positional_response():
    q = torch.tensor([[[1., 0.]]])
    k = torch.tensor([[[2., 0.], [1., 0.]]])
    result = score_activations(q, k, [0], [[0, 0]], [[0, 0, 1]], [0.], 20, distance_samples=None)
    torch.testing.assert_close(result.semantic_by_head, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(result.positional_by_head, torch.zeros(1, dtype=torch.float64))
    assert result.invalid_cells == 0


def test_tiny_native_frequency_survives_kernel_evaluation():
    q = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    k = torch.tensor([[[2., 0.], [1., 0.]]], dtype=torch.float64)
    result = score_activations(q, k, [0], [[0, 0]], [[0, 0, 1]], [1e-13], 100_003,
                               distance_samples=7)
    assert torch.isfinite(result.semantic_by_head).all()
    assert torch.isfinite(result.positional_by_head).all()
    assert result.invalid_cells == 0


def test_sampling_is_seeded_and_estimates_same_score_as_explicit_sampled_distances():
    q, k, mapping, pairs, triplets, omega = _example()
    cache = ScoreCache()
    result = score_activations(q, k, mapping, pairs, triplets, omega, 601,
                               distance_samples=17, seed=19, cache=cache)
    generator = torch.Generator().manual_seed(19)
    distances = torch.randint(600, (17,), generator=generator)
    expected = []
    for head, kv_head in enumerate(mapping):
        values = []
        for query, key in pairs:
            series = _direct_scores(q[head, query], k[kv_head, key], omega, 601, 8, "split_half", 8 ** -0.5, 1)
            qr, qi = _planes(q[head, query], 8, "split_half")
            kr, ki = _planes(k[kv_head, key], 8, "split_half")
            norm = torch.sqrt(((qr.square() + qi.square()) * (kr.square() + ki.square())).sum()) / math.sqrt(8)
            values.append((series[distances + 1] - series[distances]).abs().mean() / norm)
        expected.append(torch.stack(values).mean())
    torch.testing.assert_close(result.positional_by_head, torch.stack(expected), rtol=1e-12, atol=1e-12)
    cache.clear()
    repeated = score_activations(q, k, mapping, pairs, triplets, omega, 601,
                                 distance_samples=17, seed=19, cache=cache)
    assert torch.equal(result.positional_by_head, repeated.positional_by_head)
    changed = score_activations(q, k, mapping, pairs, triplets, omega, 601,
                                distance_samples=17, seed=20, cache=cache)
    assert not torch.equal(result.positional_by_head, changed.positional_by_head)
    assert result.distance_sample_count == 17
    torch.testing.assert_close(result.semantic_by_head, changed.semantic_by_head)


def test_window_two_and_sample_count_over_window_use_exhaustive_distances():
    q, k, mapping, pairs, triplets, omega = _example()
    sampled = score_activations(q, k, mapping, pairs, triplets, omega, 2, distance_samples=256)
    exhaustive = score_activations(q, k, mapping, pairs, triplets, omega, 2, distance_samples=None)
    assert sampled.distance_sample_count == 1
    torch.testing.assert_close(sampled.positional_by_head, exhaustive.positional_by_head)


def test_positive_attention_scaling_cancels_in_both_full_rotary_scores():
    q, k, mapping, pairs, triplets, omega = _example()
    first = score_activations(q, k, mapping, pairs, triplets, omega, 41, attention_scale=0.1, rope_scale=0.7)
    second = score_activations(q, k, mapping, pairs, triplets, omega, 41, attention_scale=4, rope_scale=2.3)
    torch.testing.assert_close(first.semantic_by_head, second.semantic_by_head, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(first.positional_by_head, second.positional_by_head, rtol=1e-12, atol=1e-12)


def test_cache_is_bounded_tracks_dtype_and_invalidates_mutated_native_frequencies():
    q, k, mapping, pairs, triplets, omega = _example()
    cache = ScoreCache(max_entries=3)
    first = score_activations(q, k, mapping, pairs, triplets, omega, 47, cache=cache)
    assert cache.misses == 2
    first_metadata = cache.metadata()
    assert first_metadata["entries"] == 2
    assert first_metadata["max_entries"] == 3
    assert first_metadata["hits"] == 0
    assert first_metadata["misses"] == 2
    assert first_metadata["cached_tensor_bytes"] > 0
    score_activations(q, k, mapping, pairs, triplets, omega, 47, cache=cache)
    assert cache.hits == 2
    assert cache.metadata() == {**first_metadata, "hits": 2}
    omega.add_(0.2)
    changed = score_activations(q, k, mapping, pairs, triplets, omega, 47, cache=cache)
    assert cache.misses == 4
    assert len(cache) == 3
    assert not torch.allclose(first.positional_by_head, changed.positional_by_head)
    score_activations(q, k, mapping, pairs, triplets, omega, 47, dtype="float32", cache=cache)
    assert cache.misses == 6
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.metadata() == {
        "entries": 0, "max_entries": 3, "hits": 0, "misses": 0,
        "cached_tensor_bytes": 0,
    }


def test_cache_metadata_counts_unique_value_storage_and_excludes_source_anchors(monkeypatch):
    cache = ScoreCache(max_entries=3)
    native_source = torch.ones(1000)
    base = torch.arange(12, dtype=torch.float32)
    view = base[1:4]
    independent_copy = base[:2].clone()
    cache._get(("first",), lambda: {"full": base, "sliced": view,
                                   "nested": [base, independent_copy]}, native_source)
    cache._get(("second",), lambda: (view, independent_copy), native_source)
    cache._get(("first",), lambda: pytest.fail("cache hit rebuilt the payload"), native_source)

    def forbid_value_readback(*args, **kwargs):
        pytest.fail("cache metadata read tensor values back to the host")

    for method in ("cpu", "item", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbid_value_readback)

    # A three-element view keeps the original twelve-element storage alive;
    # both entries share it. Only the independent two-element copy adds bytes.
    assert cache.metadata() == {
        "entries": 2, "max_entries": 3, "hits": 1, "misses": 2,
        "cached_tensor_bytes": (12 + 2) * 4,
    }


def test_nonfinite_activations_are_counted_and_do_not_poison_other_heads():
    q, k, mapping, pairs, triplets, omega = _example()
    q[1, :, 0] = torch.nan
    result = score_activations(q, k, mapping, pairs, triplets, omega, 23)
    assert torch.isnan(result.semantic_by_head[1])
    assert torch.isnan(result.positional_by_head[1])
    assert result.semantic_invalid_cells == 3
    assert result.positional_invalid_cells == 3
    assert result.invalid_cells == 6
    assert torch.isfinite(result.semantic_by_head[[0, 2, 3]]).all()


def test_materially_negative_float32_variance_is_excluded_and_counted(monkeypatch):
    original = scores._moment_kernels

    def broken(omega, window):
        mean, minus, plus = original(omega, window)
        return mean, -100 * torch.eye(len(omega), device=omega.device, dtype=minus.dtype), torch.zeros_like(plus)

    monkeypatch.setattr(scores, "_moment_kernels", broken)
    q, k, mapping, pairs, triplets, omega = _example(torch.float32)
    result = score_activations(q, k, mapping, pairs, triplets, omega, 29, dtype="float32", cache=ScoreCache())
    assert result.negative_variance_cells == 12
    assert result.semantic_invalid_cells == 12
    assert result.semantic_valid_triplets == 0
    assert torch.isnan(result.semantic_by_head).all()
    assert torch.isfinite(result.positional_by_head).all()


def test_empty_cell_collections_return_nan_and_zero_counts_on_input_device():
    q, k, mapping, _, _, omega = _example()
    result = score_activations(q, k, mapping, [], [], omega, 3)
    assert torch.isnan(result.semantic_by_head).all()
    assert torch.isnan(result.positional_by_head).all()
    assert result.invalid_cells == 0
    assert result.semantic_valid_triplets == 0
    assert result.positional_valid_pairs == 0
    for value in vars(result).values():
        assert value.device == q.device
        assert not value.requires_grad


@pytest.mark.parametrize("kwargs", [{"window": 1}, {"window": True}, {"distance_samples": 0},
                                     {"rotary_dim": 3}, {"layout": "other"}, {"dtype": "bfloat16"}])
def test_rejects_invalid_configuration(kwargs):
    q, k, mapping, pairs, triplets, omega = _example()
    options = {"window": 23, **kwargs}
    with pytest.raises(ValueError):
        score_activations(q, k, mapping, pairs, triplets, omega, **options)


@pytest.mark.parametrize("bad_indices", [[[-1, 0]], [[0, -1]], [[0, 4]]])
def test_indices_outside_native_token_arrays_do_not_wrap(bad_indices):
    q, k, mapping, _, triplets, omega = _example()
    with pytest.raises((IndexError, RuntimeError)):
        score_activations(q, k, mapping, bad_indices, triplets, omega, 23)


def test_rejects_fractional_indices_and_negative_head_mapping():
    q, k, mapping, pairs, triplets, omega = _example()
    with pytest.raises(ValueError, match="integer"):
        score_activations(q, k, mapping, [[0, 0.5]], triplets, omega, 23)
    with pytest.raises(ValueError, match="integer"):
        score_activations(q, k, [0., 1., 0., 1.], pairs, triplets, omega, 23)
    with pytest.raises((IndexError, RuntimeError)):
        score_activations(q, k, [0, -1, 0, 1], pairs, triplets, omega, 23)
