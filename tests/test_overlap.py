from __future__ import annotations

import numpy as np
import pytest

from fastlabelops import overlap_counts


def reference_counts(a, b, *, include_background=False):
    pairs: dict[tuple[int, int], int] = {}
    for av, bv in zip(a.flat, b.flat, strict=True):
        ai = int(av)
        bi = int(bv)
        if not include_background and (ai == 0 or bi == 0):
            continue
        key = (ai, bi)
        pairs[key] = pairs.get(key, 0) + 1
    a_ids = np.asarray([key[0] for key in pairs], dtype=a.dtype)
    b_ids = np.asarray([key[1] for key in pairs], dtype=b.dtype)
    counts = np.asarray(list(pairs.values()), dtype=np.uint64)
    return a_ids, b_ids, counts


def assert_matches_reference(a, b, *, include_background=False):
    actual = overlap_counts(a, b, include_background=include_background)
    expected = reference_counts(a, b, include_background=include_background)
    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


def test_basic_foreground_overlap():
    a = np.array([[0, 7, 7, 7], [20, 20, 0, 0]], dtype=np.uint32)
    b = np.array([[0, 4, 4, 9], [9, 9, 0, 3]], dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert np.array_equal(a_ids, np.array([7, 7, 20], dtype=np.uint32))
    assert np.array_equal(b_ids, np.array([4, 9, 9], dtype=np.uint32))
    assert np.array_equal(counts, np.array([2, 1, 2], dtype=np.uint64))


def test_background_can_be_included():
    a = np.array([0, 7, 7, 0, 20], dtype=np.uint32)
    b = np.array([0, 4, 0, 3, 0], dtype=np.uint32)
    assert_matches_reference(a, b, include_background=True)


def test_all_background_is_empty_by_default():
    a = np.zeros((8, 8), dtype=np.uint32)
    b = np.zeros((8, 8), dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert a_ids.size == b_ids.size == counts.size == 0


def test_empty_arrays():
    a = np.empty((0, 4), dtype=np.uint32)
    b = np.empty((0, 4), dtype=np.uint64)
    assert_matches_reference(a, b)


def test_mixed_uint32_uint64_preserves_output_dtypes():
    a = np.array([1, 1, 2, 2], dtype=np.uint32)
    b = np.array([5, 5, 2**40, 2**40], dtype=np.uint64)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert a_ids.dtype == np.uint32
    assert b_ids.dtype == np.uint64
    assert counts.dtype == np.uint64
    assert_matches_reference(a, b)


def test_uint64_ids_above_uint32_range():
    high_a = 2**48 + 123
    high_b = 2**52 + 456
    a = np.array([high_a, high_a, 9, high_a], dtype=np.uint64)
    b = np.array([high_b, high_b, high_b, 11], dtype=np.uint64)
    assert_matches_reference(a, b)


def test_first_occurrence_order_is_deterministic():
    a = np.array([8, 2, 8, 3, 2], dtype=np.uint32)
    b = np.array([4, 9, 4, 1, 7], dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert list(zip(a_ids.tolist(), b_ids.tolist(), strict=True)) == [(8, 4), (2, 9), (3, 1), (2, 7)]
    assert counts.tolist() == [2, 1, 1, 1]


def test_arbitrary_nd_shape():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 12, size=(4, 5, 6), dtype=np.uint32)
    b = rng.integers(0, 15, size=(4, 5, 6), dtype=np.uint32)
    assert_matches_reference(a, b)


def test_noncontiguous_inputs_are_supported_via_contiguous_copy():
    a0 = np.arange(64, dtype=np.uint32).reshape(8, 8) % 7
    b0 = np.arange(64, dtype=np.uint32).reshape(8, 8) % 5
    a = a0[:, ::2]
    b = b0[:, ::2]
    assert not a.flags.c_contiguous
    assert not b.flags.c_contiguous
    assert_matches_reference(a, b)


def test_read_only_inputs_are_supported():
    a = np.array([1, 1, 2], dtype=np.uint32)
    b = np.array([3, 3, 4], dtype=np.uint32)
    a.flags.writeable = False
    b.flags.writeable = False
    assert_matches_reference(a, b)


def test_repeated_pair_separated_by_background_is_counted_together():
    a = np.array([7, 0, 7, 0, 7], dtype=np.uint32)
    b = np.array([9, 0, 9, 4, 9], dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert a_ids.tolist() == [7]
    assert b_ids.tolist() == [9]
    assert counts.tolist() == [3]


def test_requires_numpy_arrays():
    with pytest.raises(TypeError, match="NumPy arrays"):
        overlap_counts([1, 2], np.array([1, 2], dtype=np.uint32))  # type: ignore[arg-type]


def test_rejects_unsupported_dtype():
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([1, 2], dtype=np.uint32)
    with pytest.raises(TypeError, match="uint32 or uint64"):
        overlap_counts(a, b)


def test_rejects_shape_mismatch():
    a = np.zeros((2, 3), dtype=np.uint32)
    b = np.zeros((6,), dtype=np.uint32)
    with pytest.raises(ValueError, match="same shape"):
        overlap_counts(a, b)


def test_many_unique_pairs_force_hash_growth():
    n = 50_000
    a = np.arange(1, n + 1, dtype=np.uint32)
    b = np.arange(n, 0, -1, dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    assert np.array_equal(a_ids, a)
    assert np.array_equal(b_ids, b)
    assert np.all(counts == 1)


def test_large_random_input_matches_reference():
    rng = np.random.default_rng(123)
    a = rng.integers(0, 10_000, size=100_000, dtype=np.uint32)
    b = rng.integers(0, 10_000, size=100_000, dtype=np.uint32)
    assert_matches_reference(a, b)


def test_include_background_counts_every_position():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 50, size=(64, 32), dtype=np.uint32)
    b = rng.integers(0, 70, size=(64, 32), dtype=np.uint64)
    _, _, counts = overlap_counts(a, b, include_background=True)
    assert int(counts.sum()) == a.size


def test_foreground_only_count_sum_matches_joint_foreground_pixels():
    rng = np.random.default_rng(8)
    a = rng.integers(0, 20, size=(32, 48), dtype=np.uint32)
    b = rng.integers(0, 25, size=(32, 48), dtype=np.uint32)
    _, _, counts = overlap_counts(a, b)
    expected = np.count_nonzero((a != 0) & (b != 0))
    assert int(counts.sum()) == int(expected)
