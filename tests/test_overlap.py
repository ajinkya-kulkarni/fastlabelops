from __future__ import annotations

import numpy as np
import pytest

from fastlabelops import overlap_counts


def reference_counts(a, b, *, include_background=False):
    pairs: dict[tuple[int, int], int] = {}
    for av, bv in zip(a.flat, b.flat, strict=True):
        ai, bi = int(av), int(bv)
        if not include_background and (ai == 0 or bi == 0):
            continue
        pairs[(ai, bi)] = pairs.get((ai, bi), 0) + 1
    return (
        np.asarray([k[0] for k in pairs], dtype=a.dtype),
        np.asarray([k[1] for k in pairs], dtype=b.dtype),
        np.asarray(list(pairs.values()), dtype=np.uint64),
    )


def assert_matches(a, b, *, include_background=False):
    got = overlap_counts(a, b, include_background=include_background)
    want = reference_counts(a, b, include_background=include_background)
    for x, y in zip(got, want, strict=True):
        np.testing.assert_array_equal(x, y)


def test_basic():
    a = np.array([[0, 7, 7, 7], [20, 20, 0, 0]], dtype=np.uint32)
    b = np.array([[0, 4, 4, 9], [9, 9, 0, 3]], dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    np.testing.assert_array_equal(a_ids, [7, 7, 20])
    np.testing.assert_array_equal(b_ids, [4, 9, 9])
    np.testing.assert_array_equal(counts, [2, 1, 2])


def test_background_empty_mixed_and_uint64():
    a = np.array([0, 7, 7, 0, 20], dtype=np.uint32)
    b = np.array([0, 4, 0, 3, 0], dtype=np.uint64)
    assert_matches(a, b, include_background=True)
    assert_matches(np.empty((0, 4), np.uint32), np.empty((0, 4), np.uint64))
    high = 2**52 + 456
    assert_matches(
        np.array([2**48 + 123, 2**48 + 123, 9], dtype=np.uint64),
        np.array([high, high, high], dtype=np.uint64),
    )


def test_nd_noncontiguous_and_readonly():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 12, size=(4, 5, 6), dtype=np.uint32)
    b = rng.integers(0, 15, size=(4, 5, 6), dtype=np.uint32)
    assert_matches(a, b)
    a = a[:, :, ::2]
    b = b[:, :, ::2]
    assert_matches(a, b)
    a.flags.writeable = False
    b.flags.writeable = False
    assert_matches(a, b)


def test_validation():
    with pytest.raises(TypeError):
        overlap_counts([1, 2], np.array([1, 2], dtype=np.uint32))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        overlap_counts(np.array([1], np.int32), np.array([1], np.uint32))
    with pytest.raises(ValueError):
        overlap_counts(np.zeros((2, 3), np.uint32), np.zeros((6,), np.uint32))


def test_many_unique_pairs_and_random():
    n = 50_000
    a = np.arange(1, n + 1, dtype=np.uint32)
    b = np.arange(n, 0, -1, dtype=np.uint32)
    a_ids, b_ids, counts = overlap_counts(a, b)
    np.testing.assert_array_equal(a_ids, a)
    np.testing.assert_array_equal(b_ids, b)
    assert np.all(counts == 1)

    rng = np.random.default_rng(123)
    a = rng.integers(0, 10_000, size=100_000, dtype=np.uint32)
    b = rng.integers(0, 10_000, size=100_000, dtype=np.uint32)
    assert_matches(a, b)
