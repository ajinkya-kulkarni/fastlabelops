import numpy as np

from fastlabelops import overlap_counts


def test_high_entropy_uint32_preserves_counts_and_first_occurrence_order():
    rng = np.random.default_rng(2026)
    a = rng.integers(1, 10_001, size=300_000, dtype=np.uint32)
    b = rng.integers(1, 10_001, size=300_000, dtype=np.uint32)

    keys = (a.astype(np.uint64) << np.uint64(32)) | b.astype(np.uint64)
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    order = np.argsort(first)
    unique = unique[order]

    expected_a = (unique >> np.uint64(32)).astype(np.uint32)
    expected_b = (unique & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    expected_counts = counts[order].astype(np.uint64, copy=False)

    actual_a, actual_b, actual_counts = overlap_counts(a, b)
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
    np.testing.assert_array_equal(actual_counts, expected_counts)
