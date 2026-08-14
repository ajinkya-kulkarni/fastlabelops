import numpy as np
import pytest

from fastlabelops import label_counts


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_basic_sorted_and_excludes_background(dtype):
    labels = np.array([0, 9, 5, 5, 2, 9, 5, 0], dtype=dtype)
    ids, counts = label_counts(labels)

    np.testing.assert_array_equal(ids, [2, 5, 9])
    np.testing.assert_array_equal(counts, [1, 3, 2])
    assert ids.dtype == np.dtype(dtype)
    assert counts.dtype == np.uint64


def test_include_background():
    labels = np.array([[0, 5, 5], [9, 0, 5]], dtype=np.uint32)
    ids, counts = label_counts(labels, include_background=True)
    np.testing.assert_array_equal(ids, [0, 5, 9])
    np.testing.assert_array_equal(counts, [2, 3, 1])


def test_include_background_only_when_observed():
    labels = np.array([5, 5, 9], dtype=np.uint32)
    ids, counts = label_counts(labels, include_background=True)
    np.testing.assert_array_equal(ids, [5, 9])
    np.testing.assert_array_equal(counts, [2, 1])


def test_empty_array():
    labels = np.array([], dtype=np.uint32)
    ids, counts = label_counts(labels, include_background=True)
    assert ids.size == 0
    assert counts.size == 0


def test_all_background():
    labels = np.zeros((4, 5), dtype=np.uint64)
    ids, counts = label_counts(labels)
    assert ids.size == 0
    assert counts.size == 0

    ids, counts = label_counts(labels, include_background=True)
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_array_equal(counts, [20])


def test_arbitrary_dimensionality():
    labels = np.zeros((2, 3, 4), dtype=np.uint32)
    labels[0, 0, 0] = 17
    labels[0, 0, 1] = 17
    labels[1, 2, 3] = 91

    ids, counts = label_counts(labels)
    np.testing.assert_array_equal(ids, [17, 91])
    np.testing.assert_array_equal(counts, [2, 1])


def test_sparse_uint64_ids():
    labels = np.array([0, 2**63, 7, 2**63, 2**64 - 1], dtype=np.uint64)
    ids, counts = label_counts(labels)
    np.testing.assert_array_equal(ids, np.array([7, 2**63, 2**64 - 1], dtype=np.uint64))
    np.testing.assert_array_equal(counts, [1, 2, 1])


def test_non_contiguous_input_allowed():
    labels = np.array([[0, 4, 0, 7], [7, 4, 9, 9]], dtype=np.uint32)[:, ::2]
    assert not labels.flags.c_contiguous
    ids, counts = label_counts(labels)
    np.testing.assert_array_equal(ids, [7, 9])
    np.testing.assert_array_equal(counts, [1, 1])


def test_bad_dtype():
    with pytest.raises(TypeError, match="uint32 or uint64"):
        label_counts(np.array([0, 1], dtype=np.int32))


def test_requires_numpy_array():
    with pytest.raises(TypeError, match="NumPy array"):
        label_counts([0, 1, 1])


@pytest.mark.parametrize("include_background", [False, True])
@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_randomized_matches_numpy(dtype, include_background):
    rng = np.random.default_rng(1234)
    for _ in range(20):
        labels = rng.integers(0, 5000, size=(37, 53), dtype=dtype)
        labels[labels % 7 == 0] = 0

        expected_ids, expected_counts = np.unique(labels, return_counts=True)
        if not include_background:
            keep = expected_ids != 0
            expected_ids = expected_ids[keep]
            expected_counts = expected_counts[keep]

        ids, counts = label_counts(labels, include_background=include_background)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(counts, expected_counts.astype(np.uint64))
