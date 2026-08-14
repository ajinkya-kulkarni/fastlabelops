import numpy as np
import pytest

from fastlabelops import remove_small_objects


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_basic(dtype):
    labels = np.array(
        [
            [0, 10, 10, 0],
            [7, 0, 10, 9],
            [7, 0, 0, 0],
        ],
        dtype=dtype,
    )

    out = remove_small_objects(labels, max_size=2)

    np.testing.assert_array_equal(out, [[0, 10, 10, 0], [0, 0, 10, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(labels, [[0, 10, 10, 0], [7, 0, 10, 9], [7, 0, 0, 0]])


def test_threshold_is_inclusive():
    labels = np.array([5, 5, 0, 7, 7, 7], dtype=np.uint32)
    out = remove_small_objects(labels, max_size=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 7, 7, 7])


def test_same_id_is_one_object_even_if_disconnected():
    labels = np.array([99, 0, 99, 0, 99], dtype=np.uint64)
    out = remove_small_objects(labels, max_size=2)
    np.testing.assert_array_equal(out, labels)


def test_arbitrary_dimensionality():
    labels = np.zeros((2, 3, 4), dtype=np.uint32)
    labels[0, 0, 0] = 10
    labels[0, 0, 1] = 10
    labels[1, 2, 3] = 4_000_000_000

    out = remove_small_objects(labels, max_size=1)

    assert out.shape == labels.shape
    assert out[0, 0, 0] == 10
    assert out[0, 0, 1] == 10
    assert out[1, 2, 3] == 0


def test_sparse_uint64_ids():
    labels = np.array([0, 2**63, 0, 2**63, 2**64 - 1], dtype=np.uint64)
    out = remove_small_objects(labels, max_size=1)
    np.testing.assert_array_equal(out, [0, 2**63, 0, 2**63, 0])


def test_zero_threshold_is_noop():
    labels = np.array([0, 5, 10], dtype=np.uint32)
    out = remove_small_objects(labels, max_size=0)
    np.testing.assert_array_equal(out, labels)


def test_threshold_at_least_array_size_removes_all_foreground():
    labels = np.array([0, 5, 5, 10], dtype=np.uint32)
    out = remove_small_objects(labels, max_size=labels.size)
    np.testing.assert_array_equal(out, np.zeros_like(labels))


def test_in_place_identity():
    labels = np.array([0, 5, 5, 10], dtype=np.uint32)
    out = remove_small_objects(labels, max_size=1, in_place=True)
    assert out is labels
    np.testing.assert_array_equal(labels, [0, 5, 5, 0])


def test_non_contiguous_in_place_rejected():
    labels = np.arange(16, dtype=np.uint32).reshape(4, 4)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        remove_small_objects(labels, max_size=1, in_place=True)


def test_readonly_in_place_rejected():
    labels = np.array([0, 5], dtype=np.uint32)
    labels.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        remove_small_objects(labels, max_size=1, in_place=True)


def test_non_contiguous_copy_allowed():
    labels = np.array([[0, 4, 0, 7], [7, 4, 9, 9]], dtype=np.uint32)[:, ::2]
    out = remove_small_objects(labels, max_size=1)
    assert out.flags.c_contiguous


def test_bad_dtype():
    with pytest.raises(TypeError, match="uint32 or uint64"):
        remove_small_objects(np.array([0, 1], dtype=np.int32), max_size=1)


@pytest.mark.parametrize("max_size", [1.5, np.float64(2.5), True, np.bool_(False)])
def test_max_size_requires_integer(max_size):
    labels = np.array([0, 7], dtype=np.uint32)
    with pytest.raises(TypeError, match="max_size must be an integer"):
        remove_small_objects(labels, max_size=max_size)


def test_negative_max_size_rejected():
    labels = np.array([0, 7], dtype=np.uint32)
    with pytest.raises(ValueError, match="non-negative"):
        remove_small_objects(labels, max_size=-1)


def test_max_size_over_uint64_rejected():
    labels = np.array([0, 7], dtype=np.uint32)
    with pytest.raises(OverflowError, match="uint64"):
        remove_small_objects(labels, max_size=2**64)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_randomized_matches_reference(dtype):
    rng = np.random.default_rng(1234)
    for _ in range(25):
        labels = rng.integers(0, 1000, size=(37, 53), dtype=dtype)
        labels[labels % 5 == 0] = 0
        max_size = 3

        values, counts = np.unique(labels[labels != 0], return_counts=True)
        remove = set(values[counts <= max_size].tolist())
        expected = labels.copy()
        if remove:
            expected[np.isin(expected, list(remove))] = 0

        out = remove_small_objects(labels, max_size=max_size)
        np.testing.assert_array_equal(out, expected)
