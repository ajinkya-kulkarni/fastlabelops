import numpy as np
import pytest

from fastlabelops import relabel_sequential


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_basic(dtype):
    x = np.array([[0, 17, 17], [91, 0, 5002]], dtype=dtype)
    y, n = relabel_sequential(x)
    np.testing.assert_array_equal(y, [[0, 1, 1], [2, 0, 3]])
    assert n == 3
    np.testing.assert_array_equal(x, [[0, 17, 17], [91, 0, 5002]])


def test_first_occurrence_order():
    x = np.array([0, 999, 2, 999, 50], dtype=np.uint64)
    y, n = relabel_sequential(x)
    np.testing.assert_array_equal(y, [0, 1, 2, 1, 3])
    assert n == 3


def test_offset_and_in_place():
    x = np.array([0, 9, 9, 20], dtype=np.uint32)
    y, n = relabel_sequential(x, offset=100)
    np.testing.assert_array_equal(y, [0, 101, 101, 102])
    assert n == 2
    z = np.array([0, 9, 20], dtype=np.uint32)
    out, n = relabel_sequential(z, in_place=True)
    assert out is z
    np.testing.assert_array_equal(z, [0, 1, 2])
    assert n == 2


def test_non_contiguous_and_bad_dtype():
    x = np.arange(16, dtype=np.uint32).reshape(4, 4)[:, ::2]
    with pytest.raises(ValueError):
        relabel_sequential(x, in_place=True)
    y, _ = relabel_sequential(x)
    assert y.flags.c_contiguous
    with pytest.raises(TypeError):
        relabel_sequential(np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_offset_capacity(dtype):
    max_label = int(np.iinfo(dtype).max)
    empty = np.zeros(4, dtype=dtype)
    y, n = relabel_sequential(empty, offset=max_label)
    np.testing.assert_array_equal(y, empty)
    assert n == 0
    x = np.array([0, 50, 10, 50, 999], dtype=dtype)
    y, n = relabel_sequential(x, offset=max_label - 3)
    np.testing.assert_array_equal(y, np.array([0, max_label - 2, max_label - 1, max_label - 2, max_label], dtype=dtype))
    assert n == 3
    with pytest.raises(OverflowError):
        relabel_sequential(np.array([1], dtype=dtype), offset=max_label)


def _reference(x, offset=0):
    out = np.empty_like(x)
    mapping = {}
    next_id = offset
    for idx, raw in np.ndenumerate(x):
        old = int(raw)
        if old == 0:
            out[idx] = 0
            continue
        if old not in mapping:
            next_id += 1
            mapping[old] = next_id
        out[idx] = mapping[old]
    return out, len(mapping)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_randomized_matches_reference(dtype):
    rng = np.random.default_rng(1234)
    for _ in range(25):
        x = rng.integers(0, 5000, size=(37, 53), dtype=dtype)
        x[x % 7 == 0] = 0
        expected, expected_n = _reference(x, offset=17)
        got, n = relabel_sequential(x, offset=17)
        np.testing.assert_array_equal(got, expected)
        assert n == expected_n
