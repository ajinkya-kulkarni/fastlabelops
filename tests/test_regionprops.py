import numpy as np
import pytest

from fastlabelops import regionprops


def test_basic() -> None:
    labels = np.array([[0, 7, 7, 0], [0, 7, 0, 20], [20, 20, 0, 0]], dtype=np.uint32)
    out = regionprops(labels)
    np.testing.assert_array_equal(out["label"], [7, 20])
    np.testing.assert_array_equal(out["area"], [3, 3])
    np.testing.assert_array_equal(out["bbox"], [[0, 1, 2, 3], [1, 0, 3, 4]])
    np.testing.assert_allclose(out["centroid"], [[1 / 3, 4 / 3], [5 / 3, 4 / 3]])
    np.testing.assert_array_equal(out["area_bbox"], [4, 8])


def test_sparse_uint64_ids() -> None:
    large_id = 2**63 + 5
    labels = np.array([[0, large_id], [91, 17]], dtype=np.uint64)
    out = regionprops(labels)
    np.testing.assert_array_equal(out["label"], [17, 91, large_id])


def test_many_unique_labels_force_hash_growth() -> None:
    labels = np.arange(1, 20_001, dtype=np.uint32).reshape(200, 100)
    out = regionprops(labels)
    np.testing.assert_array_equal(out["label"], np.arange(1, 20_001, dtype=np.uint32))
    np.testing.assert_array_equal(out["area"], np.ones(20_000, dtype=np.int64))


def test_non_contiguous_empty_and_validation() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.uint32)[:, ::-1]
    assert not labels.flags.c_contiguous
    np.testing.assert_array_equal(regionprops(labels)["area"], [1, 1])
    out = regionprops(np.zeros((3, 4), dtype=np.uint32))
    assert out["bbox"].shape == (0, 4)
    assert out["centroid"].shape == (0, 2)
    for dtype in (np.int32, np.float32, np.bool_):
        with pytest.raises(TypeError):
            regionprops(np.zeros((2, 2), dtype=dtype))
    with pytest.raises(ValueError):
        regionprops(np.zeros((2, 2, 2), dtype=np.uint32))
    with pytest.raises(TypeError):
        regionprops([[0, 1]])  # type: ignore[arg-type]
