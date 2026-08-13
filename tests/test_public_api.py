import numpy as np
import fastlabelops


def test_public_api_and_integration() -> None:
    labels = np.array([[0, 10, 10], [20, 20, 0]], dtype=np.uint32)
    relabeled, n = fastlabelops.relabel_sequential(labels)
    assert n == 2
    np.testing.assert_array_equal(fastlabelops.regionprops(relabeled)["area"], [2, 2])
    a_ids, b_ids, counts = fastlabelops.overlap_counts(labels, relabeled)
    np.testing.assert_array_equal(a_ids, [10, 20])
    np.testing.assert_array_equal(b_ids, [1, 2])
    np.testing.assert_array_equal(counts, [2, 2])
