import numpy as np

import fastlabelops


def test_public_api_and_integration() -> None:
    assert set(fastlabelops.__all__) == {
        "label_counts",
        "relabel_sequential",
        "remove_small_objects",
        "overlap_counts",
        "regionprops",
    }

    labels = np.array([[0, 10, 10], [20, 20, 0]], dtype=np.uint32)

    ids, counts = fastlabelops.label_counts(labels)
    np.testing.assert_array_equal(ids, [10, 20])
    np.testing.assert_array_equal(counts, [2, 2])

    relabeled, n = fastlabelops.relabel_sequential(labels)
    assert n == 2

    props = fastlabelops.regionprops(relabeled)
    np.testing.assert_array_equal(props["label"], [1, 2])
    np.testing.assert_array_equal(props["area"], [2, 2])

    a_ids, b_ids, overlap = fastlabelops.overlap_counts(labels, relabeled)
    np.testing.assert_array_equal(a_ids, [10, 20])
    np.testing.assert_array_equal(b_ids, [1, 2])
    np.testing.assert_array_equal(overlap, [2, 2])

    filtered = fastlabelops.remove_small_objects(
        np.array([0, 10, 10, 20], dtype=np.uint32),
        max_size=1,
    )
    np.testing.assert_array_equal(filtered, [0, 10, 10, 0])
