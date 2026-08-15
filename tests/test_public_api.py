import inspect
from collections.abc import Callable

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


def test_options_are_keyword_only() -> None:
    functions_and_options: dict[Callable[..., object], tuple[str, ...]] = {
        fastlabelops.label_counts: ("include_background",),
        fastlabelops.relabel_sequential: ("offset", "in_place"),
        fastlabelops.remove_small_objects: ("max_size", "in_place"),
        fastlabelops.overlap_counts: ("include_background",),
    }
    for function, option_names in functions_and_options.items():
        parameters = inspect.signature(function).parameters
        for name in option_names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_compact_and_sparse_labels_share_all_single_label_operations() -> None:
    high = np.uint32(4_000_000_000)
    labels = np.array([[0, 1, high], [high, 2, 1]], dtype=np.uint32)

    ids, counts = fastlabelops.label_counts(labels)
    np.testing.assert_array_equal(ids, [1, 2, high])
    np.testing.assert_array_equal(counts, [2, 1, 2])

    relabeled, n = fastlabelops.relabel_sequential(labels)
    np.testing.assert_array_equal(relabeled, [[0, 1, 2], [2, 3, 1]])
    assert n == 3

    filtered = fastlabelops.remove_small_objects(labels, max_size=1)
    np.testing.assert_array_equal(filtered, [[0, 1, high], [high, 0, 1]])

    props = fastlabelops.regionprops(labels)
    np.testing.assert_array_equal(props["label"], [1, 2, high])
    np.testing.assert_array_equal(props["area"], [2, 1, 2])


def test_uint32_direct_buckets_survive_growth_and_sparse_fallback() -> None:
    labels = np.arange(1, 50_001, dtype=np.uint32).reshape(200, 250)
    labels[-1, -1] = 4_000_000_000
    expected_ids = np.concatenate(
        [np.arange(1, 50_000, dtype=np.uint32), np.array([4_000_000_000], dtype=np.uint32)]
    )

    ids, counts = fastlabelops.label_counts(labels)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(counts, np.ones(50_000, dtype=np.uint64))

    relabeled, n = fastlabelops.relabel_sequential(labels)
    np.testing.assert_array_equal(
        relabeled,
        np.arange(1, 50_001, dtype=np.uint32).reshape(labels.shape),
    )
    assert n == 50_000

    filtered = fastlabelops.remove_small_objects(labels, max_size=1)
    np.testing.assert_array_equal(filtered, np.zeros_like(labels))

    props = fastlabelops.regionprops(labels)
    np.testing.assert_array_equal(props["label"], expected_ids)
    np.testing.assert_array_equal(props["area"], np.ones(50_000, dtype=np.int64))
