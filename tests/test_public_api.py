import inspect
from collections.abc import Callable

import numpy as np

import fastlabelops


def test_public_api_and_integration() -> None:
    assert set(fastlabelops.__all__) == {
        "LabelCounts",
        "OverlapCounts",
        "label_counts",
        "relabel_sequential",
        "remove_small_objects",
        "overlap_counts",
        "regionprops",
    }

    labels = np.array([[0, 10, 10], [20, 20, 0]], dtype=np.uint32)

    result = fastlabelops.label_counts(labels)
    assert isinstance(result, fastlabelops.LabelCounts)
    np.testing.assert_array_equal(result.ids, [10, 20])
    np.testing.assert_array_equal(result.counts, [2, 2])

    relabeled, n = fastlabelops.relabel_sequential(labels)
    assert n == 2

    props = fastlabelops.regionprops(relabeled)
    np.testing.assert_array_equal(props["label"], [1, 2])
    np.testing.assert_array_equal(props["area"], [2, 2])

    overlap = fastlabelops.overlap_counts(labels, relabeled)
    assert isinstance(overlap, fastlabelops.OverlapCounts)
    np.testing.assert_array_equal(overlap.a_ids, [10, 20])
    np.testing.assert_array_equal(overlap.b_ids, [1, 2])
    np.testing.assert_array_equal(overlap.counts, [2, 2])

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


def test_fragmented_compact_and_sparse_labels_across_growth() -> None:
    # This ID hashes into the compact range, covering the former long-probe regression.
    sparse = np.uint32(4_000_027_670)
    dense_prefix_size = 16_000
    compact_label_count = 30_000
    sparse_output_label = dense_prefix_size + 1
    repeats = 128
    flat = np.concatenate(
        [
            np.arange(1, dense_prefix_size + 1, dtype=np.uint32),
            np.array([sparse], dtype=np.uint32),
            np.arange(dense_prefix_size + 1, compact_label_count + 1, dtype=np.uint32),
            np.tile(np.array([sparse, 1], dtype=np.uint32), repeats),
        ]
    )
    labels = flat.reshape(1, -1)
    expected_ids = np.concatenate(
        [
            np.arange(1, compact_label_count + 1, dtype=np.uint32),
            np.array([sparse], dtype=np.uint32),
        ]
    )
    expected_counts = np.ones(compact_label_count + 1, dtype=np.uint64)
    expected_counts[0] = repeats + 1
    expected_counts[-1] = repeats + 1

    ids, counts = fastlabelops.label_counts(labels)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(counts, expected_counts)

    relabeled, n = fastlabelops.relabel_sequential(labels)
    expected_relabeled = np.concatenate(
        [
            np.arange(1, dense_prefix_size + 1, dtype=np.uint32),
            np.array([sparse_output_label], dtype=np.uint32),
            np.arange(sparse_output_label + 1, compact_label_count + 2, dtype=np.uint32),
            np.tile(np.array([sparse_output_label, 1], dtype=np.uint32), repeats),
        ]
    )
    np.testing.assert_array_equal(relabeled, expected_relabeled.reshape(labels.shape))
    assert n == compact_label_count + 1

    filtered = fastlabelops.remove_small_objects(labels, max_size=1)
    expected_filtered = np.zeros_like(flat)
    expected_filtered[0] = 1
    expected_filtered[dense_prefix_size] = sparse
    expected_filtered[-2 * repeats :] = np.tile(np.array([sparse, 1], dtype=np.uint32), repeats)
    np.testing.assert_array_equal(filtered, expected_filtered.reshape(labels.shape))

    props = fastlabelops.regionprops(labels)
    np.testing.assert_array_equal(props["label"], expected_ids)
    np.testing.assert_array_equal(props["area"], expected_counts)
