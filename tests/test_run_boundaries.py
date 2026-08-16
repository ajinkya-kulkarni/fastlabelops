import numpy as np

from fastlabelops import label_counts, overlap_counts, relabel_sequential


def _reference_overlap(a: np.ndarray, b: np.ndarray):
    pairs: dict[tuple[int, int], int] = {}
    for av, bv in zip(a, b, strict=True):
        ai = int(av)
        bi = int(bv)
        if ai == 0 or bi == 0:
            continue
        key = (ai, bi)
        pairs[key] = pairs.get(key, 0) + 1
    return (
        np.asarray([key[0] for key in pairs], dtype=a.dtype),
        np.asarray([key[1] for key in pairs], dtype=b.dtype),
        np.asarray(list(pairs.values()), dtype=np.uint64),
    )


def _assert_overlap_matches_reference(a: np.ndarray, b: np.ndarray) -> None:
    actual = overlap_counts(a, b)
    expected = _reference_overlap(a, b)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_runs_cross_vector_boundaries():
    values = np.array([0, 91, 7, 5002, 13, 400, 2, 99], dtype=np.uint32)
    lengths = np.array([1, 7, 8, 9, 15, 16, 17, 31])
    labels = np.tile(np.repeat(values, lengths), 4)

    ids, counts = label_counts(labels)
    expected_ids, expected_counts = np.unique(labels[labels != 0], return_counts=True)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(counts, expected_counts.astype(np.uint64))

    relabeled, n_instances = relabel_sequential(labels)
    mapping: dict[int, int] = {}
    expected = np.zeros_like(labels)
    next_id = 1
    for i, value in enumerate(labels):
        label = int(value)
        if label == 0:
            continue
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        expected[i] = mapping[label]
    np.testing.assert_array_equal(relabeled, expected)
    assert n_instances == len(mapping)

    _assert_overlap_matches_reference(labels, np.roll(labels, 5))


def test_dense_transition_fallback_matches_reference():
    for dtype in (np.uint32, np.uint64):
        labels = np.arange(1, 2049, dtype=dtype)
        relabeled, n_instances = relabel_sequential(labels)
        np.testing.assert_array_equal(relabeled, labels)
        assert n_instances == labels.size

    labels = np.arange(1, 2049, dtype=np.uint32)
    _assert_overlap_matches_reference(labels, np.roll(labels, 1))


def test_sparse_foreground_overlap_matches_reference():
    a = np.zeros(4096, dtype=np.uint32)
    b = np.zeros_like(a)
    a[100:180] = 7
    a[2000:2050] = 11
    b[110:170] = 5
    b[2010:2040] = 13
    b[3000:3600:2] = np.arange(100, 400, dtype=np.uint32)

    _assert_overlap_matches_reference(a, b)
