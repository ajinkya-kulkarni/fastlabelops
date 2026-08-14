import gc
import time
from collections.abc import Callable

import numpy as np

from fastlabelops import label_counts

try:
    import fastremap
except ImportError:
    fastremap = None

BINCOUNT_MAX_LABEL = 10_000_000


def make_mask(size: int, sparse: bool = False) -> tuple[np.ndarray, int]:
    out = np.zeros((size, size), dtype=np.uint32)
    label = 1
    for y in range(6, size - 12, 18):
        for x in range(6, size - 12, 18):
            out[y : y + 12, x : x + 12] = label * (37_003 if sparse else 1)
            label += 1
    return out, label - 1


def numpy_unique_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids != 0
    return ids[keep], counts[keep].astype(np.uint64, copy=False)


def numpy_bincount_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense = np.bincount(labels.ravel())
    ids = np.flatnonzero(dense[1:]) + 1
    return ids.astype(labels.dtype, copy=False), dense[ids].astype(np.uint64, copy=False)


def fastremap_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fastremap is None:
        raise RuntimeError("fastremap is not installed")
    ids, counts = fastremap.unique(labels, return_counts=True)
    keep = ids != 0
    return ids[keep], counts[keep].astype(np.uint64, copy=False)


def check_equal(
    expected: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
) -> None:
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def best(fn: Callable[[], object], repeats: int = 5) -> float:
    fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def format_time(seconds: float | None) -> str:
    return "skipped" if seconds is None else f"{seconds * 1e3:.2f} ms"


def main() -> None:
    cases = [(1024, False), (2048, False), (4096, False), (2048, True)]
    print("Counts exclude background and are returned as sorted (label, count) arrays.")
    print("Correctness is checked against NumPy before timing.")
    print("fastremap:", "available" if fastremap is not None else "not installed; skipped")
    print()
    print(
        f"{'Mask':<24} {'Objects':>9} {'fastlabelops':>16} {'np.unique':>12} "
        f"{'np.bincount':>14} {'fastremap':>12}"
    )
    print("-" * 96)

    for size, sparse in cases:
        labels, n = make_mask(size, sparse)
        expected = numpy_unique_counts(labels)
        check_equal(expected, label_counts(labels))

        fast = best(lambda labels=labels: label_counts(labels))
        unique = best(lambda labels=labels: numpy_unique_counts(labels))

        bincount = None
        if int(labels.max()) <= BINCOUNT_MAX_LABEL:
            check_equal(expected, numpy_bincount_counts(labels))
            bincount = best(lambda labels=labels: numpy_bincount_counts(labels))

        remap = None
        if fastremap is not None:
            check_equal(expected, fastremap_counts(labels))
            remap = best(lambda labels=labels: fastremap_counts(labels))

        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")
        print(
            f"{name:<24} {n:>9,d} {format_time(fast):>16} {format_time(unique):>12} "
            f"{format_time(bincount):>14} {format_time(remap):>12}"
        )


if __name__ == "__main__":
    main()
