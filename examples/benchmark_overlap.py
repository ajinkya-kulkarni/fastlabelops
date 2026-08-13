"""Benchmark fastlabelops overlap counting against NumPy, scikit-image, and StarDist."""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable

import numpy as np
import skimage  # type: ignore
from skimage.metrics import contingency_table  # type: ignore

try:
    import stardist  # type: ignore
    from stardist.matching import _label_overlap, relabel_sequential as sd_relabel  # type: ignore
except ImportError:
    stardist = None  # type: ignore
    _label_overlap = None  # type: ignore
    sd_relabel = None  # type: ignore

from fastlabelops import overlap_counts, relabel_sequential as fl_relabel

rng = np.random.default_rng(42)


def make_segmentation_like_masks(shape, n_ids, max_id=None):
    """Create deterministic blocky label masks with repeated neighboring pairs."""
    h, w = shape
    block = 32
    gh = (h + block - 1) // block
    gw = (w + block - 1) // block

    if max_id is None:
        max_id = n_ids
    if max_id == n_ids:
        ids = np.arange(1, n_ids + 1, dtype=np.uint32)
        rng.shuffle(ids)
    else:
        sampled: set[int] = set()
        while len(sampled) < n_ids:
            needed = n_ids - len(sampled)
            values = rng.integers(1, max_id + 1, size=needed * 2, dtype=np.uint32)
            sampled.update(int(value) for value in values)
        ids = np.fromiter(list(sampled)[:n_ids], dtype=np.uint32, count=n_ids)

    coarse_a = rng.choice(ids, size=(gh, gw))
    coarse_a[rng.random((gh, gw)) < 0.15] = 0

    coarse_b = coarse_a.copy()
    changed = rng.random((gh, gw)) < 0.20
    coarse_b[changed] = rng.choice(ids, size=int(changed.sum()))
    coarse_b[rng.random((gh, gw)) < 0.15] = 0

    a = np.repeat(np.repeat(coarse_a, block, axis=0), block, axis=1)[:h, :w]
    b = np.repeat(np.repeat(coarse_b, block, axis=0), block, axis=1)[:h, :w]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def fastoverlap_full(a, b):
    return overlap_counts(a, b, include_background=True)


def numpy_packed_counts(a, b):
    """Count every uint32 pair by packing it into one uint64 key."""
    keys = (a.ravel().astype(np.uint64) << 32) | b.ravel().astype(np.uint64)
    return np.unique(keys, return_counts=True)


def skimage_counts(a, b):
    return contingency_table(a, b, normalize=False)


def stardist_raw(a, b):
    assert _label_overlap is not None
    return _label_overlap(a, b)


def stardist_prepare(a, b):
    assert _label_overlap is not None and sd_relabel is not None
    a_seq, _, _ = sd_relabel(a)
    b_seq, _, _ = sd_relabel(b)
    return _label_overlap(a_seq, b_seq)


def timed(func: Callable, a, b, *, repeats=5, warmup=1):
    for _ in range(warmup):
        func(a, b)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(a, b)
        times.append(time.perf_counter() - start)
        del result
    return statistics.median(times)


def triplet_dict(a_ids, b_ids, counts):
    return {
        (int(a_id), int(b_id)): int(count)
        for a_id, b_id, count in zip(a_ids, b_ids, counts, strict=True)
    }


def check_equivalent(a, b):
    fast = triplet_dict(*fastoverlap_full(a, b))

    keys, counts = numpy_packed_counts(a, b)
    numpy_result = {
        (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF))): int(count)
        for key, count in zip(keys, counts, strict=True)
    }
    assert fast == numpy_result

    sk = skimage_counts(a, b).tocoo()
    skimage_result = {
        (int(row), int(col)): int(count)
        for row, col, count in zip(sk.row, sk.col, sk.data, strict=True)
        if count
    }
    assert fast == skimage_result

    if stardist is not None:
        dense = stardist_raw(a, b)
        rows, cols = np.nonzero(dense)
        stardist_result = {
            (int(row), int(col)): int(dense[row, col])
            for row, col in zip(rows, cols, strict=True)
        }
        assert fast == stardist_result


def sequential_masks(a, b):
    a_seq, _ = fl_relabel(a)
    b_seq, _ = fl_relabel(b)
    return (
        np.ascontiguousarray(a_seq, dtype=np.uint32),
        np.ascontiguousarray(b_seq, dtype=np.uint32),
    )


def bench_case(shape, n_ids, *, repeats=5):
    original_a, original_b = make_segmentation_like_masks(shape, n_ids)
    a, b = sequential_masks(original_a, original_b)

    if stardist is not None:
        stardist_raw(a, b)
    check_equivalent(a, b)

    results = [
        ("fastlabelops", timed(fastoverlap_full, a, b, repeats=repeats)),
        ("NumPy unique", timed(numpy_packed_counts, a, b, repeats=repeats)),
        ("scikit-image", timed(skimage_counts, a, b, repeats=repeats)),
    ]
    if stardist is not None:
        results.extend(
            [
                ("StarDist raw", timed(stardist_raw, a, b, repeats=repeats)),
                (
                    "StarDist relabel+overlap",
                    timed(stardist_prepare, original_a, original_b, repeats=repeats),
                ),
            ]
        )

    fast_time = results[0][1]
    dense_bytes = (int(a.max()) + 1) * (int(b.max()) + 1) * np.dtype(np.uintp).itemsize
    observed_pairs = len(fastoverlap_full(a, b)[2])

    print(f"\n{shape[0]}x{shape[1]}, {n_ids:,}-ID pool, {observed_pairs:,} observed pairs")
    for name, elapsed in results:
        ratio = elapsed / fast_time
        suffix = "" if name == "fastlabelops" else f"  ({ratio:.2f}x vs fastlabelops)"
        print(f"  {name:25s} {elapsed * 1e3:9.2f} ms{suffix}")
    print(f"  dense matrix at max IDs    {dense_bytes / 1e6:9.1f} MB")


def sparse_id_stress(*, repeats=3):
    a, b = make_segmentation_like_masks(
        (4096, 4096),
        n_ids=1_000,
        max_id=2_000_000_000,
    )
    fast_time = timed(fastoverlap_full, a, b, repeats=repeats)
    numpy_time = timed(numpy_packed_counts, a, b, repeats=repeats)
    max_a = int(a.max())
    max_b = int(b.max())
    dense_bytes = (max_a + 1) * (max_b + 1) * np.dtype(np.uintp).itemsize

    print("\n4096x4096, 1K sparse IDs sampled up to 2B")
    print(f"  fastlabelops                {fast_time * 1e3:9.2f} ms")
    print(
        f"  NumPy unique               {numpy_time * 1e3:9.2f} ms  "
        f"({numpy_time / fast_time:.2f}x vs fastlabelops)"
    )
    print(f"  dense matrix at max IDs    {dense_bytes / 1e18:9.2f} EB")
    print("  scikit-image / StarDist    skipped: output shape depends on max(label)")


def main():
    print("Sparse overlap counting for integer instance masks")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-image: {skimage.__version__}")
    if stardist is None:
        print("StarDist: not installed (skipped)")
    else:
        print(f"StarDist: {stardist.__version__}")
    print("All normal-case methods count the same full contingency, including background.")

    bench_case((1024, 1024), n_ids=1_000)
    bench_case((2048, 2048), n_ids=3_000)
    bench_case((4096, 4096), n_ids=5_000, repeats=3)
    sparse_id_stress()


if __name__ == "__main__":
    main()
