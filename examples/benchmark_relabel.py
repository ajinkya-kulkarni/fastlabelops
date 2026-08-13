"""Benchmark fastlabelops relabeling vs skimage and fastremap on large 2D instance masks."""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    import fastremap
except ImportError:
    fastremap = None
from numpy.random import default_rng
from skimage.segmentation import relabel_sequential as sk_relabel

from fastlabelops import relabel_sequential as fr_relabel

rng = default_rng(42)


def make_mask(shape, n_instances, max_id, dtype=np.uint32):
    """Create a 2D mask with roughly n_instances non-zero connected blobs."""
    h, w = shape
    mask = np.zeros(shape, dtype=dtype)
    ids = rng.integers(1, max_id + 1, size=n_instances, dtype=dtype)
    ys = rng.integers(0, h, size=n_instances)
    xs = rng.integers(0, w, size=n_instances)
    radii = rng.integers(5, 25, size=n_instances)
    for y, x, r, inst_id in zip(ys, xs, radii, ids, strict=True):
        y0, y1 = max(0, y - r), min(h, y + r)
        x0, x1 = max(0, x - r), min(w, x + r)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        region = mask[y0:y1, x0:x1]
        dist = (yy - y) ** 2 + (xx - x) ** 2
        region[(dist <= r**2) & (region == 0)] = inst_id
    return mask


def timed(func, arr, repeats=5, warmup=1):
    """Time func(arr) over `repeats` runs, returning best and mean seconds."""
    for _ in range(warmup):
        func(np.array(arr, copy=True, order="C"))
    times = []
    for i in range(repeats):
        a = np.array(arr, copy=True, order="C")
        t0 = time.perf_counter()
        func(a)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"    run {i + 1}/{repeats}: {elapsed * 1e3:8.3f} ms")
    return min(times), sum(times) / len(times)


def check_valid(out, ref, name):
    """Verify that `out` is a valid sequential relabeling of `ref`."""
    assert out.shape == ref.shape, f"{name}: shape mismatch"
    assert (out == 0).sum() == (ref == 0).sum(), f"{name}: background changed"
    unique = np.unique(out)
    assert unique[0] == 0, f"{name}: background not preserved"
    expected = np.arange(unique.size, dtype=out.dtype)
    assert np.array_equal(unique, expected), f"{name}: not sequential {unique}"
    mask_nonzero = ref != 0
    mapping: dict[np.integer, np.integer] = {}
    for inp, outp in zip(ref[mask_nonzero].flat, out[mask_nonzero].flat, strict=True):
        if inp in mapping:
            assert mapping[inp] == outp, f"{name}: inconsistent mapping for {inp}"
        else:
            mapping[inp] = outp
    return unique.size - 1


def bench_case(shape, n_instances, max_id, dtype=np.uint32, repeats=5):
    print(f"\n{'=' * 60}")
    dtype_name = dtype.__name__
    print(
        f"Benchmark: shape={shape}, n_instances={n_instances}, max_id={max_id}, dtype={dtype_name}"
    )
    print("=" * 60)

    print("Generating mask...")
    mask = make_mask(shape, n_instances, max_id, dtype=dtype)
    actual_instances = int(np.unique(mask[mask != 0]).size)
    print(f"  actual non-zero ids: {actual_instances}")
    print(f"  mask memory: {mask.nbytes / 1e6:.2f} MB")

    print("\nVerifying correctness...")
    fr_out, fr_n = fr_relabel(np.array(mask, copy=True))
    sk_out, _, _ = sk_relabel(np.array(mask, copy=True), offset=1)
    n_fr = check_valid(fr_out, mask, "fastlabelops")
    n_sk = check_valid(sk_out, mask, "skimage")
    assert n_fr == n_sk == actual_instances
    if fastremap is not None:
        fm_out, _ = fastremap.renumber(np.array(mask, copy=True), start=1, preserve_zero=True)
        n_fm = check_valid(fm_out, mask, "fastremap")
        assert n_fm == actual_instances
    assert fr_n == actual_instances
    print("  correctness OK")

    print("\n  fastlabelops:")
    fr_best, fr_mean = timed(fr_relabel, mask, repeats=repeats)
    print("  skimage:")
    sk_best, sk_mean = timed(sk_relabel, mask, repeats=repeats)
    fm_result = None
    if fastremap is not None:
        print("  fastremap:")
        renumber = fastremap.renumber
        fm_result = timed(
            lambda a: renumber(a, start=1, preserve_zero=True),
            mask,
            repeats=repeats,
        )

    print("\nSummary:")
    print(f"  fastlabelops  best={fr_best * 1e3:8.3f} ms  mean={fr_mean * 1e3:8.3f} ms")
    sk_ratio = sk_best / fr_best
    print(
        f"  skimage       best={sk_best * 1e3:8.3f} ms  mean={sk_mean * 1e3:8.3f} ms  "
        f"({sk_ratio:5.2f}x slower)"
    )
    if fm_result is not None:
        fm_best, fm_mean = fm_result
        print(
            f"  fastremap     best={fm_best * 1e3:8.3f} ms  mean={fm_mean * 1e3:8.3f} ms  "
            f"({fm_best / fr_best:5.2f}x slower)"
        )


def main():
    print("Benchmarking sequential relabeling on large 2D masks")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"fastlabelops: {fr_relabel.__module__}")
    if fastremap is None:
        print("fastremap: not installed (skipped)")
    else:
        version = fastremap.__version__ if hasattr(fastremap, "__version__") else "unknown"
        print(f"fastremap: {version}")

    bench_case((2048, 2048), n_instances=5_000, max_id=5_000)
    bench_case((8192, 8192), n_instances=50_000, max_id=50_000, repeats=3)
    bench_case((8192, 8192), n_instances=1_000, max_id=2_000_000_000, dtype=np.uint32, repeats=3)


if __name__ == "__main__":
    main()
