"""Compare fastlabelops regionprops against Martin Weigert's fast-regionprops."""

from __future__ import annotations

import gc
import importlib.metadata
import statistics
import time

import numpy as np
from fast_regionprops import regionprops_table_fast  # type: ignore

from fastlabelops import regionprops

COMMON_PROPERTIES = ("label", "area", "bbox", "centroid")


def make_mask(size: int, sparse: bool = False) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.uint32)
    label = 1
    for y in range(6, size - 12, 18):
        for x in range(6, size - 12, 18):
            out[y : y + 12, x : x + 12] = label * (37 if sparse else 1)
            label += 1
    return out


def martin_common(labels: np.ndarray) -> dict[str, np.ndarray]:
    return regionprops_table_fast(labels, properties=COMMON_PROPERTIES)


def martin_equivalent(labels: np.ndarray) -> dict[str, np.ndarray]:
    out = martin_common(labels)
    out["area_bbox"] = (out["bbox-2"] - out["bbox-0"]) * (
        out["bbox-3"] - out["bbox-1"]
    )
    return out


def check(labels: np.ndarray) -> None:
    fast = regionprops(labels)
    martin = martin_equivalent(labels)

    np.testing.assert_array_equal(fast["label"], martin["label"])
    np.testing.assert_array_equal(fast["area"], martin["area"])
    np.testing.assert_array_equal(fast["bbox"][:, 0], martin["bbox-0"])
    np.testing.assert_array_equal(fast["bbox"][:, 1], martin["bbox-1"])
    np.testing.assert_array_equal(fast["bbox"][:, 2], martin["bbox-2"])
    np.testing.assert_array_equal(fast["bbox"][:, 3], martin["bbox-3"])
    np.testing.assert_allclose(fast["centroid"][:, 0], martin["centroid-0"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast["centroid"][:, 1], martin["centroid-1"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fast["area_bbox"], martin["area_bbox"])


def median_ms(fn, repeats: int) -> float:
    fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(times)


def main() -> None:
    print("fastlabelops vs fast-regionprops")
    print(f"NumPy: {np.__version__}")
    print(f"fast-regionprops: {importlib.metadata.version('fast-regionprops')}")
    print()
    print(
        f"{'Mask':<24} {'Objects':>9} {'fastlabelops(5)':>18} "
        f"{'Martin(4)':>14} {'Martin(4)+area_bbox':>22} {'vs M4':>9} {'vs M5':>9}"
    )
    print("-" * 112)

    cases = [
        (1024, False, 9),
        (2048, False, 7),
        (4096, False, 5),
        (2048, True, 7),
    ]
    for size, sparse, repeats in cases:
        labels = make_mask(size, sparse)
        check(labels)

        fast_ms = median_ms(lambda: regionprops(labels), repeats)
        martin4_ms = median_ms(lambda: martin_common(labels), repeats)
        martin5_ms = median_ms(lambda: martin_equivalent(labels), repeats)
        objects = regionprops(labels)["label"].size
        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")

        print(
            f"{name:<24} {objects:>9,d} {fast_ms:>15.3f} ms "
            f"{martin4_ms:>11.3f} ms {martin5_ms:>19.3f} ms "
            f"{martin4_ms / fast_ms:>7.2f}x {martin5_ms / fast_ms:>7.2f}x"
        )

    print()
    print("fastlabelops always computes label, area, bbox, centroid, and area_bbox.")
    print("Martin(4) requests only the four properties shared by fast-regionprops.")
    print("Martin(4)+area_bbox additionally derives area_bbox from Martin's bbox columns.")


if __name__ == "__main__":
    main()
