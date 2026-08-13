import gc
import time

import numpy as np
from skimage.measure import regionprops as sk_regionprops

from fastlabelops import regionprops

PROPS = ("label", "area", "bbox", "centroid", "area_bbox")


def make_mask(size: int, sparse: bool = False) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.uint32)
    label = 1
    for y in range(6, size - 12, 18):
        for x in range(6, size - 12, 18):
            out[y : y + 12, x : x + 12] = label * (37 if sparse else 1)
            label += 1
    return out


def skimage_props(labels: np.ndarray) -> dict[str, np.ndarray]:
    r = sk_regionprops(labels)
    return {
        "label": np.asarray([x.label for x in r]),
        "area": np.asarray([x.area for x in r]),
        "bbox": np.asarray([x.bbox for x in r]),
        "centroid": np.asarray([x.centroid for x in r]),
        "area_bbox": np.asarray([x.area_bbox for x in r]),
    }


def check(labels: np.ndarray) -> None:
    fast, ref = regionprops(labels), skimage_props(labels)
    for key in ("label", "area", "bbox", "area_bbox"):
        np.testing.assert_array_equal(fast[key], ref[key])
    np.testing.assert_allclose(fast["centroid"], ref["centroid"], rtol=0, atol=1e-12)


def best(fn, repeats: int = 5) -> float:
    fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    cases = [(1024, False), (2048, False), (4096, False), (2048, True)]
    print("Properties:", ", ".join(PROPS))
    print("Correctness is checked against skimage.measure.regionprops before timing.\n")
    print(f"{'Mask':<24} {'Objects':>9} {'fastlabelops':>16} {'skimage':>12} {'Speedup':>10}")
    print("-" * 78)
    for size, sparse in cases:
        labels = make_mask(size, sparse)
        check(labels)
        fast = best(lambda labels=labels: regionprops(labels))
        slow = best(lambda labels=labels: skimage_props(labels))
        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")
        n = regionprops(labels)["label"].size
        print(
            f"{name:<24} {n:>9,d} {fast * 1e3:>13.2f} ms {slow * 1e3:>9.2f} ms {slow / fast:>8.2f}x"
        )


if __name__ == "__main__":
    main()
