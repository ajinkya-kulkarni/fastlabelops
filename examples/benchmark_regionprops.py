from functools import partial

import numpy as np
from _benchmark_utils import best, make_mask
from skimage.measure import regionprops as sk_regionprops

from fastlabelops import regionprops

PROPS = ("label", "area", "bbox", "centroid", "area_bbox")


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


def main() -> None:
    cases = [(1024, False), (2048, False), (4096, False), (2048, True)]
    print("Properties:", ", ".join(PROPS))
    print("Correctness is checked against skimage.measure.regionprops before timing.\n")
    print(f"{'Mask':<24} {'Objects':>9} {'fastlabelops':>16} {'skimage':>12} {'Speedup':>10}")
    print("-" * 78)
    for size, sparse in cases:
        labels = make_mask(size, sparse=sparse)[0]
        check(labels)
        fast = best(partial(regionprops, labels))
        slow = best(partial(skimage_props, labels))
        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")
        n = regionprops(labels)["label"].size
        print(
            f"{name:<24} {n:>9,d} {fast * 1e3:>13.2f} ms {slow * 1e3:>9.2f} ms {slow / fast:>8.2f}x"
        )


if __name__ == "__main__":
    main()
