import gc
import time

import numpy as np
from skimage.morphology import remove_small_objects as sk_remove_small_objects

from fastlabelops import remove_small_objects

MAX_SIZE = 64


def make_mask(size: int, sparse: bool = False) -> tuple[np.ndarray, int]:
    out = np.zeros((size, size), dtype=np.uint32)
    label = 1
    for y in range(4, size - 12, 20):
        for x in range(4, size - 12, 20):
            side = 4 if label % 2 else 12
            out[y : y + side, x : x + side] = label * (37 if sparse else 1)
            label += 1
    return out, label - 1


def check(labels: np.ndarray) -> None:
    fast = remove_small_objects(labels, max_size=MAX_SIZE)
    ref = sk_remove_small_objects(labels, max_size=MAX_SIZE)
    np.testing.assert_array_equal(fast, ref)


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
    print(f"max_size: {MAX_SIZE}")
    print("Correctness is checked against skimage.morphology.remove_small_objects before timing.\n")
    print(f"{'Mask':<24} {'Objects':>9} {'fastlabelops':>16} {'skimage':>12} {'Speedup':>10}")
    print("-" * 78)
    for size, sparse in cases:
        labels, n = make_mask(size, sparse)
        check(labels)
        fast = best(lambda labels=labels: remove_small_objects(labels, max_size=MAX_SIZE))
        slow = best(lambda labels=labels: sk_remove_small_objects(labels, max_size=MAX_SIZE))
        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")
        print(
            f"{name:<24} {n:>9,d} {fast * 1e3:>13.2f} ms "
            f"{slow * 1e3:>9.2f} ms {slow / fast:>8.2f}x"
        )


if __name__ == "__main__":
    main()
