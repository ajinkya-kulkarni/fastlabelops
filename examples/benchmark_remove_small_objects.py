from functools import partial

import numpy as np
from _benchmark_utils import best, make_mask
from skimage.morphology import remove_small_objects as sk_remove_small_objects

from fastlabelops import remove_small_objects

MAX_SIZE = 64


def check(labels: np.ndarray) -> None:
    fast = remove_small_objects(labels, max_size=MAX_SIZE)
    ref = sk_remove_small_objects(labels, max_size=MAX_SIZE)
    np.testing.assert_array_equal(fast, ref)


def main() -> None:
    cases = [(1024, False), (2048, False), (4096, False), (2048, True)]
    print(f"max_size: {MAX_SIZE}")
    print("Correctness is checked against skimage.morphology.remove_small_objects before timing.\n")
    print(f"{'Mask':<24} {'Objects':>9} {'fastlabelops':>16} {'skimage':>12} {'Speedup':>10}")
    print("-" * 78)
    for size, sparse in cases:
        labels, n = make_mask(
            size,
            sparse=sparse,
            start=4,
            step=20,
            side=lambda label: 4 if label % 2 else 12,
        )
        check(labels)
        fast = best(partial(remove_small_objects, labels, max_size=MAX_SIZE))
        slow = best(partial(sk_remove_small_objects, labels, max_size=MAX_SIZE))
        name = f"{size}x{size}" + (" sparse IDs" if sparse else "")
        print(
            f"{name:<24} {n:>9,d} {fast * 1e3:>13.2f} ms {slow * 1e3:>9.2f} ms {slow / fast:>8.2f}x"
        )


if __name__ == "__main__":
    main()
