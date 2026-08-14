"""Shared helpers for the fastlabelops benchmark scripts."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import numpy as np


def make_mask(
    size: int,
    *,
    sparse: bool = False,
    start: int = 6,
    step: int = 18,
    end_offset: int = 12,
    side: int | Callable[[int], int] = 12,
    sparse_multiplier: int = 37,
) -> tuple[np.ndarray, int]:
    """Create a square mask with a grid of labeled squares.

    Returns the mask and the number of objects placed.
    """
    out = np.zeros((size, size), dtype=np.uint32)
    label = 1
    end = size - end_offset
    for y in range(start, end, step):
        for x in range(start, end, step):
            current_side = side(label) if callable(side) else side
            value = label * (sparse_multiplier if sparse else 1)
            out[y : y + current_side, x : x + current_side] = value
            label += 1
    return out, label - 1


def best(fn: Callable[[], object], repeats: int = 5) -> float:
    """Return the best timing of ``repeats`` runs, with one warmup run."""
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


def check_equal(
    expected: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
) -> None:
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def wrap[T](fn: Callable[[np.ndarray], T], array: np.ndarray) -> Callable[[], T]:
    """Wrap a single-argument function so it can be timed without arguments."""
    return lambda: fn(array)
