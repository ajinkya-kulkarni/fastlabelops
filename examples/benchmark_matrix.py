"""Benchmark fastlabelops across representative label-mask workloads."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from _benchmark_utils import make_mask
from numpy.typing import DTypeLike

from fastlabelops import (
    label_counts,
    overlap_counts,
    regionprops,
    relabel_sequential,
    remove_small_objects,
)


def _noop() -> None:
    pass


@dataclass(frozen=True)
class Benchmark:
    name: str
    run: Callable[[], object]
    prepare: Callable[[], None] = _noop


def _repeated_labels(
    shape: tuple[int, int],
    n_labels: int,
    *,
    dtype: DTypeLike,
    sparse: bool = False,
    shuffled: bool = False,
) -> np.ndarray:
    size = math.prod(shape)
    ids = np.arange(1, n_labels + 1, dtype=np.uint64)
    if sparse:
        ids *= np.uint64(1_000_003)

    run_length = (size + n_labels - 1) // n_labels
    flat = np.repeat(ids, run_length)[:size].astype(dtype, copy=False)
    if shuffled:
        rng = np.random.default_rng(42)
        rng.shuffle(flat)
    return np.ascontiguousarray(flat.reshape(shape))


def _array_benchmark(
    name: str,
    labels: np.ndarray,
    operation: Callable[[np.ndarray], object],
) -> Benchmark:
    return Benchmark(name, lambda: operation(labels))


def _pair_benchmark(
    name: str,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    operation: Callable[[np.ndarray, np.ndarray], object],
) -> Benchmark:
    return Benchmark(name, lambda: operation(labels_a, labels_b))


def _in_place_benchmark(
    name: str,
    labels: np.ndarray,
    operation: Callable[[np.ndarray], object],
) -> Benchmark:
    working = np.empty_like(labels, order="C")

    def prepare() -> None:
        np.copyto(working, labels)

    return Benchmark(name, lambda: operation(working), prepare)


def _build_benchmarks() -> list[Benchmark]:
    blocky = _repeated_labels((2048, 2048), 32_768, dtype=np.uint32)
    fragmented = blocky.copy()
    np.random.default_rng(42).shuffle(fragmented.reshape(-1))
    fragmented_sparse = _repeated_labels(
        (2048, 2048),
        32_768,
        dtype=np.uint64,
        sparse=True,
        shuffled=True,
    )
    high_cardinality = _repeated_labels((1024, 1024), 50_000, dtype=np.uint32)
    segmentation, _ = make_mask(2048)
    background = np.zeros((2048, 2048), dtype=np.uint32)
    strided = segmentation[:, ::2]

    masks = [
        ("blocky-compact", blocky),
        ("fragmented-compact", fragmented),
        ("fragmented-sparse64", fragmented_sparse),
        ("high-cardinality", high_cardinality),
        ("segmentation", segmentation),
        ("all-background", background),
        ("strided-view", strided),
    ]

    benchmarks: list[Benchmark] = []
    for case_name, labels in masks:
        benchmarks.extend(
            [
                _array_benchmark(
                    f"label_counts/{case_name}",
                    labels,
                    lambda value: label_counts(value),
                ),
                _array_benchmark(
                    f"relabel_sequential/out/{case_name}",
                    labels,
                    lambda value: relabel_sequential(value),
                ),
                _array_benchmark(
                    f"remove_small_objects/out/{case_name}",
                    labels,
                    lambda value: remove_small_objects(value, max_size=64),
                ),
                _pair_benchmark(
                    f"overlap_counts/{case_name}",
                    labels,
                    labels,
                    lambda a, b: overlap_counts(a, b, include_background=True),
                ),
                _array_benchmark(
                    f"regionprops/{case_name}",
                    labels,
                    lambda value: regionprops(value),
                ),
            ]
        )

    for case_name, labels in [
        ("blocky-compact", blocky),
        ("fragmented-compact", fragmented),
    ]:
        benchmarks.extend(
            [
                _in_place_benchmark(
                    f"relabel_sequential/in/{case_name}",
                    labels,
                    lambda value: relabel_sequential(value, in_place=True),
                ),
                _in_place_benchmark(
                    f"remove_small_objects/in/{case_name}",
                    labels,
                    lambda value: remove_small_objects(value, max_size=64, in_place=True),
                ),
            ]
        )

    mixed_removal, _ = make_mask(2048, side=lambda label: 4 if label % 2 else 12)
    benchmarks.extend(
        [
            _array_benchmark(
                "remove_small_objects/out/mixed-removal",
                mixed_removal,
                lambda value: remove_small_objects(value, max_size=64),
            ),
            _in_place_benchmark(
                "remove_small_objects/in/mixed-removal",
                mixed_removal,
                lambda value: remove_small_objects(value, max_size=64, in_place=True),
            ),
        ]
    )
    return benchmarks


def _measure(benchmark: Benchmark, *, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        benchmark.prepare()
        result = benchmark.run()
        del result

    gc.collect()
    times: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            benchmark.prepare()
            start = time.perf_counter_ns()
            result = benchmark.run()
            elapsed = time.perf_counter_ns() - start
            del result
            times.append(elapsed / 1e6)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(times)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Run benchmark names containing this text; may be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    benchmarks = _build_benchmarks()
    if args.filter:
        benchmarks = [
            benchmark
            for benchmark in benchmarks
            if any(text in benchmark.name for text in args.filter)
        ]
    if not benchmarks:
        raise SystemExit("no benchmarks matched the requested filters")

    timings = {
        benchmark.name: _measure(benchmark, repeats=args.repeats, warmups=args.warmups)
        for benchmark in benchmarks
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, elapsed in timings.items():
        grouped[name.split("/", 1)[0]].append(elapsed)
    operation_scores = {
        operation: statistics.geometric_mean(values)
        for operation, values in sorted(grouped.items())
    }
    overall_score = statistics.geometric_mean(operation_scores.values())

    if args.json:
        print(
            json.dumps(
                {
                    "metadata": {
                        "platform": platform.platform(),
                        "python": sys.version.split()[0],
                        "numpy": np.__version__,
                        "repeats": args.repeats,
                        "warmups": args.warmups,
                    },
                    "benchmarks_ms": timings,
                    "operation_scores_ms": operation_scores,
                    "overall_score_ms": overall_score,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("fastlabelops benchmark matrix")
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print(f"Median of {args.repeats} runs after {args.warmups} warmups\n")
    width = max(len(name) for name in timings)
    for name, elapsed in timings.items():
        print(f"{name:{width}s}  {elapsed:9.3f} ms")
    print("\nEqual-weight operation scores")
    for operation, elapsed in operation_scores.items():
        print(f"{operation:24s}  {elapsed:9.3f} ms")
    print(f"{'overall':24s}  {overall_score:9.3f} ms")


if __name__ == "__main__":
    main()
