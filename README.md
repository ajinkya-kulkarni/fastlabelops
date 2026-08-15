# fastlabelops

Fast CPU primitives for integer instance masks. Requires Python 3.12+.

`fastlabelops` combines five small operations that are commonly needed around instance
segmentation and labeled images:

- `label_counts` — count elements for each observed label
- `relabel_sequential` — compact arbitrary IDs to sequential labels
- `remove_small_objects` — remove labeled objects at or below a size threshold
- `overlap_counts` — count only observed label-pair overlaps between two masks
- `regionprops` — compute a small set of common per-label properties in 2D

The package is deliberately low-level: NumPy arrays in, NumPy arrays out, one compiled C++
extension, no scikit-image dependency at runtime.

```bash
python -m pip install .
```

```python
import numpy as np
from fastlabelops import (
    label_counts,
    overlap_counts,
    regionprops,
    relabel_sequential,
    remove_small_objects,
)

labels = np.array(
    [
        [0, 17, 17],
        [91, 0, 5002],
    ],
    dtype=np.uint32,
)

ids, counts = label_counts(labels)
relabeled, n = relabel_sequential(labels)
filtered = remove_small_objects(labels, max_size=1)
props = regionprops(labels)

a_ids, b_ids, overlap = overlap_counts(labels, relabeled)
```

Label `0` is background throughout. Inputs already contain instance IDs; none of these functions
perform connected-component labeling.

## `label_counts`

```python
ids, counts = label_counts(labels, include_background=False)
```

Counts elements for each observed label ID. IDs are returned in ascending order and counts are
`uint64`. Background label `0` is excluded by default; use `include_background=True` to include it
when it is present in the input.

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- accepts non-contiguous inputs
- memory scales with the number of observed labels, not `max(label)`
- shares the same sparse counting scan used internally by `remove_small_objects`

## `relabel_sequential`

```python
relabeled, n = relabel_sequential(labels, offset=0, in_place=False)
```

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- deterministic first-occurrence relabeling
- `in_place=True` mutates a writable C-contiguous array
- `offset=N` starts foreground labels at `N + 1`
- memory scales with the number of observed labels, not `max(label)`

## `remove_small_objects`

```python
filtered = remove_small_objects(labels, max_size=64, in_place=False)
```

Removes every nonzero label whose total size is less than or equal to `max_size` elements. Surviving
objects keep their original IDs. Labels are treated as authoritative instance IDs, so disconnected
elements carrying the same nonzero label count toward the same object's size.

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- `in_place=True` mutates a writable C-contiguous array
- `max_size=0` is a no-op
- memory scales with the number of observed labels, not `max(label)`

## `overlap_counts`

```python
a_ids, b_ids, counts = overlap_counts(labels_a, labels_b, include_background=False)
```

The three output arrays describe only label pairs that actually occur. By default, positions
where either input is background are ignored. Use `include_background=True` when foreground-vs-
background pairs are needed, for example to reconstruct full contingency/IoU denominators.

- same-shape inputs of arbitrary dimensionality
- supports `uint32` and `uint64`, including mixed dtypes
- deterministic first-occurrence pair ordering
- memory scales with observed label pairs, not a dense `(max_a + 1) x (max_b + 1)` matrix

## `regionprops`

```python
props = regionprops(labels)
```

Currently 2D only and intentionally limited to:

```text
label
area
bbox
centroid
area_bbox
```

Output shapes are:

```text
label       (N,)
area        (N,)
bbox        (N, 4)
centroid    (N, 2)
area_bbox   (N,)
```

Rows are sorted by ascending label. Bounding boxes use
`(min_row, min_col, max_row_exclusive, max_col_exclusive)`. Arbitrary/gappy `uint32` and
`uint64` IDs are handled directly without relabeling first.

## Benchmarks

Fresh CPU-only benchmark run for the current package on a MacBook Air with an Apple M3
(8 CPU cores: 4 performance and 4 efficiency), 16 GB RAM, macOS 14.8.7 arm64,
Python 3.12.13, NumPy 2.5.2, and scikit-image 0.26.0. Representative correctness
checks run before timing. Absolute timings are machine-dependent.

Each suite was launched twice; relabeling was launched three times after cooling because its
268 MB cases showed more thermal variance on the fanless MacBook Air. Tables use the best observed
timing for best-based scripts and the lower observed median for overlap. Ratios are calculated from
the displayed timings.

The dependency-free workload matrix smoke-checks every public operation before timing and covers
blocky, fragmented, sparse-ID, high-cardinality, all-background, strided, and mixed-removal cases:

```bash
uv run examples/benchmark_matrix.py
```

`fastremap` and StarDist remain optional benchmark competitors. The scripts skip them cleanly
when they are not installed; the fresh tables below include only competitors available in this run.

### Label counting

```bash
uv run examples/benchmark_label_counts.py
# Optional extra competitor:
uv run --with fastremap examples/benchmark_label_counts.py
```

The benchmark returns the same sparse `(ids, counts)` output for each method, excludes background,
and checks correctness against NumPy before timing. The `np.bincount` comparison includes extracting
only observed IDs, so it produces the same output shape as `label_counts`; it is skipped when
`max(label)` would require an impractically large dense count array. `fastremap` was not installed
in this benchmark environment, so it is included by the script but not in the table below.

| Mask | Objects | `fastlabelops` | NumPy `unique` | NumPy `bincount` + extract |
|---|---:|---:|---:|---:|
| 1024×1024 | 3,136 | **0.79 ms** | 7.45 ms (9.43×) | 5.96 ms (7.54×) |
| 2048×2048 | 12,769 | **3.21 ms** | 35.18 ms (10.96×) | 23.10 ms (7.20×) |
| 4096×4096 | 51,529 | **12.41 ms** | 153.04 ms (12.33×) | 93.39 ms (7.53×) |
| 2048×2048, sparse IDs up to 472M | 12,769 | **2.87 ms** | 34.78 ms (12.12×) | skipped |

### Relabeling

```bash
uv run --with scikit-image examples/benchmark_relabel.py
# Optional extra competitor:
uv run --with scikit-image --with fastremap examples/benchmark_relabel.py
```

The script reports best and mean timings; the table below uses the best timing.

| Mask | Observed IDs | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 2048×2048 | 3,016 | **5.79 ms** | 45.39 ms | **7.84×** |
| 8192×8192 | 30,742 | **115.75 ms** | 763.76 ms | **6.60×** |
| 8192×8192, sparse IDs up to 2B | 999 | **78.88 ms** | 730.53 ms | **9.26×** |

### Small-object removal

```bash
uv run --with scikit-image examples/benchmark_remove_small_objects.py
```

Correctness is checked against `skimage.morphology.remove_small_objects` with labeled `uint32`
inputs and `max_size=64` before timing. Both methods use their default copy-returning mode. The
script uses the best of repeated runs.

| Mask | Objects | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 1024×1024 | 2,601 | **1.66 ms** | 7.95 ms | **4.79×** |
| 2048×2048 | 10,404 | **6.47 ms** | 32.27 ms | **4.99×** |
| 4096×4096 | 41,616 | **27.14 ms** | 129.91 ms | **4.79×** |
| 2048×2048, sparse IDs | 10,404 | **6.60 ms** | 32.73 ms | **4.96×** |

### Sparse overlap counting

```bash
uv run --with scikit-image examples/benchmark_overlap.py
# Optional extra competitor:
uv run --with "stardist==0.9.2" --with scikit-image examples/benchmark_overlap.py
```

The overlap script reports the median of repeated runs. Normal cases count the full contingency,
including background.

| Mask | Observed pairs | `fastlabelops` | NumPy `unique` | `scikit-image` |
|---|---:|---:|---:|---:|
| 1024×1024, 1K-ID pool | 749 | **0.80 ms** | 23.56 ms (29.45×) | 35.09 ms (43.86×) |
| 2048×2048, 3K-ID pool | 2,801 | **3.50 ms** | 106.76 ms (30.50×) | 144.16 ms (41.19×) |
| 4096×4096, 5K-ID pool | 8,677 | **14.61 ms** | 468.48 ms (32.07×) | 828.52 ms (56.71×) |
| 4096×4096, 1K sparse IDs up to 2B | — | **13.50 ms** | 434.08 ms (32.15×) | skipped |

For the sparse-ID stress case, a dense matrix indexed directly by the observed maximum IDs would
require about **31.79 EB**, so scikit-image is intentionally skipped.

### Region properties

```bash
uv run --with scikit-image examples/benchmark_regionprops.py
```

All five supported properties are accessed inside the timed scikit-image call. The script uses
the best of repeated runs.

| Mask | Objects | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 1024×1024 | 3,136 | **0.81 ms** | 87.72 ms | **108.30×** |
| 2048×2048 | 12,769 | **3.02 ms** | 359.27 ms | **118.96×** |
| 4096×4096 | 51,529 | **13.47 ms** | 1479.79 ms | **109.86×** |
| 2048×2048, sparse IDs | 12,769 | **3.12 ms** | 377.05 ms | **120.85×** |

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src tests examples
uv run pre-commit install
```

See `examples/usage_example.py` for a runnable example.
