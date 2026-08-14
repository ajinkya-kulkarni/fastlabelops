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
Python 3.12.13, NumPy 2.5.2, and scikit-image 0.26.0. Correctness is checked before
timing. Absolute timings are machine-dependent.

Each suite was launched twice; relabeling was launched three times after cooling because its
268 MB cases showed more thermal variance on the fanless MacBook Air. Tables use the best observed
timing for best-based scripts and the lower observed median for overlap. Ratios are calculated from
the displayed timings.

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
| 1024×1024 | 3,136 | **0.73 ms** | 7.53 ms (10.32×) | 5.57 ms (7.63×) |
| 2048×2048 | 12,769 | **3.14 ms** | 33.77 ms (10.75×) | 23.02 ms (7.33×) |
| 4096×4096 | 51,529 | **12.36 ms** | 151.45 ms (12.25×) | 93.55 ms (7.57×) |
| 2048×2048, sparse IDs up to 472M | 12,769 | **3.09 ms** | 35.04 ms (11.34×) | skipped |

### Relabeling

```bash
uv run --with scikit-image examples/benchmark_relabel.py
# Optional extra competitor:
uv run --with scikit-image --with fastremap examples/benchmark_relabel.py
```

The script reports best and mean timings; the table below uses the best timing.

| Mask | Observed IDs | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 2048×2048 | 3,016 | **5.92 ms** | 45.66 ms | **7.71×** |
| 8192×8192 | 30,742 | **129.71 ms** | 763.71 ms | **5.89×** |
| 8192×8192, sparse IDs up to 2B | 999 | **76.83 ms** | 734.42 ms | **9.56×** |

### Small-object removal

```bash
uv run --with scikit-image examples/benchmark_remove_small_objects.py
```

Correctness is checked against `skimage.morphology.remove_small_objects` with labeled `uint32`
inputs and `max_size=64` before timing. Both methods use their default copy-returning mode. The
script uses the best of repeated runs.

| Mask | Objects | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 1024×1024 | 2,601 | **1.58 ms** | 7.99 ms | **5.06×** |
| 2048×2048 | 10,404 | **6.44 ms** | 33.02 ms | **5.13×** |
| 4096×4096 | 41,616 | **26.96 ms** | 131.03 ms | **4.86×** |
| 2048×2048, sparse IDs | 10,404 | **6.54 ms** | 32.69 ms | **5.00×** |

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
| 1024×1024, 1K-ID pool | 749 | **0.80 ms** | 22.67 ms (28.34×) | 34.65 ms (43.31×) |
| 2048×2048, 3K-ID pool | 2,801 | **3.95 ms** | 110.12 ms (27.88×) | 146.54 ms (37.10×) |
| 4096×4096, 5K-ID pool | 8,677 | **14.26 ms** | 470.89 ms (33.02×) | 804.54 ms (56.42×) |
| 4096×4096, 1K sparse IDs up to 2B | — | **13.35 ms** | 453.79 ms (33.99×) | skipped |

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
| 1024×1024 | 3,136 | **0.78 ms** | 88.32 ms | **113.23×** |
| 2048×2048 | 12,769 | **3.22 ms** | 360.47 ms | **111.95×** |
| 4096×4096 | 51,529 | **13.77 ms** | 1454.43 ms | **105.62×** |
| 2048×2048, sparse IDs | 12,769 | **3.22 ms** | 369.78 ms | **114.84×** |

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
