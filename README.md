# fastlabelops

Fast CPU primitives for integer instance masks. Requires Python 3.12+.

`fastlabelops` combines three small operations that are commonly needed around instance
segmentation and labeled images:

- `relabel_sequential` — compact arbitrary IDs to sequential labels
- `overlap_counts` — count only observed label-pair overlaps between two masks
- `regionprops` — compute a small set of common per-label properties in 2D

The package is deliberately low-level: NumPy arrays in, NumPy arrays out, one compiled C++
extension, no scikit-image dependency at runtime.

```bash
python -m pip install .
```

```python
import numpy as np
from fastlabelops import overlap_counts, regionprops, relabel_sequential

labels = np.array(
    [
        [0, 17, 17],
        [91, 0, 5002],
    ],
    dtype=np.uint32,
)

relabeled, n = relabel_sequential(labels)
props = regionprops(labels)

a_ids, b_ids, counts = overlap_counts(labels, relabeled)
```

Label `0` is background throughout. Inputs already contain instance IDs; none of these functions
perform connected-component labeling.

## `relabel_sequential`

```python
labels, n = relabel_sequential(labels, offset=0, in_place=False)
```

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- deterministic first-occurrence relabeling
- `in_place=True` mutates a writable C-contiguous array
- `offset=N` starts foreground labels at `N + 1`
- memory scales with the number of observed labels, not `max(label)`

## `overlap_counts`

```python
a_ids, b_ids, counts = overlap_counts(labels_a, labels_b)
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

Fresh benchmark run for the merged package on Linux x86_64 with an AMD EPYC 9V74
(5 logical CPUs visible), Python 3.13.5, NumPy 2.3.5, and scikit-image 0.26.0.
Correctness is checked before timing. Absolute timings are machine-dependent.

`fastremap` and StarDist remain optional benchmark competitors. The scripts skip them cleanly
when they are not installed; the fresh tables below include only competitors available in this run.

### Relabeling

```bash
uv run --with scikit-image examples/benchmark_relabel.py
# Optional extra competitor:
uv run --with scikit-image --with fastremap examples/benchmark_relabel.py
```

The script reports best and mean timings; the table below uses the best timing.

| Mask | Observed IDs | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 2048×2048 | 3,016 | **5.31 ms** | 19.95 ms | **3.76×** |
| 8192×8192 | 30,742 | **110.46 ms** | 513.95 ms | **4.65×** |
| 8192×8192, sparse IDs up to 2B | 999 | **79.05 ms** | 504.49 ms | **6.38×** |

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
| 1024×1024, 1K-ID pool | 749 | **0.70 ms** | 40.85 ms (58.77×) | 17.70 ms (25.47×) |
| 2048×2048, 3K-ID pool | 2,801 | **2.79 ms** | 211.50 ms (75.84×) | 75.82 ms (27.19×) |
| 4096×4096, 5K-ID pool | 8,677 | **12.30 ms** | 919.58 ms (74.74×) | 454.91 ms (36.97×) |
| 4096×4096, 1K sparse IDs up to 2B | — | **11.90 ms** | 836.63 ms (70.29×) | skipped |

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
| 1024×1024 | 3,136 | **1.00 ms** | 82.45 ms | **82.86×** |
| 2048×2048 | 12,769 | **3.97 ms** | 343.04 ms | **86.31×** |
| 4096×4096 | 51,529 | **16.08 ms** | 1525.61 ms | **94.89×** |
| 2048×2048, sparse IDs | 12,769 | **4.13 ms** | 357.01 ms | **86.44×** |

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
