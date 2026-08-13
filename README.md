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

The benchmark scripts keep the comparisons from the original standalone kernels and verify
correctness before timing where applicable.

### Relabeling

```bash
uv run --with scikit-image --with fastremap examples/benchmark_relabel.py
```

Sample results from Python 3.12 / NumPy 2.5.2 / Clang:

| Mask | `fastlabelops` | `skimage` | `fastremap` |
|---|---:|---:|---:|
| 2048×2048, ~3K IDs | **6.3 ms** | 46.9 ms (7.5×) | 8.5 ms (1.4×) |
| 8192×8192, ~31K IDs | **122.7 ms** | 791.7 ms (6.5×) | 167.3 ms (1.4×) |
| 8192×8192, ~1K IDs up to 2B | **94.2 ms** | 751.9 ms (8.0×) | 115.1 ms (1.2×) |

### Sparse overlap counting

```bash
uv run --with "stardist==0.9.2" --with scikit-image examples/benchmark_overlap.py
```

Sample MacBook Air results from the original overlap benchmark:

| Mask | `fastlabelops` | NumPy `unique` | `scikit-image` | StarDist raw |
|---|---:|---:|---:|---:|
| 1024×1024, 1K-ID pool | **1.76 ms** | 23.67 ms (13.42×) | 36.40 ms (20.63×) | 3.74 ms (2.12×) |
| 2048×2048, 3K-ID pool | **7.04 ms** | 104.12 ms (14.78×) | 143.46 ms (20.37×) | 15.75 ms (2.24×) |
| 4096×4096, 5K-ID pool | **27.81 ms** | 475.07 ms (17.08×) | 835.10 ms (30.02×) | 82.70 ms (2.97×) |
| 4096×4096, 1K sparse IDs up to 2B | **26.22 ms** | 460.92 ms (17.58×) | skipped | skipped |

### Region properties

```bash
uv run --with scikit-image examples/benchmark_regionprops.py
```

MacBook Air results for all five supported properties against
`skimage.measure.regionprops`:

| Mask | Objects | `fastlabelops` | `scikit-image` | Speedup |
|---|---:|---:|---:|---:|
| 1024×1024 | 3,136 | **1.01 ms** | 88.12 ms | **87.62×** |
| 2048×2048 | 12,769 | **4.03 ms** | 351.36 ms | **87.16×** |
| 4096×4096 | 51,529 | **16.31 ms** | 1439.00 ms | **88.22×** |
| 2048×2048, sparse IDs | 12,769 | **3.93 ms** | 370.42 ms | **94.34×** |

Absolute timings are machine-dependent.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/fastlabelops
uv run pre-commit install
```

See `examples/usage_example.py` for a runnable example.
