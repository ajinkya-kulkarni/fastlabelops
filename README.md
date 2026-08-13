# fastlabelops

Fast CPU operations for integer instance masks. Requires Python 3.12+.

```python
from fastlabelops import overlap_counts, regionprops, relabel_sequential
```

The package combines the three standalone kernels into one lightweight NumPy + C++ package:

- `relabel_sequential(labels, offset=0, in_place=False)` — compact arbitrary IDs in first-occurrence order.
- `overlap_counts(a, b, include_background=False)` — sparse observed label-pair counts; no dense contingency allocation.
- `regionprops(labels)` — 2D `label`, `area`, `bbox`, `centroid`, and `area_bbox`.

Label `0` is background. `uint32` and `uint64` are supported. `overlap_counts` accepts mixed dtypes. The inputs already contain instance IDs; this package does not perform connected-component labeling.

## Example

```python
import numpy as np
from fastlabelops import overlap_counts, regionprops, relabel_sequential

labels = np.array([[0, 17, 17], [91, 0, 5002]], dtype=np.uint32)
relabeled, n = relabel_sequential(labels)
props = regionprops(labels)
a_ids, b_ids, counts = overlap_counts(labels, relabeled)
```

`regionprops` returns arrays with shapes `(N,)`, `(N,)`, `(N, 4)`, `(N, 2)`, and `(N,)` respectively. Rows are sorted by ascending label; bounding boxes use `(min_row, min_col, max_row_exclusive, max_col_exclusive)`.

## Benchmarks

The merged package uses the same kernels as the original standalone repos. Benchmark scripts are kept separately under `examples/`.

### Relabel

```bash
uv run --with scikit-image --with fastremap examples/benchmark_relabel.py
```

| Mask | `fastlabelops` | `skimage` | `fastremap` |
|---|---:|---:|---:|
| 2048×2048, ~3K IDs | **6.3 ms** | 46.9 ms (7.5×) | 8.5 ms (1.4×) |
| 8192×8192, ~31K IDs | **122.7 ms** | 791.7 ms (6.5×) | 167.3 ms (1.4×) |
| 8192×8192, ~1K IDs up to 2B | **94.2 ms** | 751.9 ms (8.0×) | 115.1 ms (1.2×) |

### Overlap

```bash
uv run --with "stardist==0.9.2" --with scikit-image examples/benchmark_overlap.py
```

| Mask | `fastlabelops` | NumPy `unique` | `scikit-image` | StarDist raw |
|---|---:|---:|---:|---:|
| 1024×1024, 1K IDs | **1.76 ms** | 23.67 ms | 36.40 ms | 3.74 ms |
| 2048×2048, 3K IDs | **7.04 ms** | 104.12 ms | 143.46 ms | 15.75 ms |
| 4096×4096, 5K IDs | **27.81 ms** | 475.07 ms | 835.10 ms | 82.70 ms |

### Region properties

```bash
uv run --with scikit-image examples/benchmark_regionprops.py
```

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
```
