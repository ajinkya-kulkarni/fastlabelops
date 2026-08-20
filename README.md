# fastlabelops

Fast CPU primitives for integer instance masks. Requires Python 3.12+.

`fastlabelops` provides five small operations commonly needed around instance segmentation and
labeled images:

- `label_counts` — count elements for each observed label
- `relabel_sequential` — compact arbitrary IDs to sequential labels
- `remove_small_objects` — remove labeled objects at or below a size threshold
- `overlap_counts` — count only observed label-pair overlaps between two masks
- `regionprops` — compute a small set of common per-label properties in 2D

The package is deliberately low-level: NumPy arrays in, NumPy arrays out, one compiled C++
extension, no scikit-image dependency at runtime.

## Installation

```bash
pip install fastlabelops
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
`uint64`. Background label `0` is excluded by default.

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- accepts non-contiguous inputs
- memory scales with the number of observed labels, not `max(label)`
- with `include_background=True`, the background row is emitted only when its count is nonzero

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
objects keep their original IDs. Disconnected regions carrying the same nonzero label are treated as
one instance.

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- `in_place=True` mutates a writable C-contiguous array
- `max_size=0` is a no-op

## `overlap_counts`

```python
a_ids, b_ids, counts = overlap_counts(labels_a, labels_b, include_background=False)
```

Returns only label pairs that actually occur. By default, positions where either input is background
are ignored. Use `include_background=True` when full foreground/background contingency counts are
needed.

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

Rows are sorted by ascending label. Bounding boxes use
`(min_row, min_col, max_row_exclusive, max_col_exclusive)`. Arbitrary/gappy `uint32` and
`uint64` IDs are handled directly without relabeling first.

## Performance

`fastlabelops` is CPU-only. On supported x86 CPUs, `uint32` workloads automatically use AVX2 run
scanning where it helps; lightweight input sampling selects between SIMD and fallback paths.

Representative speedups versus common reference implementations on 2048²–4096² instance masks:

| Operation | vs scikit-image | vs NumPy | vs fastremap |
|---|---:|---:|---:|
| `label_counts` | — | **9–16×** | **4–7×** |
| `relabel_sequential` | **6–9×** | — | **1.1–1.3×** |
| `remove_small_objects` | **4.5–5×** | — | — |
| `overlap_counts` | **38–55×** | **26–33×** | — |
| `regionprops` | **95–97×** | — | — |

`overlap_counts` and `regionprops` show the largest gains because the reference implementations
size internal structures by `max(label) + 1`, which is prohibitively expensive for gappy or
sparse IDs. `fastlabelops` sizes by *observed* labels, so sparse IDs cost the same as compact ones.
In the benchmarked sparse `uint64` cases, scikit-image's `remove_small_objects` and `regionprops`
raised `MemoryError` when `max(label)` exceeded available memory.

Run the comparison benchmarks (requires `scikit-image`, `scipy`, and `fastremap`):

```bash
uv sync --dev
uv run examples/benchmark_label_counts.py
uv run examples/benchmark_relabel.py
uv run examples/benchmark_remove_small_objects.py
uv run examples/benchmark_overlap.py
uv run examples/benchmark_regionprops.py
```

The dependency-free benchmark matrix is also available:

```bash
uv run examples/benchmark_matrix.py
```

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
