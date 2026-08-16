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
`uint64`. Background label `0` is excluded by default.

- supports `uint32` and `uint64`
- arbitrary NumPy dimensionality
- accepts non-contiguous inputs
- memory scales with the number of observed labels, not `max(label)`

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

Representative gains from the current AVX2 implementation versus the previous implementation on
segmentation-style workloads:

| Operation | Typical improvement |
|---|---:|
| `label_counts` | **1.6–1.7×** |
| `relabel_sequential` | **1.5–2.2×** |
| `remove_small_objects` | **1.2–1.5×** |
| `regionprops` | **1.1–1.2×** |
| `overlap_counts` | **1.2–2.0×** |

On a 2048² mask derived from an immunohistochemistry image, measured gains were **1.64×** for
`label_counts`, **1.53×** for relabeling, **1.22×** for `regionprops`, **1.70×** for foreground
`overlap_counts`, and **1.16×** for full-contingency overlap. A fragmented sparse prediction case
reached about **2.18×** for foreground overlap.

Run the dependency-free benchmark matrix with:

```bash
uv run examples/benchmark_matrix.py
```

Additional comparison scripts are available in `examples/` for label counting, relabeling,
small-object removal, overlap counting, and region properties.

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