from __future__ import annotations

import numpy as np

from . import _core

_SUPPORTED = (np.dtype(np.uint32), np.dtype(np.uint64))


def relabel_sequential(
    labels: np.ndarray,
    *,
    offset: int = 0,
    in_place: bool = False,
) -> tuple[np.ndarray, int]:
    """Relabel positive integer IDs sequentially in first-occurrence order.

    Background label 0 is preserved. New labels are ``offset + 1`` through
    ``offset + n_instances``.
    """
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    if labels.dtype not in _SUPPORTED:
        raise TypeError("labels dtype must be uint32 or uint64")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise TypeError("offset must be an integer")
    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if offset > np.iinfo(labels.dtype).max:
        raise OverflowError("offset exceeds label dtype range")

    if in_place:
        if not labels.flags.c_contiguous:
            raise ValueError("in_place=True requires a C-contiguous array")
        if not labels.flags.writeable:
            raise ValueError("in_place=True requires a writable array")
        out = labels
    else:
        out = np.array(labels, copy=True, order="C")

    n = _core.relabel_inplace(out, int(offset))
    return out, int(n)


def overlap_counts(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    include_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count sparse label-pair overlaps between two integer instance masks.

    Pairs are returned in deterministic first-occurrence order. By default,
    positions where either input label is 0 are ignored.
    """
    if not isinstance(labels_a, np.ndarray) or not isinstance(labels_b, np.ndarray):
        raise TypeError("labels_a and labels_b must be NumPy arrays")
    if labels_a.dtype not in _SUPPORTED or labels_b.dtype not in _SUPPORTED:
        raise TypeError("labels_a and labels_b dtypes must be uint32 or uint64")
    if labels_a.shape != labels_b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    a = np.ascontiguousarray(labels_a)
    b = np.ascontiguousarray(labels_b)
    return _core.overlap_counts(a, b, bool(include_background))


def regionprops(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Compute common properties for labels in a 2D integer instance mask.

    Returns ``label``, ``area``, ``bbox``, ``centroid``, and ``area_bbox``.
    Output rows are sorted by ascending label. Bounding boxes use
    ``(min_row, min_col, max_row_exclusive, max_col_exclusive)``.
    """
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")
    if labels.dtype not in _SUPPORTED:
        raise TypeError("labels dtype must be uint32 or uint64")

    out_label, stats, centroid = _core.regionprops2d(np.ascontiguousarray(labels))
    return {
        "label": out_label,
        "area": stats[:, 0],
        "bbox": stats[:, 1:5],
        "centroid": centroid,
        "area_bbox": stats[:, 5],
    }
