from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _core

_SUPPORTED_DTYPES = (np.dtype(np.uint32), np.dtype(np.uint64))
_UINT64_MAX = int(np.iinfo(np.uint64).max)


class LabelCounts(NamedTuple):
    ids: np.ndarray
    counts: np.ndarray


class OverlapCounts(NamedTuple):
    a_ids: np.ndarray
    b_ids: np.ndarray
    counts: np.ndarray


def _validate_labels(labels: np.ndarray) -> None:
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    if labels.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("labels dtype must be uint32 or uint64")


def _validate_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _prepare_output(labels: np.ndarray, *, in_place: bool) -> np.ndarray:
    if in_place:
        if not labels.flags.c_contiguous:
            raise ValueError("in_place=True requires a C-contiguous array")
        if not labels.flags.writeable:
            raise ValueError("in_place=True requires a writable array")
        return labels
    return np.array(labels, copy=True, order="C")


def label_counts(
    labels: np.ndarray,
    *,
    include_background: bool = False,
) -> LabelCounts:
    """Count elements for each observed label in an integer instance mask.

    Label IDs are returned in ascending order. Background label 0 is excluded by
    default; set ``include_background=True`` to include it when it is present.
    Counts are returned as ``uint64``. When ``include_background=True`` and the
    background count is zero, no background row is emitted.
    """
    _validate_labels(labels)
    ids, counts = _core.label_counts(np.ascontiguousarray(labels), bool(include_background))
    return LabelCounts(ids, counts)


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
    _validate_labels(labels)
    offset = _validate_non_negative_int(offset, "offset")
    if offset > int(np.iinfo(labels.dtype).max):
        raise OverflowError("offset exceeds label dtype range")

    out = _prepare_output(labels, in_place=in_place)
    n = _core.relabel_inplace(out, offset)
    return out, int(n)


def remove_small_objects(
    labels: np.ndarray,
    *,
    max_size: int = 64,
    in_place: bool = False,
) -> np.ndarray:
    """Remove labeled objects whose size is at most ``max_size`` elements.

    Background label 0 is preserved. Nonzero labels are treated as existing
    instance IDs, so disconnected elements carrying the same ID count toward the
    same object's size. Surviving labels keep their original IDs.
    """
    _validate_labels(labels)
    max_size = _validate_non_negative_int(max_size, "max_size")
    if max_size > _UINT64_MAX:
        raise OverflowError("max_size exceeds uint64 range")

    out = _prepare_output(labels, in_place=in_place)
    _core.remove_small_objects_inplace(out, max_size)
    return out


def overlap_counts(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    include_background: bool = False,
) -> OverlapCounts:
    """Count sparse label-pair overlaps between two integer instance masks.

    Pairs are returned in deterministic first-occurrence order. By default,
    positions where either input label is 0 are ignored.
    """
    _validate_labels(labels_a)
    _validate_labels(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    a = np.ascontiguousarray(labels_a)
    b = np.ascontiguousarray(labels_b)
    a_ids, b_ids, counts = _core.overlap_counts(a, b, bool(include_background))
    return OverlapCounts(a_ids, b_ids, counts)


def regionprops(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Compute common properties for labels in a 2D integer instance mask.

    Returns ``label``, ``area``, ``bbox``, ``centroid``, and ``area_bbox``.
    Output rows are sorted by ascending label. Background label ``0`` is
    excluded from the output. Bounding boxes use
    ``(min_row, min_col, max_row_exclusive, max_col_exclusive)``.
    """
    _validate_labels(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    out_label, stats, centroid = _core.regionprops2d(np.ascontiguousarray(labels))
    return {
        "label": out_label,
        "area": stats[:, 0],
        "bbox": stats[:, 1:5],
        "centroid": centroid,
        "area_bbox": stats[:, 5],
    }
