from __future__ import annotations

import numpy as np
import numpy.typing as npt

_LabelArray = npt.NDArray[np.uint32] | npt.NDArray[np.uint64]

def label_counts(
    labels: _LabelArray,
    include_background: bool,
) -> tuple[_LabelArray, npt.NDArray[np.uint64]]: ...
def relabel_inplace(labels: _LabelArray, offset: int) -> int: ...
def remove_small_objects_inplace(labels: _LabelArray, max_size: int) -> None: ...
def overlap_counts(
    labels_a: _LabelArray,
    labels_b: _LabelArray,
    include_background: bool,
) -> tuple[_LabelArray, _LabelArray, npt.NDArray[np.uint64]]: ...
def regionprops2d(
    labels: _LabelArray,
) -> tuple[_LabelArray, npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
