from __future__ import annotations

import numpy as np

def relabel_inplace(labels: np.ndarray, offset: int) -> int: ...
def overlap_counts(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    include_background: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def regionprops2d(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
