import numpy as np

from fastlabelops import (
    label_counts,
    overlap_counts,
    regionprops,
    relabel_sequential,
    remove_small_objects,
)

labels = np.array([[0, 17, 17], [91, 0, 5002]], dtype=np.uint32)
ids, counts = label_counts(labels)
print("label counts:", list(zip(ids, counts, strict=True)))

relabeled, n = relabel_sequential(labels)
print("relabeled:\n", relabeled)
print("instances:", n)
print("props:", regionprops(labels))
print("without one-pixel objects:\n", remove_small_objects(labels, max_size=1))

other = np.array([[0, 4, 4], [9, 0, 3]], dtype=np.uint32)
a_ids, b_ids, overlap = overlap_counts(labels, other)
print("overlaps:", list(zip(a_ids, b_ids, overlap, strict=True)))
