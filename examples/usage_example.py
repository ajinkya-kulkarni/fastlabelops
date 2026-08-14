import numpy as np

from fastlabelops import overlap_counts, regionprops, relabel_sequential, remove_small_objects

labels = np.array([[0, 17, 17], [91, 0, 5002]], dtype=np.uint32)
relabeled, n = relabel_sequential(labels)
print("relabeled:\n", relabeled)
print("instances:", n)
print("props:", regionprops(labels))
print("without one-pixel objects:\n", remove_small_objects(labels, max_size=1))

other = np.array([[0, 4, 4], [9, 0, 3]], dtype=np.uint32)
a_ids, b_ids, counts = overlap_counts(labels, other)
print("overlaps:", list(zip(a_ids, b_ids, counts, strict=True)))
