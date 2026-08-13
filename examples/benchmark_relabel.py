import time
import fastremap  # type: ignore
import numpy as np
from skimage.segmentation import relabel_sequential as sk_relabel  # type: ignore
from fastlabelops import relabel_sequential

rng = np.random.default_rng(42)


def make_mask(shape, n_instances, max_id, dtype=np.uint32):
    h, w = shape
    mask = np.zeros(shape, dtype=dtype)
    ids = rng.integers(1, max_id + 1, size=n_instances, dtype=dtype)
    ys = rng.integers(0, h, size=n_instances)
    xs = rng.integers(0, w, size=n_instances)
    radii = rng.integers(5, 25, size=n_instances)
    for y, x, r, inst_id in zip(ys, xs, radii, ids, strict=True):
        y0, y1 = max(0, y-r), min(h, y+r); x0, x1 = max(0, x-r), min(w, x+r)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        region = mask[y0:y1, x0:x1]
        region[((yy-y)**2 + (xx-x)**2 <= r**2) & (region == 0)] = inst_id
    return mask


def timed(fn, arr, repeats=5):
    fn(np.array(arr, copy=True, order="C")); times=[]
    for _ in range(repeats):
        a=np.array(arr, copy=True, order="C"); t=time.perf_counter(); fn(a); times.append(time.perf_counter()-t)
    return min(times)


def bench(shape, n_instances, max_id, dtype=np.uint32, repeats=5):
    mask=make_mask(shape,n_instances,max_id,dtype)
    fast=relabel_sequential(mask)[0]; sk=sk_relabel(mask,offset=1)[0]
    assert np.unique(fast).size == np.unique(sk).size
    ft=timed(relabel_sequential,mask,repeats); st=timed(sk_relabel,mask,repeats)
    mt=timed(lambda a: fastremap.renumber(a,start=1,preserve_zero=True),mask,repeats)
    print(f"{shape[0]}x{shape[1]}  fastlabelops={ft*1e3:.2f} ms  skimage={st*1e3:.2f} ms ({st/ft:.2f}x)  fastremap={mt*1e3:.2f} ms ({mt/ft:.2f}x)")


if __name__ == "__main__":
    bench((2048,2048),5_000,5_000)
    bench((8192,8192),50_000,50_000,repeats=3)
    bench((8192,8192),1_000,2_000_000_000,dtype=np.uint32,repeats=3)
