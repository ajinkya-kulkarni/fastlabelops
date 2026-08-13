import statistics
import time
import numpy as np
from skimage.metrics import contingency_table  # type: ignore
from stardist.matching import _label_overlap, relabel_sequential  # type: ignore
from fastlabelops import overlap_counts

rng = np.random.default_rng(42)


def make_masks(shape, n_ids):
    h,w=shape; block=32; gh=(h+block-1)//block; gw=(w+block-1)//block
    ids=np.arange(1,n_ids+1,dtype=np.uint32); rng.shuffle(ids)
    a=rng.choice(ids,size=(gh,gw)); a[rng.random((gh,gw))<0.15]=0
    b=a.copy(); changed=rng.random((gh,gw))<0.20; b[changed]=rng.choice(ids,size=int(changed.sum())); b[rng.random((gh,gw))<0.15]=0
    a=np.repeat(np.repeat(a,block,0),block,1)[:h,:w]; b=np.repeat(np.repeat(b,block,0),block,1)[:h,:w]
    a,_,_=relabel_sequential(a); b,_,_=relabel_sequential(b)
    return np.ascontiguousarray(a,dtype=np.uint32),np.ascontiguousarray(b,dtype=np.uint32)


def fast(a,b): return overlap_counts(a,b,include_background=True)
def numpy_counts(a,b):
    keys=(a.ravel().astype(np.uint64)<<32)|b.ravel().astype(np.uint64)
    return np.unique(keys,return_counts=True)

def timed(fn,a,b,repeats=5):
    fn(a,b); times=[]
    for _ in range(repeats):
        t=time.perf_counter(); result=fn(a,b); times.append(time.perf_counter()-t); del result
    return statistics.median(times)


def bench(shape,n_ids,repeats=5):
    a,b=make_masks(shape,n_ids); _label_overlap(a,b)
    ft=timed(fast,a,b,repeats); nt=timed(numpy_counts,a,b,repeats)
    st=timed(lambda x,y: contingency_table(x,y,normalize=False),a,b,repeats); dt=timed(_label_overlap,a,b,repeats)
    print(f"{shape[0]}x{shape[1]}  fastlabelops={ft*1e3:.2f} ms  numpy={nt*1e3:.2f} ms ({nt/ft:.2f}x)  skimage={st*1e3:.2f} ms ({st/ft:.2f}x)  stardist={dt*1e3:.2f} ms ({dt/ft:.2f}x)")


if __name__ == "__main__":
    bench((1024,1024),1_000)
    bench((2048,2048),3_000)
    bench((4096,4096),5_000,repeats=3)
