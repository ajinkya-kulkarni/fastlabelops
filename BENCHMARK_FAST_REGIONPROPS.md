# fastlabelops vs fast-regionprops

Comparison of `fastlabelops.regionprops` against Martin Weigert's `fast-regionprops`.

## Method

`fast-regionprops` 0.1.0 and current `main` use the same computational path for the four properties shared with `fastlabelops`: `label`, `area`, `bbox`, and `centroid`. `fast-regionprops` does not provide `area_bbox`.

The benchmark therefore reports two Martin timings:

1. **Martin(4)** — requests only `label`, `area`, `bbox`, and `centroid`.
2. **Martin(4)+area_bbox** — the same call plus a NumPy calculation of `area_bbox` from the returned bbox columns.

`fastlabelops` is conservative in this comparison: its single `regionprops()` call always computes all five properties, including `area_bbox`.

All outputs are checked for numerical equality before timing. Times below are medians of repeated runs.

Run the benchmark with:

```bash
uv run --with fast-regionprops==0.1.0 examples/benchmark_fast_regionprops.py
```

## Results

Fresh run on Linux x86_64, AMD EPYC 9V74 (5 logical CPUs visible), Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0:

| Mask | Objects | `fastlabelops` (5 props) | Martin (4 props) | Martin (4 + `area_bbox`) | Speedup vs Martin 4 | Speedup vs equivalent 5 |
|---|---:|---:|---:|---:|---:|---:|
| 1024×1024 | 3,136 | **1.024 ms** | 57.714 ms | 58.288 ms | **56.33×** | **56.89×** |
| 2048×2048 | 12,769 | **3.969 ms** | 248.353 ms | 254.499 ms | **62.57×** | **64.12×** |
| 4096×4096 | 51,529 | **17.089 ms** | 1539.774 ms | 1502.813 ms | **90.10×** | **87.94×** |
| 2048×2048, sparse IDs | 12,769 | **4.181 ms** | 246.459 ms | 245.026 ms | **58.94×** | **58.60×** |

Absolute timings are machine-dependent. The meaningful result is that on these 2D label-mask workloads, the compiled single-scan `fastlabelops` implementation is substantially faster than the NumPy scatter implementation in `fast-regionprops`, even while computing one additional property.
