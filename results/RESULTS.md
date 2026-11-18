# Results


## Part 1: precision x batch size (NVIDIA GeForce RTX 5070 Ti Laptop GPU)

|   batch |   bf16 |   fp16 |   fp32 |
|--------:|-------:|-------:|-------:|
|      32 | 1769   | 1528.9 | 1449.7 |
|      64 | 2124.2 | 2094.3 | 1347.4 |
|     128 | 2149.1 | 2238.7 | 1456.1 |
|     256 | 2191.2 | 2345.5 | 1516.9 |

*samples/s*

| Summary | Value | Note |
|---|---|---|
| Smallest batch tested | 32 |  |
| Largest batch that fit, fp32 | 256 | no OOM in the sweep |
| Largest batch that fit, fp16 | 256 | no OOM in the sweep |
| Largest batch that fit, bf16 | 256 | no OOM in the sweep |
| fp16 speed-up over fp32 | 1.55x | batch 256 |
| fp16 memory saving vs fp32 | 73% | batch 256 |
| fp16 best speed-up at any common batch | 1.55x |  |
| bf16 speed-up over fp32 | 1.44x | batch 256 |
| bf16 memory saving vs fp32 | 73% | batch 256 |
| bf16 best speed-up at any common batch | 1.58x |  |

## Optimisation: channels_last memory format

| precision   |   batch |   NCHW (default) |   channels_last |   channels_last + native BN | channels_last vs default   | channels_last + native BN vs default   |
|:------------|--------:|-----------------:|----------------:|----------------------------:|:---------------------------|:---------------------------------------|
| bf16        |      32 |           1769   |          1874.8 |                      1977.6 | 1.06x                      | 1.12x                                  |
| bf16        |      64 |           2124.2 |          2257.3 |                      2235.1 | 1.06x                      | 1.05x                                  |
| bf16        |     128 |           2149.1 |          2166.1 |                      2322.4 | 1.01x                      | 1.08x                                  |
| bf16        |     256 |           2191.2 |          2191.8 |                      2359.5 | 1.00x                      | 1.08x                                  |
| fp16        |      32 |           1528.9 |           771.1 |                      1522.4 | 0.50x                      | 1.00x                                  |
| fp16        |      64 |           2094.3 |           806   |                      2150.3 | 0.38x                      | 1.03x                                  |
| fp16        |     128 |           2238.7 |           732.2 |                      2340.7 | 0.33x                      | 1.05x                                  |
| fp16        |     256 |           2345.5 |           703.7 |                      2445.4 | 0.30x                      | 1.04x                                  |
| fp32        |      32 |           1449.7 |           665.3 |                      1317.1 | 0.46x                      | 0.91x                                  |
| fp32        |      64 |           1347.4 |           608.2 |                      1227.4 | 0.45x                      | 0.91x                                  |
| fp32        |     128 |           1456.1 |           551.3 |                      1324.3 | 0.38x                      | 0.91x                                  |
| fp32        |     256 |           1516.9 |           553.6 |                      1383.5 | 0.36x                      | 0.91x                                  |

*samples/s*


**Profiler breakdown (batch 128):**

| precision   | variant                   |   GPU ms/step |   batch-norm ms/step |   layout conversion ms/step |
|:------------|:--------------------------|--------------:|---------------------:|----------------------------:|
| fp32        | NCHW                      |        89.473 |               17.142 |                       0     |
| fp32        | channels_last             |       223.2   |              141.243 |                       2.447 |
| fp32        | channels_last + native BN |        93.815 |               15.094 |                       2.149 |
| fp16        | NCHW                      |        55.511 |               13.218 |                       7.372 |
| fp16        | channels_last             |       164.698 |              124.622 |                       4.717 |
| fp16        | channels_last + native BN |        52.383 |               11.711 |                       4.995 |
| bf16        | NCHW                      |        57.241 |               17.637 |                       6.251 |
| bf16        | channels_last             |        51.634 |               13.226 |                       4.225 |
| bf16        | channels_last + native BN |        53.479 |               14.066 |                       4.525 |


## Part 2c: data loading (ASVspoof 2019 LA train (FLAC from disk))

| setting     |   workers |   seconds |   samples_per_s | data_wait_share   | vs_gpu_bound   |
|:------------|----------:|----------:|----------------:|:------------------|:---------------|
| precomputed |       nan |      1.63 |          1957.8 | 0%                | 100%           |
| on_the_fly  |         0 |     24.32 |           131.6 | 95%               | 7%             |
| on_the_fly  |         2 |     10.24 |           312.6 | 87%               | 16%            |
| on_the_fly  |         4 |      5.91 |           541.4 | 74%               | 28%            |
| on_the_fly  |         8 |      3.55 |           900.8 | 39%               | 46%            |
| on_the_fly  |        12 |      2.8  |          1142.4 | 1%                | 58%            |
| on_the_fly  |        16 |      2.64 |          1213.6 | 2%                | 62%            |

| Summary | Value | Note |
|---|---|---|
| 0 workers (SAFE-Net default) | 24.32 s, 131.6 samples/s | waiting on data 95% |
| Best worker count | 16 workers, 1213.6 samples/s | 9.2x faster than 0 workers |
| GPU-only bound (precomputed features) | 1957.8 samples/s | best loader reaches 62% of it |

## Part 3: CUDA matmul vs cuBLAS

|   size |   cublas |   naive |   tiled |
|-------:|---------:|--------:|--------:|
|    512 |   5723.3 |   991.3 |  1238.1 |
|   1024 |   3657.5 |  1012.6 |  1328.3 |
|   2048 |  10792.8 |  1086.6 |  1360.2 |
|   4096 |  11843.2 |   882.4 |  1269.9 |

*GFLOPS*

| Summary | Value | Note |
|---|---|---|
| Best tiled GFLOPS | 1360 | N = 2048 |
| Tiled vs naive at N=4096 | 1.4x |  |
| Tiled as % of cuBLAS at N=4096 | 11% |  |
| Max error vs cuBLAS | 3.9e-05 | OK |
