"""Part 3 fallback: the same naive and tiled CUDA kernels, compiled at runtime by CuPy.

Use this only if nvcc / Visual Studio Build Tools won't set up. Same kernels,
same output format as matmul.cu (writes matmul_results.csv).

    pip install cupy-cuda12x
    python matmul_cupy.py
"""
import csv

import cupy as cp
import numpy as np

SRC = r"""
#define TILE 32
extern "C" __global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x, b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}
"""
mod = cp.RawModule(code=SRC)
naive, tiled = mod.get_function("matmul_naive"), mod.get_function("matmul_tiled")


def time_ms(fn, reps=10):
    fn()
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return cp.cuda.get_elapsed_time(s, e) / reps


def main():
    props = cp.cuda.runtime.getDeviceProperties(0)
    print("GPU:", props["name"].decode())
    rows = []
    for N in [512, 1024, 2048, 4096]:
        rng = np.random.default_rng(42)
        A = cp.asarray(rng.random((N, N), dtype=np.float32) - 0.5)
        B = cp.asarray(rng.random((N, N), dtype=np.float32) - 0.5)
        C = cp.empty((N, N), dtype=cp.float32)
        flops = 2.0 * N ** 3
        grid, block = ((N + 31) // 32, (N + 31) // 32), (32, 32)
        n32 = np.int32(N)

        t = time_ms(lambda: cp.matmul(A, B, out=C))          # cuBLAS SGEMM
        ref = C.copy()
        rows.append((N, "cublas", t, flops / (t * 1e6), 0.0))
        for name, k in [("naive", naive), ("tiled", tiled)]:
            C.fill(0)
            t = time_ms(lambda: k(grid, block, (A, B, C, n32)), reps=3 if name == "naive" and N >= 4096 else 10)
            err = float(cp.abs(C - ref).max())
            rows.append((N, name, t, flops / (t * 1e6), err))
        for r in rows[-3:]:
            print(f"N={r[0]:5d} {r[1]:7s} {r[2]:9.3f} ms {r[3]:9.1f} GFLOPS  err {r[4]:.2e}")

    with open("matmul_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "kernel", "time_ms", "gflops", "max_abs_error_vs_cublas"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.1f}", f"{r[4]:.2e}"])
    print("Saved matmul_results.csv")


if __name__ == "__main__":
    main()
