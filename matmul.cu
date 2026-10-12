// Part 3: matrix multiplication C = A x B on the GPU, three ways.
//   1. naive   - one thread per output element, reads straight from global memory
//   2. tiled   - blocks load tiles of A and B into fast shared memory and reuse them
//   3. cuBLAS  - NVIDIA's tuned library, used as the reference and the target
//
// Why this matters for SAFE-Net: its residual block, autoencoder, BiLSTM gates
// and FC head are all Linear layers, i.e. GEMMs like this one, and cuDNN turns
// its convolutions into GEMM-like work too.
//
// Build (Windows, "x64 Native Tools Command Prompt for VS 2022"):
//   nvcc -O3 -arch=sm_120 matmul.cu -lcublas -o matmul
// sm_120 = compute capability 12.0 (RTX 50 series). Check yours with:
//   nvidia-smi --query-gpu=compute_cap --format=csv
// Run:  matmul > matmul_results.csv

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 32

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d (%s:%d)\n", (int)st,            \
                    __FILE__, __LINE__);                                     \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// 1. Naive: every thread computes one C element and reads a full row of A and
//    column of B from global memory. Lots of repeated, slow memory reads.
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// 2. Tiled: the block cooperatively copies a TILE x TILE piece of A and of B
//    into shared memory, then every thread reuses those values TILE times.
//    This cuts global-memory reads by a factor of about TILE.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        // Neighbouring threads read neighbouring addresses -> coalesced loads
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();                 // wait until the whole tile is loaded

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                 // wait before the tile is overwritten
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Time a kernel launch with CUDA events (GPU-side timing), averaged over reps.
template <typename Launch>
float time_ms(Launch launch, int reps = 10) {
    launch();                             // warm-up
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) launch();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s (compute capability %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // Keep cuBLAS in strict FP32 so the comparison with our FP32 kernels is fair
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    printf("size,kernel,time_ms,gflops,max_abs_error_vs_cublas\n");
    int sizes[] = {512, 1024, 2048, 4096};

    for (int N : sizes) {
        size_t bytes = (size_t)N * N * sizeof(float);
        std::vector<float> hA((size_t)N * N), hB((size_t)N * N),
                           hRef((size_t)N * N), hOut((size_t)N * N);
        srand(42);
        for (size_t i = 0; i < (size_t)N * N; ++i) {
            hA[i] = rand() / (float)RAND_MAX - 0.5f;
            hB[i] = rand() / (float)RAND_MAX - 0.5f;
        }

        float *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, bytes));
        CHECK(cudaMalloc(&dB, bytes));
        CHECK(cudaMalloc(&dC, bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        double flops = 2.0 * N * N * (double)N;   // one multiply + one add per term
        const float alpha = 1.0f, beta = 0.0f;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // cuBLAS is column-major. Computing B x A in column-major order gives
        // A x B in row-major order, which is how our arrays are stored.
        auto run_cublas = [&] {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, dB, N, dA, N, &beta, dC, N);
        };
        auto run_naive = [&] { matmul_naive<<<grid, block>>>(dA, dB, dC, N); };
        auto run_tiled = [&] { matmul_tiled<<<grid, block>>>(dA, dB, dC, N); };

        float t = time_ms(run_cublas);
        CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));
        printf("%d,cublas,%.3f,%.1f,0\n", N, t, flops / (t * 1e6));

        const char* names[] = {"naive", "tiled"};
        for (int k = 0; k < 2; ++k) {
            CHECK(cudaMemset(dC, 0, bytes));
            t = (k == 0) ? time_ms(run_naive, N >= 4096 ? 3 : 10)
                         : time_ms(run_tiled);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
            printf("%d,%s,%.3f,%.1f,%.2e\n", N, names[k], t, flops / (t * 1e6),
                   max_abs_diff(hOut, hRef));
        }
        fflush(stdout);
        CHECK(cudaFree(dA));
        CHECK(cudaFree(dB));
        CHECK(cudaFree(dC));
    }
    cublasDestroy(handle);
    return 0;
}
