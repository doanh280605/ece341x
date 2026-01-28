#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <cmath>
#include "common.h"

/*
Lab TODO (students):
1) Understand the naive kernel (one thread computes one C[row,col]).
2) Allocate/copy A,B to device and C back to host.
3) Time the kernel with CUDA events.
4) Compare against CPU reference for correctness.
*/

// CUDA kernel, each thread computes one output element C[row, col] as the dot product of 
// row from A and col from B. Naive O(N) loop per thread. 
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// Fills a host vector with deterministic random floats in [-1, 1] using a fixed seed. Used to generate input matrices.
static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// Compare reference vs output with tolerance, print first mismatch adn returns pass/fail
static bool check_close(const std::vector<float>& ref, const std::vector<float>& out, float tol=1e-2f) {
    if (ref.size() != out.size()) return false;
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = std::fabs(ref[i] - out[i]);
        if (diff > tol) {
            fprintf(stderr, "Mismatch at %zu: ref=%f out=%f diff=%f\n", i, ref[i], out[i], diff);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int N = 512;
    int tile = 16;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tile") && i + 1 < argc) tile = std::atoi(argv[++i]);
    }

    printf("matmul_naive: N=%d tile=%d\n", N, tile);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N), hRef(N * N);
    fill_random(hA);
    fill_random(hB);

    // CPU reference
    double t0 = now_ms();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += hA[i * N + k] * hB[k * N + j];
            hRef[i * N + j] = acc;
        }
    }
    double t1 = now_ms();
    printf("CPU time: %.3f ms\n", (t1 - t0));

    // TODO-STUDENT: allocate dA,dB,dC with cudaMalloc
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;

    // each input is an N x N matrix, so need space for N^2 element. 
    CUDA_CHECK(cudaMalloc(&dA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * N * sizeof(float)));

    // TODO-STUDENT: copy hA,hB to device
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);

    // TODO-STUDENT: warmup kernel launch + synchronize
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, N); 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO-STUDENT: time kernel with CUDA events
    float ms = -1.0f;
    cudaEvent_t start, stop; 
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel time: %.3f ms (grid=%d,%d block=%d,%d)\n", ms, grid.x, grid.y, block.x, block.y);

    // TODO-STUDENT: copy dC back to hC and check correctness
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = check_close(hRef, hC);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // TODO-STUDENT: print FLOPs/TFLOP/s estimate (hint: 2*N^3 FLOPs)
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double seconds = ms / 1e3;
    double gflops = seconds > 0.0 ? (flops / seconds) / 1e9 : 0.0;
    double tflops = seconds > 0.0 ? (flops / seconds) / 1e12 : 0.0;
    printf("Perf: %.3f GFLOP/s (%.6f TFLOP/s)\n", gflops, tflops);
    // TODO-STUDENT: free device memory and destroy events
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ok ? 0 : 1;
}
