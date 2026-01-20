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

static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

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

    // TODO-STUDENT: copy hA,hB to device

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);

    // TODO-STUDENT: warmup kernel launch + synchronize

    // TODO-STUDENT: time kernel with CUDA events
    float ms = -1.0f;
    printf("GPU kernel time: %.3f ms (grid=%d,%d block=%d,%d)\n", ms, grid.x, grid.y, block.x, block.y);

    // TODO-STUDENT: copy dC back to hC and check correctness
    bool ok = check_close(hRef, hC);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // TODO-STUDENT: print FLOPs/TFLOP/s estimate (hint: 2*N^3 FLOPs)
    // TODO-STUDENT: free device memory and destroy events

    return ok ? 0 : 1;
}
