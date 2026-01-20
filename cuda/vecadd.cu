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
1) Implement vecadd_kernel (one thread per element).
2) Allocate device memory (cudaMalloc).
3) Copy inputs H2D (cudaMemcpy).
4) Launch kernel with a sensible grid/block.
5) Time kernel with CUDA events.
6) Copy output D2H and check correctness.
*/

__global__ void vecadd_kernel(const float* a, const float* b, float* c, int n) {
    // TODO-STUDENT: implement
    // Hint: idx = blockIdx.x * blockDim.x + threadIdx.x
}

static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

static bool check_close(const std::vector<float>& ref, const std::vector<float>& out, float tol=1e-4f) {
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
    int n = 10'000'000;
    int block = 256;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--block") && i + 1 < argc) block = std::atoi(argv[++i]);
    }

    printf("vecadd: n=%d block=%d\n", n, block);

    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
    fill_random(h_a);
    fill_random(h_b);

    // CPU reference
    double t0 = now_ms();
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];
    double t1 = now_ms();
    printf("CPU time: %.3f ms\n", (t1 - t0));

    // TODO-STUDENT: allocate device buffers d_a, d_b, d_c
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;

    // TODO-STUDENT: cudaMalloc for each buffer

    // TODO-STUDENT: copy H2D for a and b

    // TODO-STUDENT: set grid, launch kernel, and time it with CUDA events
    int grid = (n + block - 1) / block;

    // Warmup launch (recommended)
    // TODO-STUDENT: launch + synchronize

    // Timed launch
    float ms = -1.0f;
    // TODO-STUDENT: create events, record start/stop, elapsed time
    printf("GPU kernel time: %.3f ms (grid=%d, block=%d)\n", ms, grid, block);

    // TODO-STUDENT: copy D2H to h_c and check correctness
    bool ok = check_close(h_ref, h_c);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // TODO-STUDENT: free device memory and destroy events

    return ok ? 0 : 1;
}
