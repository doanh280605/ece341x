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
    /** 
    params: 
    __global__: marks a GPU kernel (runs on the GPU, launch from CPU)
    a, b: input float arrays
    c: output float array
    n: number of elements in each array
    */
    // TODO-STUDENT: implement
    // Hint: idx = blockIdx.x * blockDim.x + threadIdx.x
    // adds two float aray element-by-element. each GPU thread computes one output element. 

    /** 
    blockIdx.x: which block this thread is in 
    blockDim.x: number of threads per block
    threadIdx.x: thread's index inside its block
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // computes a unique global index for each thread
    if (idx < n) { // when the grid has more threads than elements
        c[idx] = a[idx] + b[idx]; // adds element idx
    }
}

// test input data, 
static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(0); // creates a Mersenne Twister RNG with fixed seed 0 
    // that means the random values are repeatable every run
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // defines the range. 
    for (auto& x : v) x = dist(rng); // the loop assigns each element x a new random float
}

//**
// compares the GPU output to the CPU reference and returns true only if every element 
// is within a small tolerance, so can verify correctness while allowing tiny floatijng point differences.  
// */
static bool check_close(const std::vector<float>& ref, const std::vector<float>& out, float tol=1e-4f) {
    if (ref.size() != out.size()) return false; // if sizes differ, return false
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = std::fabs(ref[i] - out[i]); // compute absolute difference
        if (diff > tol) {
            fprintf(stderr, "Mismatch at %zu: ref=%f out=%f diff=%f\n", i, ref[i], out[i], diff);
            return false; // if difference > tolerance, print a mismatch and return false
        }
    }
    return true;
}

/**
H2D: Host to Device
D2H: Device to Host
copies input arrays from CPU RAM to GPU VRAM, and then copies result back to CPU RAM for checking
*/

int main(int argc, char** argv) {
    int n = 10'000'000; // default vector size
    int block = 256; // default threads per block
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--block") && i + 1 < argc) block = std::atoi(argv[++i]);
    }

    printf("vecadd: n=%d block=%d\n", n, block);

    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n); // host arrays for inputs, output, and CPU reference. 
    fill_random(h_a);
    fill_random(h_b);

    // CPU reference
    double t0 = now_ms();
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i]; // CPU reference computation 
    double t1 = now_ms();
    printf("CPU time: %.3f ms\n", (t1 - t0));

    // TODO-STUDENT: allocate device buffers d_a, d_b, d_c
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;

    // TODO-STUDENT: cudaMalloc for each buffer
    // each input is a vector of length n, so allocate n * sizeof(float)
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float))); // allocate GPU memory for each array 
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    // TODO-STUDENT: copy H2D for a and b
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // copy inputs H2D
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // TODO-STUDENT: set grid, launch kernel, and time it with CUDA events
    int grid = (n + block - 1) / block; // compute number of blocks(ceil)

    // Warmup launch (recommended)
    // TODO-STUDENT: launch + synchronize
    vecadd_kernel<<<grid, block>>>(d_a, d_b, d_c, n); // warm up kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceSynchronize()); // wait so warm up finishes

    // Timed launch
    float ms = -1.0f;
    // TODO-STUDENT: create events, record start/stop, elapsed time
    cudaEvent_t start, stop; 
    CUDA_CHECK(cudaEventCreate(&start)); // create timing events
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start)); // start timing
    vecadd_kernel<<<grid, block>>>(d_a, d_b, d_c, n); // timed kernel launch 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); // end timing
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // compute elapsed time in ms
    printf("GPU kernel time: %.3f ms (grid=%d, block=%d)\n", ms, grid, block);

    // TODO-STUDENT: copy D2H to h_c and check correctness
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost)); // copy result D2H
    bool ok = check_close(h_ref, h_c); // validate GPU result vs CPU reference
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // TODO-STUDENT: free device memory and destroy events
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ok ? 0 : 1;
}
