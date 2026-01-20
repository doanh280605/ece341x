#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static inline double now_ms() {
    // Host wall clock in milliseconds (not GPU time).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}
