CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -O3 -std=c++17

# If your cluster needs a specific arch, set it:
# e.g., for A100: -arch=sm_80
# ARCH ?= -arch=sm_80
ARCH ?= -arch=sm_80

all: vecadd matmul_naive

vecadd: vecadd.cu common.h
	$(NVCC) $(NVCCFLAGS) $(ARCH) -o $@ vecadd.cu

matmul_naive: matmul_naive.cu common.h
	$(NVCC) $(NVCCFLAGS) $(ARCH) -o $@ matmul_naive.cu

clean:
	rm -f vecadd matmul_naive
