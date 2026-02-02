# Session 1 Answers (fill in)

## 1) GPU info (current state)
1. GPU model + memory: NVIDIA A30, 24GB
2. CUDA driver/runtime versions (from `nvidia-smi` and/or PyTorch): 570.133.20, CUDA 12.8, PyTorch uses CUDA 12.1
3. Current GPU utilization / processes you see: 0% utilization, no running process. 4MiB memory used, power ~29W
4. Paste (or summarize) output from `python python/bench_matmul.py --gpu-info-only`:
device_capability: 8.0
nvidia_smi_full: Mon Feb  2 14:58:11 2026 (NVIDIA A30, 29W / 165W, 4MiB / 24576MiB, 0% util)
nvidia_smi_query: 0, 0, 4, 24576, 29.36
torch_cuda_version: 12.1
torch_device_name: NVIDIA A30
torch_device_total_memory_bytes: 25339101184
torch_total_memory_gb: 23.599
torch_version: 2.5.1

**Screenshots saved:**
- `screenshots/s1_nvidia_smi.png`
- `screenshots/s1_gpu_info.png`
- `screenshots/s1_matmul_fp32.png`
- `screenshots/s1_matmul_fp16.png`
- `screenshots/s1_matmul_bf16.png`

---

## 2) Why PyTorch tensors (vs “lumpy arrays”)
Answer in ~6–10 sentences.

Prompt:
- Why are **PyTorch tensors** the right abstraction for accelerator computing?
- Why not a Python list-of-lists (“lumpy array”)?
- Mention: contiguity/strides, dtype, device placement, vectorization/kernels, autograd, and memory transfers.

PyTorch tensors are the right abstraction because tensors are optimized for deep learning with GPU acceleration 
and automatic differences, while numpy is a general purpose library for CPU-based scientific computing. Built-in 
support via the Autograd module for gradient calculation. Tensors are typed, contiguous blocks of memory that map 
directly to efficient CPU/GPU kernels. They carry dtype and stride metadata, which lets the runtime pick vertorized 
kernels and avoid per-element Python above. Tensors also know their device placement, so the same code can run
on CPU or GPU and the framework can manage transfers and synchronization correctly. List-of-lists is fragmented 
across many Python objects, so there is no single contiguous buffer and no consistent stride layout for kernels 
to use. Lists also lack a uniform dtype, which forces dynamic type checks and blocks low-level vectorization or
GPU kernel launches.    

---

## 3) CPU vs GPU matmul results (include transfer)
1. At what matrix size `N` does GPU **compute-only** beat CPU?
2. At what `N` does **end-to-end** GPU (including H2D + D2H) beat CPU?

BF16: GPU wins compute-only and end-to-end at all N 
FP16: CPU is much slower across all N (CPU min is 17ms vs GPU totals all < 1ms for N and ~3.94 at N = 2048 for end-to-end)
FP32: GPU compute wins at all N, end-to-end loses to CPU only at N = 256 (CPU 0.112ms vs GPU 0.15ms)

3. When GPU loses, what dominates and why?
When GPU loses, data transfer and synchronization overhead dominates. For small N, the cost to move inputs to the GPU 
(H2D), launch kernels, and copy results back (D2H) is larger than the actual compute time, so total GPU time can exceed
CPU even though GPU compute is faster. For example, fp32 at N=256 has very fast compute but higher gpu_total_ms_median 
due to transfer overhead.

---

## 4) Precision (FP32 vs FP16 vs BF16)
1. Which dtype was fastest on GPU? Did it depend on N?
Overall, bf16 was the fastest on GPU, while fp32 is the slowest across all N
2. Any numerical differences? (If you saw warnings, note them.)
In bf16, for N = 1024, bf16 is slower than fp16 (~0.069ms vs ~0.066ms for compute-only and ~0.93ms vs ~0.9ms for 
end-to-end)
3. Hypothesis: why might FP16/BF16 be faster or slower on your GPU?
Fp16/Bf16 can be faster because they use tensor cores and move half the data, which boosts throughput and reduces
memory bandwidth pressure. They can be slower if GPU doesn't have strong tensor core support for that dtype.

---

## 5) CUDA vecadd
1. What block size did the program use?
Default is 256 threads (int block = 256)

2. Why is vector add typically **memory bandwidth** bound?
Each element does just one add but requries 2 reads + 1 write from global memory, so throughput is limited by 
memory transfer, not compute. 

3. Compare CPU vs GPU timing for `n=1e7`. What speedup did you observe?
CPU time: 7.609ms
GPU kernel time: 0.158ms
Speedup: CPU / GPU ~ 48.16x (about 48 times faster on GPU than CPU)

**Screenshot saved:**
- `screenshots/s1_vecadd.png`



## 6) (Optional) Integer / fixed-point experiments
1. Which of int8/int16/int32/fixed8/fixed16 were supported on GPU?
2. If some were skipped, what error/reason did the script report?
3. What does this teach you about deploying quantized/fixed-point workloads on GPUs/accelerators?
