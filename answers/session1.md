# Session 1 Answers (fill in)

## 1) GPU info (current state)
1. GPU model + memory: NVIDIA A30, 24GB
2. CUDA driver/runtime versions (from `nvidia-smi` and/or PyTorch): 570.133.07, CUDA 12.8, PyTorch use CUDA 12.1
3. Current GPU utilization / processes you see: 0% utilization, no running process. 4MiB memory used, power 29W
4. Paste (or summarize) output from `python python/bench_matmul.py --gpu-info-only`:
cuda_version: 12.1
device_name: NVIDIA A30
gpu_utilization_percent: 0.0
memory_total_mb: 24576.0
memory_used_mb: 4.0
memory_utilization_percent: 0.0
nvidia_smi_full: shows NVIDIA A30, 29W / 165W, 4MiB / 24576MiB, 0% util
power_draw_watts: 29.14
torch_version: 2.5.1
total_memory_gb: 23.598876953125

**Screenshots saved:**
- `screenshots/s1_nvidia_smi.png`
- `screenshots/s1_gpu_info.png`

---

## 2) Why PyTorch tensors (vs “lumpy arrays”)
Answer in ~6–10 sentences.

Prompt:
- Why are **PyTorch tensors** the right abstraction for accelerator computing?
- Why not a Python list-of-lists (“lumpy array”)?
- Mention: contiguity/strides, dtype, device placement, vectorization/kernels, autograd, and memory transfers.

---

## 3) CPU vs GPU matmul results (include transfer)
1. At what matrix size `N` does GPU **compute-only** beat CPU?
2. At what `N` does **end-to-end** GPU (including H2D + D2H) beat CPU?
3. When GPU loses, what dominates and why?

Attach references to your CSV rows (e.g., “see results/matmul_fp32.csv, N=512”).

---

## 4) Precision (FP32 vs FP16 vs BF16)
1. Which dtype was fastest on GPU? Did it depend on N?
2. Any numerical differences? (If you saw warnings, note them.)
3. Hypothesis: why might FP16/BF16 be faster or slower on your GPU?

---

## 5) CUDA vecadd
1. What block size did the program use?
2. Why is vector add typically **memory bandwidth** bound?
3. Compare CPU vs GPU timing for `n=1e7`. What speedup did you observe?

**Screenshot saved:**
- `screenshots/s1_vecadd.png`


## 6) (Optional) Integer / fixed-point experiments
1. Which of int8/int16/int32/fixed8/fixed16 were supported on GPU?
2. If some were skipped, what error/reason did the script report?
3. What does this teach you about deploying quantized/fixed-point workloads on GPUs/accelerators?
