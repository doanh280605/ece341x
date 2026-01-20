# Session 1 Answers (fill in)

## 1) GPU info (current state)
1. GPU model + memory:
2. CUDA driver/runtime versions (from `nvidia-smi` and/or PyTorch):
3. Current GPU utilization / processes you see:
4. Paste (or summarize) output from `python python/bench_matmul.py --gpu-info-only`:

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
