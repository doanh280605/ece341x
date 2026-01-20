# Session 2 Answers (fill in)

## 1) “repeat images” simulation (throughput + energy)
For each dtype (FP32/FP16/BF16), report:
- time per iteration (ms)
- images/s
- energy per iteration (J) and/or energy per image (J/image)

Reference the CSV in `results/`.

---

## 2) Data movement cost (tensors)
1. How much time is H2D and D2H for your chosen repeat size?
2. If you keep tensors on GPU for multiple iterations, what happens and why?

---

## 3) Precision tradeoffs
1. Which dtype gives best **energy efficiency** (J/op or J/image)?
2. Which dtype gives best **throughput**?
3. How would you decide dtype for an AI accelerator workload?

---

## 4) CUDA naive matmul
1. Is naive matmul faster than PyTorch matmul on GPU? (It probably isn’t.)
2. From your output, what do you think the bottleneck is?
3. What optimization would you do next (shared memory tiling, better memory coalescing, etc.)?

**Screenshots saved:**
- `screenshots/s2_cuda_matmul_512.png`
- `screenshots/s2_cuda_matmul_1024.png`

---

## 5) (Optional) Profiling
If you profiled:
- What was the top time-consuming kernel/op?
- Did you see memcopies dominate?
- Include your screenshot filename.

If you could not profile due to restrictions, write 2–3 sentences describing the limitation.


## 6) (Optional) Quantized / fixed-point for 'repeat images'
1. Which integer/fixed modes were supported on GPU (if any)?
2. Compare energy/image and throughput where supported.
3. If unsupported, what kind of kernel/library support would be needed?
