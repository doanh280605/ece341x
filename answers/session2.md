# Session 2 Answers (fill in)

## 1) “repeat images” simulation (throughput + energy)
For each dtype (FP32/FP16/BF16), report:
- time per iteration (ms)
- images/s
- energy per iteration (J) and/or energy per image (J/image)

FP32
- Time/iter: 375.44ms 
- Images/s: 26175.27
- Energy/iter: 4524.874J 
- Energy/image: 1.1047J

FP16
- Time/iter: 345.75 ms (total)
- Images/s: 366,837.85
- Energy/iter: 1640.559 J
- Energy/image: 0.4005 J

BF16
- Time/iter: 356.97 ms (total)
- Images/s: 382,940.03
- Energy/iter: 1651.117 J
- Energy/image: 0.4031 J

Reference the CSV in `results/`.

---

## 2) Data movement cost (tensors)
1. How much time is H2D and D2H for your chosen repeat size?
FP32: H2D ~ 217.99 ms, D2H ~ 0.97 ms
FP16: H2D ~ 333.70 ms, D2H ~ 0.89 ms
BF16: H2D ~ 345.75 ms, D2H ~ 0.52 ms

2. If you keep tensors on GPU for multiple iterations, what happens and why?
I can avoid repeated H2D/D2H transfer, total time drops and throughput improves. 

---

## 3) Precision tradeoffs
1. Which dtype gives best **energy efficiency** (J/op or J/image)?
FP16 (lowest J/image and J/MAC), and BG16 is very close

2. Which dtype gives best **throughput**?
BF16 (highest images/s), with FP16 close behind. 

3. How would you decide dtype for an AI accelerator workload?
Choose the lowest precision that preserves model accuracy and stability. Start with FP16/BF16, validate 
accuracy, and only use FP32 if numerics require it. 

---

## 4) CUDA naive matmul
1. Is naive matmul faster than PyTorch matmul on GPU? (It probably isn’t.)
No. Naive matmul is much slower than PyTorch's GPU matmul

2. From your output, what do you think the bottleneck is?
Bottleneck is memory access and low arithmetic intensity. Each thread repeatedly reads global memory with poor data reuse. 

3. What optimization would you do next (shared memory tiling, better memory coalescing, etc.)?
Share memory blocking to reuse A/B tiles, then improve memory combine and consider using vectorized loads or tensor cores

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
