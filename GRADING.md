# Lab 1 Grading (100 points)

This lab is graded by:
1) Correctness of required TODO implementations (Python + CUDA)
2) Required outputs (CSVs + screenshots + answer files)
3) Basic reasoning in the answer files (short but clear)

> **Policy:** If your code runs but your timing/energy methodology is incorrect (missing sync, no warmup, etc.),
> you will lose points even if you have “numbers”.

---

## A) Python TODOs (45 points)

### A1) `python/gpu_utils.py` (25 pts)
- **get_gpu_info()** (10 pts)
  - [3] Includes device name + total memory when CUDA available
  - [3] Includes torch + CUDA runtime versions when available
  - [2] Includes nvidia-smi query output (util/mem/power) best effort
  - [2] Includes full `nvidia-smi` output best effort

- **get_current_energy()** (15 pts)
  - [5] Supports `reset_baseline=True` correctly
  - [5] Uses NVML total energy delta when available (Joules + method string)
  - [5] Falls back to power×time approximation when energy counter absent (returns method + power/elapsed)

### A2) `python/timer_utils.py` (20 pts)
- **time_gpu()** (8 pts)
  - [4] Proper `torch.cuda.synchronize()` before and after timed region
  - [2] Warmup iterations included
  - [2] Returns correct TimingResult stats (median/mean/min/max)

- **time_cuda_events()** (12 pts)
  - [5] Correct CUDA event usage (`start.record()`, `end.record()`, synchronize, `elapsed_time`)
  - [3] Warmup + synchronize before trials
  - [4] Returns correct TimingResult stats

### A3) `python/stats_utils.py` (5 pts)
- **matmul_macs()** (3 pts): returns N^3 as integer for NxN matmul
- **matmul_flops()** (2 pts): returns 2*N^3 as integer (1 MAC = 2 FLOPs)

---

## B) CUDA TODOs (35 points)

### B1) `cuda/vecadd.cu` (15 pts)
- [4] Correct kernel indexing + bounds check
- [5] Correct cudaMalloc / cudaMemcpy H2D / cudaMemcpy D2H
- [4] Correct kernel launch config (grid computed from n and block)
- [2] Correct CUDA event timing (and prints ms)

### B2) `cuda/matmul_naive.cu` (20 pts)
- [6] Correct device alloc/copies and D2H result
- [6] Correct CUDA event timing block (warmup + timed region)
- [4] Correctness check PASS for N=512 (or class-specified size)
- [4] Prints FLOPs + TFLOP/s estimate (2*N^3 FLOPs / time)

---

## C) Required outputs (20 points)

### C1) Results (10 pts)
- [6] CSVs generated in `results/` for at least:
  - `matmul_fp32.csv`
  - one other dtype (fp16/bf16/int*/fixed*)
  - one `images_repeat_...csv`
- [4] Outputs are plausible (non-zero times, speedups make sense)

### C2) Screenshots + answers (10 pts)
- [6] Required screenshots present (per README checklist)
- [4] `answers/session1.md` and `answers/session2.md` completed with concise explanations

---

## Extra credit (up to +10 points)
Pick ONE:
1) Add a `--pinned` option in `bench_matmul.py` and measure H2D/D2H change (with evidence), OR
2) Add a “keep tensors on GPU for K repeats” option and show the effect on end-to-end vs compute-only, OR
3) Add optional profiling screenshot + 3 takeaways (if Nsight allowed).

Document extra credit in the answer files and include screenshots.
