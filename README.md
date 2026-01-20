# ECE341X Lab 1 — PyTorch + CUDA Basics on the Campus Cluster (SLURM)

**Total time:** 5.5 hours over **2 sessions** (2h50m each)  
**Theme:** CPU vs GPU performance, data movement cost, precision, and basic CUDA kernels.

This lab has **no formal report**. Instead, you will:
- Save **screenshots** into `screenshots/`
- Write short answers into files in `answers/`
- Produce CSV results in `results/`

---

## What you will learn

## Should we do everything in Python?
For **Session 1**, yes: using PyTorch in Python is the fastest way to learn correct timing, data transfers, and dtype effects.
For **Session 2**, we still include **CUDA C++** so you can see what “writing your own kernel” looks like and why it’s tricky.

You *can* put most experiments in one Python script/notebook, but CUDA kernels still need compilation (we keep them in `cuda/`).
If your cluster allows it, you can later explore compiling CUDA extensions from Python, but that can be slow and brittle on shared clusters.

---

## Supported dtypes in this lab
Float:
- `fp32`, `fp16`, `bf16`

Integer / fixed-point (educational):
- `int8`, `int16`, `int32`
- `fixed8`  (Q24.8 simulated using int32)
- `fixed16` (Q16.16 simulated using int32)

**Important:** Not every dtype is supported for every operation on every GPU.  
If a dtype/op is unsupported, the scripts will **skip** it and print a reason. This is part of the learning outcome.

1. How to run GPU jobs on the campus cluster using **SLURM**.
2. How to measure **CPU vs GPU** timing correctly in PyTorch (with synchronization and warmup).
3. Why **data movement (CPU↔GPU)** often dominates performance.
4. Why PyTorch uses **tensors** (and why “lumpy arrays” / Python lists are not suitable for accelerators).
5. How performance changes with **precision**: FP32 vs FP16 vs BF16.
6. Basic CUDA programming: **host/device memory**, **kernel launch**, and a first kernel (vector add).
7. How to estimate and report:
   - Time (ms)
   - Throughput (images/s or ops/s)
   - FLOPs and MACs (for matmul)
   - Energy (J) using NVML total energy if available (fallback: power×time)

---

## Repository layout
```
lab1_pytorch_cuda/
  README.md
  python/
    bench_matmul.py
    bench_images.py
    gpu_utils.py
    timer_utils.py
    stats_utils.py
  cuda/
    vecadd.cu
    matmul_naive.cu
    common.h
    Makefile
  scripts/
    slurm_interactive.sh
    slurm_batch.sh
  answers/
    session1.md
    session2.md
  screenshots/
    (you put screenshots here)
  results/
    (generated CSV + logs go here)
```

---

## 0) Cluster quick-start (SLURM)

### Option A: interactive (recommended for the lab)
Edit `scripts/slurm_interactive.sh` if needed for your cluster (partition/QOS/account).

```bash
bash scripts/slurm_interactive.sh
```

Inside the allocated node:
```bash
module load cuda  # if your cluster uses modules
module load gcc

# Create environment (one-time)
conda env create -f env.yml
conda activate ece341x-lab1

# Sanity checks
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Option B: batch run
```bash
sbatch scripts/slurm_batch.sh
# then view the output file named like slurm-XXXXX.out
```

---

## 1) Session 1 tasks (PyTorch timing + transfer + first CUDA kernel)

### Task 1.1 — Capture GPU “current state” info
Run:
```bash
python python/bench_matmul.py --gpu-info-only
```

Save:
- A screenshot of `nvidia-smi` to `screenshots/s1_nvidia_smi.png`
- A screenshot of the `--gpu-info-only` output to `screenshots/s1_gpu_info.png`

Fill answers in `answers/session1.md` (Section 1).

---

### Task 1.2 — CPU vs GPU matmul timing (including transfer)
Run:
```bash
python python/bench_matmul.py --sizes 256 512 1024 2048 --dtype fp32 --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 2048 --dtype fp16 --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 2048 --dtype bf16 --trials 50 --warmup 10
```

This generates CSVs in `results/` like:


You may also try **integer / fixed-point** modes (some may be skipped on your GPU/build):
```bash
python python/bench_matmul.py --sizes 256 512 1024 --dtype int8   --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 --dtype int16  --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 --dtype int32  --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 --dtype fixed8 --trials 50 --warmup 10
python python/bench_matmul.py --sizes 256 512 1024 --dtype fixed16 --trials 50 --warmup 10
```


- `matmul_fp32.csv`
- `matmul_fp16.csv`
- `matmul_bf16.csv`

Take one screenshot of the console summary for each dtype:
- `screenshots/s1_matmul_fp32.png`
- `screenshots/s1_matmul_fp16.png`
- `screenshots/s1_matmul_bf16.png`

Fill answers in `answers/session1.md` (Section 2–4).

---

### Task 1.3 — Vector add in CUDA C++
Build and run:
```bash
cd cuda
make clean && make vecadd
./vecadd --n 10000000
```

Save a screenshot of the output:
- `screenshots/s1_vecadd.png`

Fill answers in `answers/session1.md` (Section 5).

---

## 2) Session 2 tasks (images “repeat” simulation + CUDA matmul + profiling)

### Task 2.1 — Simulate “repeat” images and measure throughput + energy
This uses a tiny inference-like workload (flatten + linear) to simulate processing `--repeat` images.

```bash
python python/bench_images.py --repeat 4096 --dtype fp32 --trials 100 --warmup 20
python python/bench_images.py --repeat 4096 --dtype fp16 --trials 100 --warmup 20
python python/bench_images.py --repeat 4096 --dtype bf16 --trials 100 --warmup 20
```

Save screenshots:
- `screenshots/s2_images_fp32.png`
- `screenshots/s2_images_fp16.png`
- `screenshots/s2_images_bf16.png`

Fill answers in `answers/session2.md` (Section 1–3).


You may also try integer / fixed-point modes (may be skipped on your GPU/build):
```bash
python python/bench_images.py --repeat 4096 --dtype int8
python python/bench_images.py --repeat 4096 --dtype fixed8
```


---

### Task 2.2 — CUDA naive matmul kernel (required)
Build and run:
```bash
cd cuda
make clean && make matmul_naive
./matmul_naive --n 512
./matmul_naive --n 1024
```

Save screenshots:
- `screenshots/s2_cuda_matmul_512.png`
- `screenshots/s2_cuda_matmul_1024.png`

Fill answers in `answers/session2.md` (Section 4).

---

### Task 2.3 — (Optional) profiling
If Nsight tools are available on the cluster:
```bash
# Example: profile a python run
nsys profile -o results/nsys_matmul_fp32 python python/bench_matmul.py --sizes 2048 --dtype fp32 --trials 10 --warmup 5
```

Screenshot:
- `screenshots/s2_profile.png`

Fill answers in `answers/session2.md` (Section 5).

---

## Notes on timing correctness
- **Always warm up** GPU workloads (first run can include initialization/JIT).
- For GPU timing, you must **synchronize**:
  - `torch.cuda.synchronize()` before stopping a CPU timer.
  - Use CUDA events for precise kernel timing (we provide utilities).

---

## Notes on energy reporting
We try to read energy from NVML total energy counters:
- If supported: we report **Joules** since the start of the script (or since the last baseline).
- If not supported: we approximate energy as:
  - `energy ≈ average_power_watts × elapsed_seconds`

This is good enough for *relative comparisons* across dtypes and device/transfer choices.

---

## Submission checklist
Your submission must include:
- `answers/session1.md` and `answers/session2.md` filled out
- `screenshots/` populated with the required screenshots
- `results/` CSVs generated by the scripts
- Your code changes (if any) committed or included

---

## Common troubleshooting
- `torch.cuda.is_available()` is False  
  - You are not on a GPU node, or CUDA modules are not loaded.
- Permission errors / profiling blocked  
  - Profiling may be restricted. Skip Task 2.3 and note it in answers.
- BF16 not supported  
  - Some GPUs don’t support BF16 well. The script will warn and skip if needed.

Good luck — measure carefully and explain *why* the numbers look the way they do.


---

## Student code to fill in (required)
You must complete the TODO-STUDENT sections in:

### Python
- `python/gpu_utils.py`
  - `get_gpu_info()`
  - `get_current_energy()`
- `python/timer_utils.py`
  - `time_gpu()`
  - `time_cuda_events()`
- `python/stats_utils.py`
  - `matmul_macs()`
  - `matmul_flops()`

### CUDA C++
- `cuda/vecadd.cu`
- `cuda/matmul_naive.cu`

### Quick self-check
Run:
```bash
python python/check_impl.py
```
You should see all items report `[PASS]` before doing the full benchmarks.

---

## Solutions (for TAs / instructor)
A full working reference is provided in:
- `solutions/python/`
- `solutions/cuda/`

Do not edit the solutions folder during the lab; it’s there so staff can help debug quickly.

---

## Do you need to edit `bench_matmul.py`?
**No for the core lab.** We provide the benchmark scripts so everyone measures the *same thing* and avoids common mistakes.
Your required edits are in the utility modules (`gpu_utils.py`, `timer_utils.py`, `stats_utils.py`) and in the CUDA files.

If you want **extra credit**, see `GRADING.md` for optional edits you can add to `bench_matmul.py` (e.g., pinned memory / repeat-on-GPU).

---

## Autograde-ish checker (TA helper / self-check)
After you implement the TODOs and run some benchmarks, you can run:
```bash
python autograde_check.py
```

It will:
- check your TODO implementations (no NotImplementedError)
- attempt to build/run CUDA programs
- verify some required artifacts (CSVs/screenshots/answers)

It writes: `results/autograde_score.json`

Optional (batch):
```bash
sbatch scripts/slurm_grade_check.sh
```
