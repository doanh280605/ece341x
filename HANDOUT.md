# Lab 1 Handout (Instructor-facing + Student-facing)

## Schedule overview (5.5h total)

### Session 1 (2h50m)
1) Cluster + environment setup (35m)  
2) Timing correctness + synchronization in PyTorch (20m)  
3) CPU vs GPU matmul + transfer cost + dtype sweep (75m)  
4) CUDA C++ vector add: host/device memory + kernel launch (35m)  
5) Wrap-up & checkpoints (5m)

### Session 2 (2h50m)
1) Repeat-images simulation: throughput + energy + dtype (55m)  
2) CUDA naive matmul (60m)  
3) Discussion: bottlenecks, why libraries are faster (20m)  
4) Optional profiling (35m)

---

## Checkpoints (what TAs should look for)
- Student can allocate a GPU node and show `nvidia-smi`.
- Student knows to call `torch.cuda.synchronize()` for timing.
- Student produces `results/matmul_fp32.csv` and at least one precision variant.
- Student produces vecadd output with correctness PASS.
- Student produces repeat-images CSV and can interpret transfer vs compute costs.
- Student runs CUDA matmul naive and understands why it’s slow.

---

## “Why tensors?” mini-rubric (what a good answer mentions)
- Explicit device placement (CPU vs GPU) and unified APIs
- Dtypes + contiguous storage + strides
- Vectorized kernels (cuBLAS/cuDNN) vs Python loops
- Autograd graph + operator fusion opportunities
- Data transfer costs and pinned memory concepts

---

## Notes on energy
- Prefer NVML total energy counters if supported.
- Otherwise use power×time as approximation.
- Grade based on correct methodology and *relative* comparisons, not absolute energy accuracy.
