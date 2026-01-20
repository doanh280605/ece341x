import argparse
import os
import csv
from typing import List, Dict, Any, Tuple

import torch

from gpu_utils import get_gpu_info, get_current_energy, nvml_shutdown
from timer_utils import time_cpu, time_gpu, time_cuda_events
from stats_utils import matmul_flops, matmul_macs, print_res


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


# -----------------------------
# Dtype / fixed-point helpers
# -----------------------------
def parse_mode(name: str) -> Tuple[str, torch.dtype, int]:
    '''
    Returns (mode, torch_dtype, frac_bits)

    mode:
      - "float": normal floating-point tensor matmul
      - "int": integer tensor matmul (if supported)
      - "fixed": fixed-point simulated using int32 tensor math

    frac_bits is only used for "fixed" mode.
    '''
    n = name.lower()
    if n == "fp32":
        return ("float", torch.float32, 0)
    if n == "fp16":
        return ("float", torch.float16, 0)
    if n == "bf16":
        return ("float", torch.bfloat16, 0)

    if n == "int8":
        return ("int", torch.int8, 0)
    if n == "int16":
        return ("int", torch.int16, 0)
    if n == "int32":
        return ("int", torch.int32, 0)

    # Fixed-point simulation:
    # - fixed8:  Q24.8  (stored as int32, frac_bits=8)
    # - fixed16: Q16.16 (stored as int32, frac_bits=16)
    if n == "fixed8":
        return ("fixed", torch.int32, 8)
    if n == "fixed16":
        return ("fixed", torch.int32, 16)

    raise ValueError(f"Unsupported dtype/mode: {name}")


def try_op(op_fn) -> Tuple[bool, str]:
    try:
        op_fn()
        return True, ""
    except Exception as e:
        return False, str(e)


def to_fixed_int32(x_fp32: torch.Tensor, frac_bits: int, device: str) -> torch.Tensor:
    '''
    Convert float32 tensor to fixed-point stored in int32:
      x_fixed = round(x * 2^frac_bits)
    '''
    scale = float(1 << frac_bits)
    x = x_fp32.to(device=device, dtype=torch.float32)
    xq = torch.round(x * scale).to(torch.int32)
    return xq


def fixed_matmul(A_fixed: torch.Tensor, B_fixed: torch.Tensor, frac_bits: int) -> torch.Tensor:
    '''
    Fixed-point matmul:
      C_int = A_int @ B_int   (int32 accumulation if supported)
      C_fp  = C_int / 2^(2*frac_bits)
    Because products scale by 2^frac_bits twice.
    '''
    C_int = A_fixed @ B_fixed
    denom = float(1 << (2 * frac_bits))
    return C_int.to(torch.float32) / denom


def dtype_supported_on_gpu(mode: str, torch_dtype: torch.dtype, frac_bits: int) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no_cuda"

    device = "cuda"
    if mode == "float":
        # BF16 support varies; test a tiny matmul.
        if torch_dtype == torch.bfloat16:
            ok, msg = try_op(lambda: (torch.randn(16, 16, device=device, dtype=torch.bfloat16)
                                      @ torch.randn(16, 16, device=device, dtype=torch.bfloat16)))
            if not ok:
                return False, "bf16_matmul_not_supported"
        return True, ""

    if mode == "int":
        # Integer matmul support on CUDA varies a lot by dtype and build.
        ok, msg = try_op(lambda: (torch.randint(-3, 3, (16, 16), device=device, dtype=torch_dtype)
                                  @ torch.randint(-3, 3, (16, 16), device=device, dtype=torch_dtype)))
        if not ok:
            return False, f"int_matmul_not_supported: {msg[:80]}"
        return True, ""

    if mode == "fixed":
        # Uses int32 matmul under the hood.
        ok, msg = try_op(lambda: fixed_matmul(
            to_fixed_int32(torch.randn(16, 16), frac_bits, device),
            to_fixed_int32(torch.randn(16, 16), frac_bits, device),
            frac_bits
        ))
        if not ok:
            return False, f"fixed_matmul_not_supported: {msg[:80]}"
        return True, ""

    return False, "unknown_mode"


def bench_one_size(n: int, dtype_name: str, trials: int, warmup: int) -> Dict[str, Any]:
    mode, torch_dtype, frac_bits = parse_mode(dtype_name)

    # Source tensors (fp32 on CPU). Integer/fixed will be derived from these.
    a_src = torch.randn((n, n), device="cpu", dtype=torch.float32)
    b_src = torch.randn((n, n), device="cpu", dtype=torch.float32)

    # -----------------------------
    # CPU benchmark
    # -----------------------------
    def cpu_mm():
        if mode == "float":
            a = a_src.to(dtype=torch_dtype, device="cpu")
            b = b_src.to(dtype=torch_dtype, device="cpu")
            _ = a @ b
        elif mode == "int":
            # Simple int baseline: quantize to small ints then matmul
            a = torch.round(a_src * 8).to(torch_dtype)
            b = torch.round(b_src * 8).to(torch_dtype)
            _ = a @ b
        else:  # fixed
            a = to_fixed_int32(a_src, frac_bits, "cpu")
            b = to_fixed_int32(b_src, frac_bits, "cpu")
            _ = fixed_matmul(a, b, frac_bits)

    cpu_t = time_cpu(cpu_mm, trials=trials, warmup=warmup)

    if not torch.cuda.is_available():
        return {
            "N": n,
            "dtype": dtype_name,
            "mode": mode,
            "cpu_ms_median": cpu_t.ms_median,
            "cpu_ms_mean": cpu_t.ms_mean,
            "skipped": True,
            "reason": "no_cuda",
        }

    supported, reason = dtype_supported_on_gpu(mode, torch_dtype, frac_bits)
    if not supported:
        return {
            "N": n,
            "dtype": dtype_name,
            "mode": mode,
            "cpu_ms_median": cpu_t.ms_median,
            "skipped": True,
            "reason": reason,
        }

    device = "cuda"

    # Reset energy baseline for this size
    get_current_energy(reset_baseline=True)

    # -----------------------------
    # H2D timing
    # -----------------------------
    def h2d():
        if mode == "float":
            a = a_src.to(device=device, dtype=torch_dtype, non_blocking=False)
            b = b_src.to(device=device, dtype=torch_dtype, non_blocking=False)
        elif mode == "int":
            a = torch.round(a_src * 8).to(device=device, dtype=torch_dtype, non_blocking=False)
            b = torch.round(b_src * 8).to(device=device, dtype=torch_dtype, non_blocking=False)
        else:
            a = to_fixed_int32(a_src, frac_bits, device)
            b = to_fixed_int32(b_src, frac_bits, device)
        return a, b

    h2d_t = time_gpu(lambda: h2d(), trials=trials, warmup=warmup)
    a_gpu, b_gpu = h2d()
    torch.cuda.synchronize()

    # -----------------------------
    # Compute timing (CUDA events)
    # -----------------------------
    def gpu_mm():
        if mode == "float":
            _ = a_gpu @ b_gpu
        elif mode == "int":
            _ = a_gpu @ b_gpu
        else:
            _ = fixed_matmul(a_gpu, b_gpu, frac_bits)

    gpu_compute_t = time_cuda_events(gpu_mm, trials=trials, warmup=warmup)

    # -----------------------------
    # D2H timing (copy result back)
    # -----------------------------
    def compute_once():
        if mode == "float":
            return a_gpu @ b_gpu
        elif mode == "int":
            return a_gpu @ b_gpu
        else:
            return fixed_matmul(a_gpu, b_gpu, frac_bits)

    c_gpu = compute_once()
    torch.cuda.synchronize()

    def d2h():
        _ = c_gpu.to("cpu")

    d2h_t = time_gpu(d2h, trials=trials, warmup=warmup)

    total_ms = h2d_t.ms_median + gpu_compute_t.ms_median + d2h_t.ms_median

    # Energy since baseline
    e = get_current_energy()
    joules = e.get("joules", None)

    # Work metrics (still count MACs/FLOPs for matmul shape)
    flops = matmul_flops(n)
    macs = matmul_macs(n)
    compute_s = gpu_compute_t.ms_median / 1000.0
    tflops = (flops / compute_s) / 1e12 if compute_s > 0 else None
    tmacs = (macs / compute_s) / 1e12 if compute_s > 0 else None

    j_per_flop = (joules / flops) if (joules is not None and flops > 0) else None
    j_per_mac = (joules / macs) if (joules is not None and macs > 0) else None

    return {
        "N": n,
        "dtype": dtype_name,
        "mode": mode,
        "cpu_ms_median": cpu_t.ms_median,
        "gpu_h2d_ms_median": h2d_t.ms_median,
        "gpu_compute_ms_median": gpu_compute_t.ms_median,
        "gpu_d2h_ms_median": d2h_t.ms_median,
        "gpu_total_ms_median": total_ms,
        "speedup_compute_cpu_over_gpu": (cpu_t.ms_median / gpu_compute_t.ms_median) if gpu_compute_t.ms_median > 0 else None,
        "speedup_total_cpu_over_gpu": (cpu_t.ms_median / total_ms) if total_ms > 0 else None,
        "energy_joules": joules,
        "energy_method": e.get("method"),
        "FLOPs": flops,
        "MACs": macs,
        "TFLOP_per_s_compute": tflops,
        "TMAC_per_s_compute": tmacs,
        "J_per_FLOP": j_per_flop,
        "J_per_MAC": j_per_mac,
        "note": "Integer/fixed matmul support varies; skipped rows indicate unsupported ops on this GPU/build.",
    }


def write_csv(path: str, rows: List[Dict[str, Any]]):
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "int8", "int16", "int32", "fixed8", "fixed16"],
    )
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gpu-info-only", action="store_true")
    args = ap.parse_args()

    if args.gpu_info_only:
        info = get_gpu_info(0)
        print_res("get_gpu_info()", info)
        return

    ensure_results_dir()

    rows = []
    for n in args.sizes:
        rows.append(bench_one_size(n=n, dtype_name=args.dtype, trials=args.trials, warmup=args.warmup))

    out_csv = os.path.join("results", f"matmul_{args.dtype}.csv")
    write_csv(out_csv, rows)

    print_res(f"Matmul summary ({args.dtype}) -> {out_csv}", {
        "dtype": args.dtype,
        "sizes": args.sizes,
        "csv": out_csv,
        "note": "gpu_compute_ms_median uses CUDA events; gpu_total_ms_median sums H2D + compute + D2H medians",
    })
    for r in rows:
        print_res(f"N={r.get('N')} dtype={r.get('dtype')}", r)

    nvml_shutdown()


if __name__ == "__main__":
    main()
