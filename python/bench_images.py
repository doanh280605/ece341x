import argparse
import os
import csv
from typing import Dict, Any, Tuple

import torch

from gpu_utils import get_current_energy, nvml_shutdown
from timer_utils import time_gpu, time_cuda_events
from stats_utils import print_res


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def parse_mode(name: str) -> Tuple[str, torch.dtype, int]:
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
    if n == "fixed8":
        return ("fixed", torch.int32, 8)
    if n == "fixed16":
        return ("fixed", torch.int32, 16)
    raise ValueError(f"Unsupported dtype/mode: {name}")


def try_op(op_fn):
    try:
        op_fn()
        return True, ""
    except Exception as e:
        return False, str(e)


def to_fixed_int32(x_fp32: torch.Tensor, frac_bits: int, device: str) -> torch.Tensor:
    scale = float(1 << frac_bits)
    x = x_fp32.to(device=device, dtype=torch.float32)
    return torch.round(x * scale).to(torch.int32)


def fixed_linear(x_fixed: torch.Tensor, w_fixed: torch.Tensor, b_fixed: torch.Tensor, frac_bits: int) -> torch.Tensor:
    # y_int = x_int @ w_int + b_int, output scaled by 2^(2*frac_bits)
    y_int = x_fixed @ w_fixed + b_fixed
    denom = float(1 << (2 * frac_bits))
    return y_int.to(torch.float32) / denom


def dtype_supported_on_gpu(mode: str, torch_dtype: torch.dtype, frac_bits: int) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no_cuda"
    device = "cuda"

    if mode == "float":
        if torch_dtype == torch.bfloat16:
            ok, _ = try_op(lambda: (torch.randn(16, 16, device=device, dtype=torch.bfloat16)
                                    @ torch.randn(16, 16, device=device, dtype=torch.bfloat16)))
            if not ok:
                return False, "bf16_matmul_not_supported"
        return True, ""

    # For int/fixed, matmul support is very device/build dependent.
    if mode == "int":
        ok, msg = try_op(lambda: (torch.randint(-3, 3, (16, 16), device=device, dtype=torch_dtype)
                                  @ torch.randint(-3, 3, (16, 16), device=device, dtype=torch_dtype)))
        if not ok:
            return False, f"int_matmul_not_supported: {msg[:80]}"
        return True, ""

    if mode == "fixed":
        ok, msg = try_op(lambda: fixed_linear(
            to_fixed_int32(torch.randn(16, 16), frac_bits, device),
            to_fixed_int32(torch.randn(16, 16), frac_bits, device),
            to_fixed_int32(torch.randn(16, 16), frac_bits, device),
            frac_bits
        ))
        if not ok:
            return False, f"fixed_matmul_not_supported: {msg[:80]}"
        return True, ""

    return False, "unknown_mode"


def bench(repeat: int, dtype_name: str, trials: int, warmup: int, image_shape=(3, 224, 224)) -> Dict[str, Any]:
    '''
    repeat: simulate the number of images to be processed.

    Workload (inference-like):
      - flatten image
      - single linear layer (GEMM-like)
    We also explicitly measure H2D and D2H.
    '''
    mode, torch_dtype, frac_bits = parse_mode(dtype_name)

    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "no_cuda"}

    supported, reason = dtype_supported_on_gpu(mode, torch_dtype, frac_bits)
    if not supported:
        return {"skipped": True, "reason": reason, "dtype": dtype_name, "mode": mode}

    device = "cuda"
    c, h, w = image_shape
    in_dim = c * h * w
    out_dim = 1024

    # CPU source images
    x_cpu = torch.randn((repeat, c, h, w), dtype=torch.float32, device="cpu")

    # Reset energy baseline for this run
    get_current_energy(reset_baseline=True)

    # H2D timing
    def h2d():
        if mode == "float":
            return x_cpu.to(device=device, dtype=torch_dtype)
        if mode == "int":
            # Quantize to small integers (toy)
            return torch.round(x_cpu * 8).to(device=device, dtype=torch_dtype)
        # fixed
        return to_fixed_int32(x_cpu, frac_bits, device)

    h2d_t = time_gpu(h2d, trials=trials, warmup=warmup)
    x = h2d()
    torch.cuda.synchronize()

    # Prepare weights on GPU
    if mode == "float":
        W = torch.randn((in_dim, out_dim), device=device, dtype=torch_dtype)
        b = torch.randn((out_dim,), device=device, dtype=torch_dtype)
    elif mode == "int":
        W = torch.randint(-3, 3, (in_dim, out_dim), device=device, dtype=torch_dtype)
        b = torch.randint(-10, 10, (out_dim,), device=device, dtype=torch_dtype)
    else:
        W = to_fixed_int32(torch.randn((in_dim, out_dim), dtype=torch.float32), frac_bits, device)
        b = to_fixed_int32(torch.randn((out_dim,), dtype=torch.float32), frac_bits, device)

    def forward():
        x_flat = x.view(repeat, in_dim)
        if mode == "float":
            y = x_flat @ W
            y = y + b
            return y
        if mode == "int":
            return x_flat @ W + b
        # fixed
        return fixed_linear(x_flat, W, b, frac_bits)

    compute_t = time_cuda_events(forward, trials=trials, warmup=warmup)

    y = forward()
    torch.cuda.synchronize()

    def d2h():
        _ = y.to("cpu")

    d2h_t = time_gpu(d2h, trials=trials, warmup=warmup)

    total_ms = h2d_t.ms_median + compute_t.ms_median + d2h_t.ms_median
    energy = get_current_energy()
    joules = energy.get("joules", None)

    macs = int(repeat) * int(in_dim) * int(out_dim)
    flops = 2 * macs

    compute_s = compute_t.ms_median / 1000.0
    images_per_s = (repeat / compute_s) if compute_s > 0 else None
    tflops = (flops / compute_s) / 1e12 if compute_s > 0 else None

    j_per_image = (joules / repeat) if (joules is not None and repeat > 0) else None
    j_per_mac = (joules / macs) if (joules is not None and macs > 0) else None

    return {
        "repeat": repeat,
        "dtype": dtype_name,
        "mode": mode,
        "image_shape": str(image_shape),
        "h2d_ms_median": h2d_t.ms_median,
        "compute_ms_median": compute_t.ms_median,
        "d2h_ms_median": d2h_t.ms_median,
        "total_ms_median": total_ms,
        "images_per_s_compute": images_per_s,
        "energy_joules": joules,
        "energy_method": energy.get("method"),
        "FLOPs": flops,
        "MACs": macs,
        "TFLOP_per_s_compute": tflops,
        "J_per_image": j_per_image,
        "J_per_MAC": j_per_mac,
        "note": "Integer/fixed support varies; if skipped, that is expected and part of the lesson.",
    }


def write_csv(path: str, row: Dict[str, Any]):
    keys = list(row.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=4096)
    ap.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "int8", "int16", "int32", "fixed8", "fixed16"],
    )
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    ensure_results_dir()

    row = bench(repeat=args.repeat, dtype_name=args.dtype, trials=args.trials, warmup=args.warmup)
    out_csv = os.path.join("results", f"images_repeat_{args.repeat}_{args.dtype}.csv")
    write_csv(out_csv, row)

    print_res(f"Repeat-images summary ({args.dtype}) -> {out_csv}", row)
    nvml_shutdown()


if __name__ == "__main__":
    main()
