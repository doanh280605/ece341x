import traceback

def run_check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except NotImplementedError as e:
        print(f"[TODO]  {name}: {e}")
    except Exception as e:
        print(f"[FAIL]  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

def main():
    import torch
    from gpu_utils import get_gpu_info, get_current_energy
    from timer_utils import time_gpu, time_cuda_events
    from stats_utils import matmul_macs, matmul_flops

    run_check("matmul_macs(16)", lambda: matmul_macs(16))
    run_check("matmul_flops(16)", lambda: matmul_flops(16))

    if torch.cuda.is_available():
        x = torch.randn(1024, device="cuda")
        run_check("time_gpu(no-op)", lambda: time_gpu(lambda: x.add_(1), trials=5, warmup=2))
        run_check("time_cuda_events(no-op)", lambda: time_cuda_events(lambda: x.add_(1), trials=5, warmup=2))
        run_check("get_gpu_info()", lambda: get_gpu_info(0))
        run_check("get_current_energy(reset)", lambda: get_current_energy(reset_baseline=True))
        run_check("get_current_energy(delta)", lambda: get_current_energy())
    else:
        print("CUDA not available: skip GPU checks")

if __name__ == "__main__":
    main()
