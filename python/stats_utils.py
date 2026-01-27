from typing import Dict, Any


def matmul_macs(n: int) -> int:
    """
    TODO-STUDENT:
      Return how many multiply-accumulate operations are performed by an NxN matrix multiply C = A @ B. 
      C = A @ B: multiply matrix A by matrix B to get matrix C.
      Each entry in C is a dot product of one row of A with one column of B. 
      each output entry is a dot product of length N, so it uses N MACs. 
    """
    return n * n * n
    raise NotImplementedError("Implement matmul_macs() in python/stats_utils.py")


def matmul_flops(n: int) -> int:
    """
    TODO-STUDENT:
      should return the same work but in FLOPs.
      Use the common convention 1 MAC = 2 FLOPs (mul + add).
      Return total FLOPs for NxN matmul.
    """
    return 2 * n * n * n
    raise NotImplementedError("Implement matmul_flops() in python/stats_utils.py")


def print_res(title: str, stats: Dict[str, Any]) -> None:
    """
    print_res: pretty-print important statistics for the console (for screenshots).
    You can modify formatting if you want, but keep it readable.
    """
    print("=" * 80)
    print(title)
    print("-" * 80)
    for k in sorted(stats.keys()):
        print(f"{k:>24s}: {stats[k]}")
    print("=" * 80)
