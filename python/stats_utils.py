from typing import Dict, Any


def matmul_macs(n: int) -> int:
    """
    TODO-STUDENT:
      For NxN matmul C=A@B, how many multiply-accumulate operations (MACs) are there?
      Return an integer.
    """
    raise NotImplementedError("Implement matmul_macs() in python/stats_utils.py")


def matmul_flops(n: int) -> int:
    """
    TODO-STUDENT:
      Use the common convention 1 MAC = 2 FLOPs (mul + add).
      Return total FLOPs for NxN matmul.
    """
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
