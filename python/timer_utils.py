import time
from dataclasses import dataclass
from typing import Callable, Any

import torch


@dataclass
class TimingResult:
    ms_median: float
    ms_mean: float
    ms_min: float
    ms_max: float


def _percentile(values, p):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return float(v[f])
    return float(v[f] * (c - k) + v[c] * (k - f))


def time_cpu(fn: Callable[[], Any], trials: int = 50, warmup: int = 10) -> TimingResult:
    times = []
    for _ in range(warmup):
        fn()
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return TimingResult(
        ms_median=_percentile(times, 50),
        ms_mean=sum(times) / len(times),
        ms_min=min(times),
        ms_max=max(times),
    )


def time_gpu(fn: Callable[[], Any], trials: int = 50, warmup: int = 10) -> TimingResult:
    """
    GPU timing using torch.cuda.synchronize + CPU wall clock.
    Good enough for end-to-end timings.

    TODO-STUDENT:
      - Make sure you synchronize *before* starting and *after* finishing the timed region.
      - Keep warmup iterations.
    """
    assert torch.cuda.is_available(), "CUDA not available"
    # TODO-STUDENT: implement correctly
    raise NotImplementedError("Implement time_gpu() in python/timer_utils.py")


def time_cuda_events(fn: Callable[[], Any], trials: int = 50, warmup: int = 10) -> TimingResult:
    """
    Kernel timing using CUDA events (more precise for device work).

    TODO-STUDENT (required):
      - Create start/end events (torch.cuda.Event(enable_timing=True))
      - Warmup fn() warmup times, then synchronize
      - For each trial:
          start.record(); fn(); end.record(); synchronize; read elapsed_time
      - Return TimingResult with median/mean/min/max
    """
    assert torch.cuda.is_available(), "CUDA not available"
    # TODO-STUDENT: implement
    raise NotImplementedError("Implement time_cuda_events() in python/timer_utils.py")
