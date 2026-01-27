import time
from dataclasses import dataclass
from typing import Callable, Any

import torch


@dataclass
class TimingResult: # a data class to hold timing results
    ms_median: float
    ms_mean: float
    ms_min: float
    ms_max: float


def _percentile(values, p): # a helper function that calculates the percentile of a list of values
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return float(v[f])
    return float(v[f] * (c - k) + v[c] * (k - f))


def time_cpu(fn: Callable[[], Any], trials: int = 50, warmup: int = 10) -> TimingResult: # times CPU operations
    times = []
    for _ in range(warmup): # warmup iterations (let CPU settle)
        fn()
    for _ in range(trials): # collect timing for actual trials 
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return TimingResult(
        ms_median=_percentile(times, 50), # calculate statistics using percentile helper
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
    times = []
    # TODO-STUDENT: implement correctly

    for _ in range(warmup): # warmup iterations
        fn()

    torch.cuda.synchronize() # wait for all pending GPU works to finish 

    for _ in range(trials): 
        torch.cuda.synchronize() # ensure all previous GPU works are done
        t0 = time.perf_counter() # get current time 
        fn() # call the function 
        torch.cuda.synchronize() # ensure all GPU works are done
        t1 = time.perf_counter() # get current time after function call
        times.append((t1 - t0) * 1000.0) # convert to milliseconds
    
    return TimingResult(
        ms_median=_percentile(times, 50),
        ms_mean=sum(times) / len(times),
        ms_min=min(times),
        ms_max=max(times),
    )

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
    times = []
    # TODO-STUDENT: implement

    for _ in range(warmup): # warmup iterations
        fn()
    
    torch.cuda.synchronize() # wait for all pending GPU works to finish

    # actual timing trials using CUDA events 
    for _ in range(trials): 
        start = torch.cuda.Event(enable_timing=True) # create start event
        end = torch.cuda.Event(enable_timing=True) # create end event

        start.record() # record start event

        fn() # call the function

        end.record() # record end event

        torch.cuda.synchronize() # ensure all GPU works are done

        # get elapsed time between start and end events
        elapsed_time = start.elapsed_time(end) # in milliseconds
        times.append(elapsed_time)
    
    return TimingResult(
        ms_median=_percentile(times, 50),
        ms_mean=sum(times) / len(times),
        ms_min=min(times),
        ms_max=max(times),
    )

    raise NotImplementedError("Implement time_cuda_events() in python/timer_utils.py")
