import time
import subprocess
from typing import Dict, Any, Optional

try:
    import torch
except Exception:
    torch = None

# NVML is preferred for power/energy. Some GPUs/drivers do not expose total energy.
try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetTotalEnergyConsumption,
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

_NVML_INIT = False
_HANDLE = None

_BASELINE_ENERGY_MJ: Optional[int] = None
_BASELINE_TIME_S: Optional[float] = None


def _run(cmd: str) -> str:
    """Run a shell command and return stdout (best-effort)."""
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return (e.output or "").strip()


def _nvml_ensure_init(device_index: int = 0) -> None:
    """
    Initialize NVML once and grab a GPU handle, set a baseline time and energy so later code can 
    compute "energy since baseline".

    TODO-STUDENT:
      - Call nvmlInit()
      - Set global _HANDLE = nvmlDeviceGetHandleByIndex(device_index)
      - Initialize baseline time and (if available) baseline total energy
    """
    global _NVML_INIT, _HANDLE, _BASELINE_ENERGY_MJ, _BASELINE_TIME_S

    if not _HAS_NVML or _NVML_INIT: # do nothing if NVML is not available or already initialized 
        return
    
    nvmlInit()
    _NVML_INIT = True
    _HANDLE = nvmlDeviceGetHandleByIndex(device_index) # stores GPU handle for later queries
    _BASELINE_TIME_S = time.time()  # records start time for elapsed calculations

    try: 
        _BASELINE_ENERGY_MJ = nvmlDeviceGetTotalEnergyConsumption(_HANDLE)
    except Exception: 
        _BASELINE_ENERGY_MJ = None

    # TODO-STUDENT: initialize NVML and set baselines
    nvmlInit()
    _NVML_INIT = True
    _HANDLE = nvmlDeviceGetHandleByIndex(device_index)
    _BASELINE_TIME_S = time.time()
    try:
        _BASELINE_ENERGY_MJ = nvmlDeviceGetTotalEnergyConsumption(_HANDLE)
    except Exception:
        _BASELINE_ENERGY_MJ = None


def get_gpu_info(device_index: int = 0) -> Dict[str, Any]:
    """
    get_gpu_info: return a dictionary of GPU info and *current state*.

    TODO-STUDENT (required):
      Include at least:
        - torch device name + total memory (if torch cuda available)
        - torch + cuda runtime versions (if available)
        - nvidia-smi query string output with utilization, memory used, power draw (best effort)
        - full `nvidia-smi` output text (best effort)

    Tip: use _run("nvidia-smi ...") and torch.cuda.get_device_properties.
    """
    # TODO-STUDENT: implement
    info: Dict[str, Any] = {} 

    if torch is not None and torch.cuda.is_available(): 
        props = torch.cuda.get_device_properties(device_index)
        info["torch_device_name"] = props.name
        info["torch_device_total_memory_bytes"] = props.total_memory
        info["torch_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 3)
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__
        info["device_capability"] = f"{props.major}.{props.minor}" 
    else: 
        info["torch_cuda_available"] = False
    
    info["nvidia_smi_query"] = _run(
        "nvidia-smi --query-gpu=ultilization.gpu,ultilization.memory,memory.used,memory.total,"
        "power.draw --format=csv,noheader,nounits"
    )

    info["nvidia_smi_full"] = _run("nvidia-smi")

    raise NotImplementedError("Implement get_gpu_info() in python/gpu_utils.py")


def get_power_watts(device_index: int = 0) -> Optional[float]:
    """Instantaneous power draw (Watts), best-effort."""
    if _HAS_NVML:
        try:
            _nvml_ensure_init(device_index)
            mw = nvmlDeviceGetPowerUsage(_HANDLE)  # milliwatts
            return float(mw) / 1000.0
        except Exception:
            pass

    out = _run("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")
    try:
        line = out.splitlines()[0].strip()
        return float(line)
    except Exception:
        return None


def get_current_energy(device_index: int = 0, reset_baseline: bool = False) -> Dict[str, Any]:
    """
    get_current_energy: return energy since the last baseline, either from NVML total energy counter (best)
    or by estimating power x time (fallback).

    Preferred: NVML total energy consumption counter (mJ) if supported.
    Fallback: approximate energy = power * time.

    TODO-STUDENT (required):
      - Support reset_baseline=True to reset baseline time (+ baseline energy if available).
      - If NVML total energy is available, return delta energy in Joules with method "nvml_total_energy_mj".
      - Otherwise, approximate energy with power×time and method "power_times_time_approx".

    Returns dict with keys:
      - joules (float or None)
      - method (string)
      - optionally power_watts, elapsed_s
    """
    # TODO-STUDENT: implement
    # _BASELINE_ENERGY_MJ: NVML total energy reading at baseline (MJ)
    global _BASELINE_ENERGY_MJ, _BASELINE_TIME_S # when we started measuring

    if _HAS_NVML:  
        _nvml_ensure_init(device_index) # sets up NVML and baseline values once

    # if reset baseline = true or no baseline yet, resets baseline time s and tries to capture
    # baseline energy mj
    if reset_baseline or _BASELINE_TIME_S is None: 
        _BASELINE_TIME_S = time.time()
        if _HAS_NVML and _HANDLE is not None: 
            try: 
                _BASELINE_ENERGY_MJ = nvmlDeviceGetTotalEnergyConsumption(_HANDLE) # HANDLE: NVML device handle
            except Exception: 
                _BASELINE_ENERGY_MJ = None
    
    # if NVML total energy is available, use it 
    # reads current total energy from NVML 
    # computes delta energy = current - baseline
    # convert mJ -> J and returns it with method nvml total energy mj
    if _HAS_NVML and _HANDLE is not None and _BASELINE_ENERGY_MJ is not None: 
        try: 
            now_mj = nvmlDeviceGetTotalEnergyConsumption(_HANDLE)
            delta_mj = now_mj - _BASELINE_ENERGY_MJ
            return {
                "joules": float(delta_mj) / 1000.0, 
                "method": "nvml_total_energy_mj",
            }
        except Exception: 
            pass
    
    # fallback: power + time
    # calls get power watts to get instantaneous power
    # computes elapsed time since baseline 
    # energy = power * time 
    # returns it with method
    power_watts = get_power_watts(device_index)
    elapsed_s = time.time() - (_BASELINE_TIME_S or time.time())
    joules = (power_watts * elapsed_s) if power_watts is not None else None

    return {
        "joules": joules, 
        "method": "power_times_time_approx", 
        "power_watts": power_watts, 
        "elapsed_s": elapsed_s,
    }

    raise NotImplementedError("Implement get_current_energy() in python/gpu_utils.py")


def nvml_shutdown() -> None:
    global _NVML_INIT
    if _HAS_NVML and _NVML_INIT:
        try:
            nvmlShutdown()
        except Exception:
            pass
        _NVML_INIT = False
