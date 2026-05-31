import os
from typing import Optional

import torch


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak memory counters when measuring a CUDA generation run."""
    if device is not None:
        device = torch.device(device)
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def current_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return current memory usage in MB for CUDA or the current CPU process."""
    if device is not None:
        device = torch.device(device)
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        return _bytes_to_mb(torch.cuda.memory_allocated(device))
    return _process_memory_mb()


def peak_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return peak CUDA memory in MB, or current process memory for CPU runs."""
    if device is not None:
        device = torch.device(device)
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        return _bytes_to_mb(torch.cuda.max_memory_allocated(device))
    return _process_memory_mb()


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def _process_memory_mb() -> float:
    try:
        import psutil

        return _bytes_to_mb(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0.0
