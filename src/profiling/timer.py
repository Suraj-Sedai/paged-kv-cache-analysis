import time
from contextlib import contextmanager
from typing import Iterator, Optional

import torch


def synchronize_if_needed(device: Optional[torch.device] = None) -> None:
    """Synchronize CUDA work before reading a wall-clock timer."""
    if device is not None:
        device = torch.device(device)
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.synchronize(device)


def now_seconds(device: Optional[torch.device] = None) -> float:
    """Return synchronized wall-clock time in seconds."""
    synchronize_if_needed(device)
    return time.perf_counter()


def elapsed_ms(start_seconds: float, end_seconds: float) -> float:
    """Convert elapsed seconds to milliseconds."""
    return (end_seconds - start_seconds) * 1000.0


class TimedBlock:
    def __init__(self) -> None:
        self.elapsed_ms = 0.0


@contextmanager
def measure_ms(device: Optional[torch.device] = None) -> Iterator[TimedBlock]:
    """Measure a code block in milliseconds, synchronizing CUDA if needed."""
    block = TimedBlock()
    start = now_seconds(device)
    try:
        yield block
    finally:
        end = now_seconds(device)
        block.elapsed_ms = elapsed_ms(start, end)
