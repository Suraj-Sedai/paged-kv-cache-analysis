from importlib import import_module

from src.profiling.memory import current_memory_mb, peak_memory_mb, reset_peak_memory
from src.profiling.metrics import GenerationMetrics, build_generation_metrics, percentile
from src.profiling.timer import elapsed_ms, measure_ms, now_seconds, synchronize_if_needed

_BENCHMARK_EXPORTS = {
    "BenchmarkConfig",
    "BenchmarkResult",
    "HardwareInfo",
    "aggregate_generation_metrics",
    "benchmark_generation",
    "collect_hardware_info",
    "estimate_kv_cache_memory_bytes",
    "select_device",
}

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "GenerationMetrics",
    "HardwareInfo",
    "aggregate_generation_metrics",
    "benchmark_generation",
    "build_generation_metrics",
    "collect_hardware_info",
    "current_memory_mb",
    "elapsed_ms",
    "estimate_kv_cache_memory_bytes",
    "measure_ms",
    "now_seconds",
    "peak_memory_mb",
    "percentile",
    "reset_peak_memory",
    "select_device",
    "synchronize_if_needed",
]


def __getattr__(name):
    if name in _BENCHMARK_EXPORTS:
        benchmark = import_module("src.profiling.benchmark")
        value = getattr(benchmark, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
