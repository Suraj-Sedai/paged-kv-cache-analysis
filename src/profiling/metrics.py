from dataclasses import dataclass
from typing import Sequence


@dataclass
class GenerationMetrics:
    prefill_latency_ms: float
    decode_latency_ms: float
    ttft_ms: float
    tpot_avg_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float
    peak_memory_mb: float


def build_generation_metrics(
    *,
    prefill_latency_ms: float,
    first_sampling_latency_ms: float,
    decode_step_times_ms: Sequence[float],
    total_latency_ms: float,
    generated_tokens: int,
    peak_memory_mb: float,
) -> GenerationMetrics:
    """Build centralized generation metrics from measured latencies."""
    decode_latency_ms = float(sum(decode_step_times_ms))
    ttft_ms = prefill_latency_ms + first_sampling_latency_ms
    return GenerationMetrics(
        prefill_latency_ms=prefill_latency_ms,
        decode_latency_ms=decode_latency_ms,
        ttft_ms=ttft_ms,
        tpot_avg_ms=_average(decode_step_times_ms),
        tpot_p50_ms=percentile(decode_step_times_ms, 50),
        tpot_p95_ms=percentile(decode_step_times_ms, 95),
        tpot_p99_ms=percentile(decode_step_times_ms, 99),
        total_latency_ms=total_latency_ms,
        throughput_tokens_per_sec=_throughput(generated_tokens, total_latency_ms),
        peak_memory_mb=peak_memory_mb,
    )


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile for a sequence of timings."""
    if not values:
        return 0.0
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percent / 100.0) * (len(sorted_values) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = rank - lower_idx
    return sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _throughput(generated_tokens: int, total_latency_ms: float) -> float:
    if generated_tokens <= 0 or total_latency_ms <= 0:
        return 0.0
    return generated_tokens / (total_latency_ms / 1000.0)
