from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import torch

from src.inference.controller import GenerationResult, InferenceController
from src.inference.sampling import SamplingConfig
from src.model_core.config import ModelConfig
from src.profiling.memory import current_memory_mb
from src.profiling.metrics import GenerationMetrics


@dataclass
class HardwareInfo:
    device_type: str
    device_name: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    total_memory_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    batch_size: int = 1
    prompt_len: int = 16
    max_new_tokens: int = 16
    warmup_runs: int = 1
    measured_runs: int = 3
    device: Optional[str] = None
    seed: int = 0
    sampling_config: Optional[SamplingConfig] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    hardware: HardwareInfo
    run_metrics: list[GenerationMetrics]
    mean_metrics: GenerationMetrics
    num_prompt_tokens: int
    num_generated_tokens: int
    kv_cache_memory_bytes: int
    current_memory_mb: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["run_metrics"] = [asdict(metrics) for metrics in self.run_metrics]
        result["mean_metrics"] = asdict(self.mean_metrics)
        result["hardware"] = self.hardware.to_dict()
        return result


def select_device(device: Optional[str] = None) -> torch.device:
    """Resolve an explicit device or choose CUDA when it is available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def collect_hardware_info(device: Optional[Union[torch.device, str]] = None) -> HardwareInfo:
    """Collect stable hardware metadata for benchmark reports."""
    resolved = select_device(str(device) if device is not None else None)
    total_memory_mb = 0.0

    if resolved.type == "cuda":
        props = torch.cuda.get_device_properties(resolved)
        device_name = props.name
        total_memory_mb = props.total_memory / (1024.0 * 1024.0)
    else:
        device_name = "cpu"

    return HardwareInfo(
        device_type=resolved.type,
        device_name=device_name,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        total_memory_mb=float(total_memory_mb),
    )


def estimate_kv_cache_memory_bytes(
    config: ModelConfig,
    batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Estimate bytes needed by the contiguous key/value cache."""
    element_size = torch.empty((), dtype=dtype).element_size()
    num_elements = (
        2
        * config.n_layers
        * batch_size
        * config.n_heads
        * max_seq_len
        * config.head_dim
    )
    return int(num_elements * element_size)


def aggregate_generation_metrics(metrics: Sequence[GenerationMetrics]) -> GenerationMetrics:
    """Average per-run generation metrics into a single summary."""
    if not metrics:
        raise ValueError("metrics must contain at least one run")

    return GenerationMetrics(
        prefill_latency_ms=_mean([item.prefill_latency_ms for item in metrics]),
        decode_latency_ms=_mean([item.decode_latency_ms for item in metrics]),
        ttft_ms=_mean([item.ttft_ms for item in metrics]),
        tpot_avg_ms=_mean([item.tpot_avg_ms for item in metrics]),
        tpot_p50_ms=_mean([item.tpot_p50_ms for item in metrics]),
        tpot_p95_ms=_mean([item.tpot_p95_ms for item in metrics]),
        tpot_p99_ms=_mean([item.tpot_p99_ms for item in metrics]),
        total_latency_ms=_mean([item.total_latency_ms for item in metrics]),
        throughput_tokens_per_sec=_mean([item.throughput_tokens_per_sec for item in metrics]),
        peak_memory_mb=_mean([item.peak_memory_mb for item in metrics]),
    )


def benchmark_generation(
    model: torch.nn.Module,
    model_config: ModelConfig,
    benchmark_config: Optional[BenchmarkConfig] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> BenchmarkResult:
    """Run warmup and measured generation passes and return aggregated metrics."""
    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()
    _validate_benchmark_config(benchmark_config, model_config)

    device = select_device(benchmark_config.device)
    model = model.to(device)
    prompt = _build_prompt(input_ids, model_config, benchmark_config, device)
    controller = InferenceController(model, model_config)

    for _ in range(benchmark_config.warmup_runs):
        controller.generate(
            prompt,
            max_new_tokens=benchmark_config.max_new_tokens,
            eos_token_id=benchmark_config.eos_token_id,
            sampling_config=benchmark_config.sampling_config,
            use_cache=benchmark_config.use_cache,
        )

    run_results: list[GenerationResult] = []
    for _ in range(benchmark_config.measured_runs):
        run_results.append(
            controller.generate(
                prompt,
                max_new_tokens=benchmark_config.max_new_tokens,
                eos_token_id=benchmark_config.eos_token_id,
                sampling_config=benchmark_config.sampling_config,
                use_cache=benchmark_config.use_cache,
            )
        )

    run_metrics = [result.metrics for result in run_results]
    dtype = next(model.parameters()).dtype
    return BenchmarkResult(
        config=benchmark_config,
        hardware=collect_hardware_info(device),
        run_metrics=run_metrics,
        mean_metrics=aggregate_generation_metrics(run_metrics),
        num_prompt_tokens=run_results[-1].num_prompt_tokens,
        num_generated_tokens=run_results[-1].num_generated_tokens,
        kv_cache_memory_bytes=estimate_kv_cache_memory_bytes(
            model_config,
            batch_size=benchmark_config.batch_size,
            max_seq_len=benchmark_config.prompt_len + benchmark_config.max_new_tokens,
            dtype=dtype,
        ),
        current_memory_mb=current_memory_mb(device),
    )


def _build_prompt(
    input_ids: Optional[torch.Tensor],
    model_config: ModelConfig,
    benchmark_config: BenchmarkConfig,
    device: torch.device,
) -> torch.Tensor:
    if input_ids is not None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if input_ids.size(0) != benchmark_config.batch_size:
            raise ValueError("input_ids batch size must match benchmark_config.batch_size")
        if input_ids.size(1) != benchmark_config.prompt_len:
            raise ValueError("input_ids length must match benchmark_config.prompt_len")
        return input_ids.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(benchmark_config.seed)
    return torch.randint(
        low=0,
        high=model_config.vocab_size,
        size=(benchmark_config.batch_size, benchmark_config.prompt_len),
        device=device,
        generator=generator,
    )


def _validate_benchmark_config(
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
) -> None:
    if benchmark_config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if benchmark_config.prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if benchmark_config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if benchmark_config.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if benchmark_config.measured_runs <= 0:
        raise ValueError("measured_runs must be positive")
    if benchmark_config.prompt_len + benchmark_config.max_new_tokens > model_config.max_seq_len:
        raise ValueError("benchmark sequence length exceeds model max_seq_len")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / len(values)

