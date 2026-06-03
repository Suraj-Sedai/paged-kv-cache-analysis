from dataclasses import dataclass
from typing import Optional

import torch

from src.kv_cache.base import BaseKVCache
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.inference.sampling import SamplingConfig, sample_next_token
from src.model_core.config import ModelConfig
from src.profiling.memory import peak_memory_mb, reset_peak_memory
from src.profiling.metrics import GenerationMetrics, build_generation_metrics
from src.profiling.timer import elapsed_ms, measure_ms, now_seconds
from src.kv_cache.base import BaseKVCache


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    num_prompt_tokens: int
    num_generated_tokens: int
    metrics: GenerationMetrics

class InferenceController:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        sampling_config: Optional[SamplingConfig] = None,
        use_cache: bool = True,
        cache:BaseKVCache = None,
    ) -> GenerationResult:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        device = input_ids.device
        reset_peak_memory(device)
        generation_start = now_seconds(device)

        batch_size, prompt_len = input_ids.shape
        total_len = prompt_len + max_new_tokens
        if total_len > self.config.max_seq_len:
            raise ValueError("Requested generation exceeds model max_seq_len")

        output = torch.empty(
            (batch_size, total_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output[:, :prompt_len] = input_ids

        if max_new_tokens == 0:
            generation_end = now_seconds(device)
            metrics = build_generation_metrics(
                prefill_latency_ms=0.0,
                first_sampling_latency_ms=0.0,
                decode_step_times_ms=[],
                total_latency_ms=elapsed_ms(generation_start, generation_end),
                generated_tokens=0,
                peak_memory_mb=peak_memory_mb(device),
            )
            return GenerationResult(
                token_ids=output[:, :prompt_len],
                num_prompt_tokens=prompt_len,
                num_generated_tokens=0,
                metrics=metrics,
            )

        # cache = None
        if use_cache:
            cache_dtype = next(self.model.parameters()).dtype
            cache = ContiguousKVCache(
                config=self.config,
                batch_size=batch_size,
                max_seq_len=total_len,
                device=device,
                dtype=cache_dtype,
            )

        self.model.eval()
        decode_step_times_ms = []
        with torch.no_grad():
            with measure_ms(device) as prefill_timer:
                logits = self.model(input_ids, kv_cache=cache, use_cache=use_cache)

            with measure_ms(device) as first_sample_timer:
                next_token = sample_next_token(
                    logits[:, -1, :],
                    generated_tokens=output[:, :prompt_len],
                    config=sampling_config,
                )
            decode_step_times_ms.append(first_sample_timer.elapsed_ms)

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            actual_len = prompt_len

            for step in range(max_new_tokens):
                position = prompt_len + step
                output[:, position] = next_token.squeeze(-1)
                actual_len = position + 1

                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                    if torch.all(finished):
                        break

                if step == max_new_tokens - 1:
                    break

                with measure_ms(device) as decode_timer:
                    if use_cache:
                        logits = self.model(next_token, kv_cache=cache, use_cache=True)
                    else:
                        logits = self.model(output[:, :actual_len], use_cache=False)
                    next_token = sample_next_token(
                        logits[:, -1, :],
                        generated_tokens=output[:, :actual_len],
                        config=sampling_config,
                    )
                decode_step_times_ms.append(decode_timer.elapsed_ms)

        generation_end = now_seconds(device)
        num_generated_tokens = actual_len - prompt_len
        metrics = build_generation_metrics(
            prefill_latency_ms=prefill_timer.elapsed_ms,
            first_sampling_latency_ms=first_sample_timer.elapsed_ms,
            decode_step_times_ms=decode_step_times_ms,
            total_latency_ms=elapsed_ms(generation_start, generation_end),
            generated_tokens=batch_size * num_generated_tokens,
            peak_memory_mb=peak_memory_mb(device),
        )
        return GenerationResult(
            token_ids=output[:, :actual_len],
            num_prompt_tokens=prompt_len,
            num_generated_tokens=num_generated_tokens,
            metrics=metrics,
        )
