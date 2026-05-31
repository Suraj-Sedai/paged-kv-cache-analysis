from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    greedy: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax token IDs for logits shaped [B, vocab_size]."""
    _validate_logits(logits)
    return torch.argmax(logits, dim=-1, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits shaped [B, vocab_size] by a positive temperature."""
    _validate_logits(logits)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """Mask logits outside the top k positions for each row to -inf."""
    _validate_logits(logits)
    if k is None:
        return logits
    if k <= 0:
        raise ValueError("top_k must be positive")

    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    top_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """Apply nucleus filtering to logits shaped [B, vocab_size]."""
    _validate_logits(logits)
    if p is None or p >= 1.0:
        return logits
    if p <= 0:
        raise ValueError("top_p must be positive")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove_mask = cumulative_probs > p
    sorted_remove_mask[:, 1:] = sorted_remove_mask[:, :-1].clone()
    sorted_remove_mask[:, 0] = False

    remove_mask = torch.zeros_like(sorted_remove_mask)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Optional[torch.Tensor],
    penalty: float,
) -> torch.Tensor:
    """Penalize previously generated token IDs in logits shaped [B, vocab_size]."""
    _validate_logits(logits)
    if generated_tokens is None or penalty == 1.0:
        return logits
    if generated_tokens.ndim != 2:
        raise ValueError("generated_tokens must have shape [B, T]")
    if generated_tokens.size(0) != logits.size(0):
        raise ValueError("generated_tokens batch size must match logits")
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")

    adjusted = logits.clone()
    vocab_size = logits.size(-1)
    for batch_idx in range(logits.size(0)):
        token_ids = torch.unique(generated_tokens[batch_idx])
        token_ids = token_ids[(token_ids >= 0) & (token_ids < vocab_size)]
        token_logits = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            token_logits > 0,
            token_logits / penalty,
            token_logits * penalty,
        )
    return adjusted


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: Optional[torch.Tensor] = None,
    config: Optional[SamplingConfig] = None,
) -> torch.Tensor:
    """Sample next token IDs from logits shaped [B, vocab_size]."""
    _validate_logits(logits)
    if config is None:
        config = SamplingConfig()
    if config.greedy:
        return greedy_sample(logits)

    processed_logits = apply_repetition_penalty(
        logits,
        generated_tokens=generated_tokens,
        penalty=config.repetition_penalty,
    )
    processed_logits = apply_temperature(processed_logits, config.temperature)
    processed_logits = apply_top_k(processed_logits, config.top_k)
    processed_logits = apply_top_p(processed_logits, config.top_p)

    probabilities = F.softmax(processed_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, vocab_size]")
