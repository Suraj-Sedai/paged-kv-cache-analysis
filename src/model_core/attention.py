import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_core.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over a batched KV cache.

    Two deliberate choices, both load-bearing for the measurement:

    1. The cache is written/read once per layer for the whole batch. The v1
       harness looped `for b in range(B)`, costing B * n_layers cache ops per
       token, which put decode in a launch-bound regime and made every timing
       column noise. That loop is gone.
    2. Attention uses F.scaled_dot_product_attention rather than an explicit
       q @ k.T. The explicit form materialises a [B, H, T, T] score matrix —
       ~4.3 GB at seq 4096 / batch 16 / 8 heads in fp16 — which dominated peak
       memory and is why the KV cache looked like only ~10% of the footprint.
       SDPA makes the cache a real share of the memory ledger, which is the
       precondition for any claim about the OOM frontier.

    NOTE: SDPA applies its own 1/sqrt(head_dim) scaling. Do NOT pre-scale q.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, kv_cache=None, layer_idx=None, start_pos=0, use_cache=False):
        if use_cache and (kv_cache is None or layer_idx is None):
            raise ValueError("kv_cache and layer_idx are required when use_cache=True")

        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)   # [B, H, T, Hd]
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        if use_cache:
            kv_cache.write(layer_idx, k, v)
            k, v = kv_cache.read(layer_idx)                    # [B, H, n, Hd]

        key_len = k.size(2)
        if T > 1:
            # is_causal aligns the mask top-left, which is only correct when the
            # query block covers the whole key range (prefill from an empty
            # cache). Chunked prefill would need an explicit mask here.
            assert T == key_len, (
                f"is_causal requires T == key_len (got T={T}, key_len={key_len}); "
                "chunked prefill needs an explicit mask"
            )
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # single decode step attends to all cached keys — no mask needed
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)
