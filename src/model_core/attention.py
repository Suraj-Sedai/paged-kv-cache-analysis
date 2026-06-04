import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_core.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
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

    def forward(
        self,
        x,
        kv_cache=None,
        layer_idx=None,
        start_pos=0,
        use_cache=False,
    ):
        if use_cache and (kv_cache is None or layer_idx is None):
            raise ValueError("kv_cache and layer_idx are required when use_cache=True")

        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]

        q = q / (Hd**0.5)

        if use_cache:
            out_k, out_v = [], []
            for b in range(B):
                kv_cache.write(layer_idx, b, k[b], v[b])   # k[b]: [H, T, Hd], seq_id = b
                kb, vb = kv_cache.read(layer_idx, b)        # [H, n, Hd]
                out_k.append(kb)
                out_v.append(vb)
            k = torch.stack(out_k, dim=0)                   # [B, H, n, Hd]
            v = torch.stack(out_v, dim=0)

            if T > 1:
                key_len = k.size(2)
                query_positions = torch.arange(start_pos, start_pos + T, device=x.device)
                key_positions = torch.arange(key_len, device=x.device)
                causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).view(1, 1, T, key_len)
            else:
                causal_mask = None
        else:
            key_len = T
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            causal_mask = causal_mask.view(1, 1, T, T)

        attn_weights = q @ k.transpose(-2, -1)

        if causal_mask is not None:
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.out_proj(attn_output)
