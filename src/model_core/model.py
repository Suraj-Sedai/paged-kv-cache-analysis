import torch
import torch.nn as nn

from .embeddings import TokenEmbedding
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None, use_cache=False, start_pos=0):
        if use_cache and kv_cache is None:
            raise ValueError("kv_cache is required when use_cache=True")

        B, T = input_ids.shape

        position_ids = torch.arange(
            start_pos, start_pos + T, device=input_ids.device,
        ).unsqueeze(0).expand(B, T)
        x = self.embedding(input_ids, position_ids=position_ids)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x, kv_cache=kv_cache, layer_idx=layer_idx,
                start_pos=start_pos, use_cache=use_cache,
            )

        return self.lm_head(x)