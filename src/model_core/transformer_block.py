import torch.nn as nn

from src.model_core.config import ModelConfig
from .attention import MultiHeadSelfAttention
from .mlp import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.d_model
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.attention = MultiHeadSelfAttention(config)
        self.mlp = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x,
        kv_cache=None,
        layer_idx=None,
        start_pos=0,
        use_cache=False,
    ):
        residual = x
        x = self.ln1(x)
        x = self.attention(
            x,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            use_cache=use_cache,
        )
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual

        return x
