import torch
import torch.nn as nn
from .config import ModelConfig

class TokenEmbedding(nn.Module):
    def __init__(self,config:ModelConfig) :
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_ids = None) :
        #token embedding
        token_embeds = self.token_embedding(x)

        #if position_ids is not provided, create it based on the input sequence length
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)

        #positional embedding
        position_embeds = self.position_embedding(position_ids)

        #add positional and token embeddings
        embeddings = token_embeds + position_embeds
        
        #apply dropuout
        embeddings = self.dropout(embeddings)
        return embeddings