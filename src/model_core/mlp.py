import torch.nn as nn
import torch.nn.functional as F

from src.model_core.config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError("Unsupported activation. Use 'gelu' or 'silu'.")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)
