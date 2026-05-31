from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 512
    n_layers: int = 4
    n_heads : int = 4
    d_model : int = 256
    d_ff : int = 1024
    dropout :float = 0.0
    bias : bool = True

    def __post_init__(self):
        assert self.d_model %self.n_heads == 0

    @property
    def head_dim(self) -> int:
        return self.d_model //self.n_heads