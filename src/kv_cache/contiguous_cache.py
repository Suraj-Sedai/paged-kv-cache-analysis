import torch

class ContiguousKVCache:
    def __init__(self, config, batch_size, max_seq_len, device=None, dtype=torch.float32):
        """
        Initialize the KV cache with preallocated memory.
        """
        self.num_layers = config.n_layers  # Number of layers (L)
        self.batch_size = batch_size  # Batch size (B)
        self.num_heads = config.n_heads  # Number of heads (H)
        self.max_seq_len = max_seq_len  # Maximum sequence length (MaxSeq)
        self.head_dim = config.head_dim  # Head dimension (Hd)
        self.device = device  # Device (e.g., 'cuda' or 'cpu')
        self.dtype = dtype  # Data type (e.g., torch.float32)
        self.current_seq_len = 0  # Track the current sequence length in the cache
        # Preallocate memory for keys and values
        self.keys = torch.zeros(
            config.n_layers,
            batch_size,
            config.n_heads,
            max_seq_len,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        self.values= torch.zeros(
            config.n_layers,
            batch_size,
            config.n_heads,
            max_seq_len,
            config.head_dim,
            device=device,
            dtype=dtype,
        )

        # Initialize sequence length
        self.current_seq_len = 0

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        expected_prefix = (self.batch_size, self.num_heads, self.head_dim)

        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("k and v must have shape [B, H, T, Hd]")

        if k.shape != v.shape:
            raise ValueError("k and v must have matching shapes")

        if (k.shape[0], k.shape[1], k.shape[3]) != expected_prefix:
            raise ValueError("k and v must have shape [B, H, T, Hd]")

        T = k.size(2)

        if start_pos < 0 or start_pos + T > self.max_seq_len:
            raise IndexError("KV cache write exceeds max_seq_len")

        self.keys[layer_idx, :, :, start_pos:start_pos + T, :] = k
        self.values[layer_idx, :, :, start_pos:start_pos + T, :] = v

    def read(self, layer_idx: int, upto_pos: int):
        if upto_pos > self.max_seq_len:
            raise IndexError("Requested position exceeds max_seq_len")
        if upto_pos < 0:
            raise IndexError("Requested position must be non-negative")

        return (
            self.keys[layer_idx, :, :, :upto_pos, :],
            self.values[layer_idx, :, :, :upto_pos, :],
        )

    def advance(self, amount: int) -> None:
        if self.current_seq_len + amount > self.max_seq_len:
          raise IndexError("KV cache length exceeds max_seq_len")
        self.current_seq_len += amount

    def reset(self) -> None:
        """
        Reset the cache by clearing all data.
        """
        self.current_seq_len = 0
        self.keys.zero_()
        self.values.zero_()

    @property
    def memory_bytes(self):
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()
        