"""INT8 paged KV cache — per-token symmetric quantization (precision axis).

Identical paging to PagedKVCache; only storage precision changes. K/V are stored
as int8 with one fp16 scale per (token, head), dequantized to compute_dtype on
read. Weights stay FP16 — cache only (cf. KIVI / SmoothQuant).

Memory: int8 data is 0.5x the fp16 cache plus a 2-bytes-per-token-head scale tax
= 2/head_dim of the data. Net (1 + 2/head_dim)/2 = 0.516x at head_dim 64.

LATENCY CAVEAT: read() dequantizes the full history every decode step, inherited
from PagedKVCache's materialising read. A fused dequant-in-attention path removes
most of it. Report as harness artifact, not intrinsic INT8 cost.
"""
import torch

from src.kv_cache.paged import PagedKVCache


class INT8KVCache(PagedKVCache):
    def __init__(self, n_layers, n_heads, dim_head, block_size, num_blocks,
                 batch_size, device, compute_dtype=torch.float16):
        super().__init__(n_layers, n_heads, dim_head, block_size, num_blocks,
                         batch_size, device, dtype=torch.int8)
        self.compute_dtype = compute_dtype
        scale_shape = (num_blocks, n_layers, n_heads, block_size)
        self.k_scales = torch.zeros(scale_shape, device=device, dtype=torch.float16)
        self.v_scales = torch.zeros(scale_shape, device=device, dtype=torch.float16)

    @staticmethod
    def _quantize(x):
        """x: [B, H, T, Hd] -> (int8 codes [B, H, T, Hd], fp16 scales [B, H, T]).

        Symmetric per-token: scale = amax(|x|)/127 over head_dim. fp32 internally
        so the divide does not underflow in fp16.
        """
        xf = x.float()
        amax = xf.abs().amax(dim=-1)                    # [B, H, T]
        scale = (amax / 127.0).clamp(min=1e-8)
        codes = torch.round(xf / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return codes, scale.to(torch.float16)

    def write(self, layer_idx, k, v):
        T = k.shape[2]
        start = self.seq_len
        self._ensure_capacity(start + T)
        kc, ks = self._quantize(k)
        vc, vs = self._quantize(v)
        phys, off = self._index(T, start)
        self.k_pages[phys, layer_idx, :, off, :] = kc.permute(0, 2, 1, 3)
        self.v_pages[phys, layer_idx, :, off, :] = vc.permute(0, 2, 1, 3)
        self.k_scales[phys, layer_idx, :, off] = ks.permute(0, 2, 1)
        self.v_scales[phys, layer_idx, :, off] = vs.permute(0, 2, 1)
        self.filled = start + T

    def read(self, layer_idx):
        n = self.filled
        phys, off = self._index(n)
        kc = self.k_pages[phys, layer_idx, :, off, :].to(self.compute_dtype)   # [B, n, H, Hd]
        vc = self.v_pages[phys, layer_idx, :, off, :].to(self.compute_dtype)
        ks = self.k_scales[phys, layer_idx, :, off].to(self.compute_dtype)     # [B, n, H]
        vs = self.v_scales[phys, layer_idx, :, off].to(self.compute_dtype)
        k = (kc * ks.unsqueeze(-1)).permute(0, 2, 1, 3)
        v = (vc * vs.unsqueeze(-1)).permute(0, 2, 1, 3)
        return k, v

    def memory_bytes(self):
        used = self.n_pages * self.batch_size
        data = self.k_pages[0].numel() * self.k_pages.element_size()     # int8 -> 1B
        scale = self.k_scales[0].numel() * self.k_scales.element_size()  # fp16 -> 2B
        return 2 * used * (data + scale)
