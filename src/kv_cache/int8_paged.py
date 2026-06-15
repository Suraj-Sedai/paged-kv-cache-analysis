"""INT8 paged KV cache — per-token symmetric quantization (Exp 3, precision axis).

Identical paging to PagedKVCache; only the *storage precision* changes. K/V are
stored as int8 with one fp16 scale per token vector (per layer, per head, per
position), and dequantized to ``compute_dtype`` on read. Weights stay FP16 — only
the cache is quantized (cf. KIVI / SmoothQuant; CLAUDE.md §6).

Scheme: symmetric per-token. For each token's head_dim vector, scale = amax/127
(zero-point free). The quantize math runs in fp32 to keep the amax/round step off
the fp16 underflow floor; codes are int8, scales are stored fp16.

Memory (the H3 lever): int8 data is 0.5x the fp16 cache, plus a scale tax of
2 bytes per token-head = 2/head_dim of the data (~3% at head_dim 64). The net
footprint is (1 + 2/head_dim)/2 = 0.516x at head_dim 64 — the headroom that should
push the OOM frontier outward (but on an RTX 5070 Ti Laptop the cache is not the
memory bottleneck, so the frontier does not move; see STATUS.md).

Drop-in for PagedKVCache except the trailing arg is ``compute_dtype`` (the dtype
read() returns / the model computes in) rather than storage ``dtype``.
"""
import torch

from src.kv_cache.paged import PagedKVCache


class INT8KVCache(PagedKVCache):
    def __init__(self, n_layers, n_heads, dim_head, block_size, num_blocks,
                 device, compute_dtype=torch.float16):
        # Page pool is int8; paging bookkeeping is inherited unchanged.
        super().__init__(n_layers, n_heads, dim_head, block_size, num_blocks,
                         device, dtype=torch.int8)
        self.compute_dtype = compute_dtype
        # One scale per (block, layer, head, slot) — fp16 keeps the scale tax small.
        scale_shape = (num_blocks, n_layers, n_heads, block_size)
        self.k_scales = torch.zeros(scale_shape, device=device, dtype=torch.float16)
        self.v_scales = torch.zeros(scale_shape, device=device, dtype=torch.float16)

    @staticmethod
    def _quantize(x):
        """x: [H, T, Hd] -> (int8 codes [H, T, Hd], fp16 scales [H, T]).

        Symmetric per-token: scale = amax(|x|)/127 over head_dim. fp32 internally
        so the divide doesn't underflow in fp16; all-zero tokens -> scale 0, codes 0.
        """
        xf = x.float()
        amax = xf.abs().amax(dim=-1)                          # [H, T]
        scale = (amax / 127.0).clamp(min=1e-8)               # guard div-by-zero
        codes = torch.round(xf / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return codes, scale.to(torch.float16)

    def write(self, layer_idx, seq_id, k, v):
        t_new = k.shape[1]
        start = self.seq_len.get(seq_id, 0)
        self._ensure_capacity(seq_id, start + t_new)
        kc, ks = self._quantize(k)
        vc, vs = self._quantize(v)
        pages, written = self.page_table[seq_id], 0
        while written < t_new:
            pos = start + written
            lp, off = pos // self.page_size, pos % self.page_size
            chunk = min(self.page_size - off, t_new - written)
            phys = pages[lp]
            sl = slice(written, written + chunk)
            self.k_pages[phys, layer_idx, :, off:off+chunk, :].copy_(kc[:, sl, :])
            self.v_pages[phys, layer_idx, :, off:off+chunk, :].copy_(vc[:, sl, :])
            self.k_scales[phys, layer_idx, :, off:off+chunk].copy_(ks[:, sl])
            self.v_scales[phys, layer_idx, :, off:off+chunk].copy_(vs[:, sl])
            written += chunk
        self.filled[seq_id] = start + t_new

    def read(self, layer_idx, seq_id):
        """Gather physical pages and dequantize to compute_dtype: [H, n, Hd]."""
        n = self.filled[seq_id]                              # KeyError if never written / freed
        pages = self.page_table[seq_id]
        need = (n + self.page_size - 1) // self.page_size
        phys = pages[:need]
        kc = self.k_pages[phys, layer_idx]                   # [P, H, ps, Hd] int8
        vc = self.v_pages[phys, layer_idx]
        ks = self.k_scales[phys, layer_idx].to(self.compute_dtype)   # [P, H, ps]
        vs = self.v_scales[phys, layer_idx].to(self.compute_dtype)
        k = kc.to(self.compute_dtype) * ks.unsqueeze(-1)
        v = vc.to(self.compute_dtype) * vs.unsqueeze(-1)
        P, H, ps, Hd = kc.shape
        k = k.permute(1, 0, 2, 3).reshape(H, P * ps, Hd)[:, :n, :]
        v = v.permute(1, 0, 2, 3).reshape(H, P * ps, Hd)[:, :n, :]
        return k, v

    def memory_bytes(self):
        used = sum(len(p) for p in self.page_table.values())   # physical pages held now
        data = self.k_pages[0].numel() * self.k_pages.element_size()     # int8 -> 1B
        scale = self.k_scales[0].numel() * self.k_scales.element_size()  # fp16 -> 2B
        return 2 * used * (data + scale)                       # k + v, data + scales
