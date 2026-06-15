"""INT8 cache correctness gate (Exp 3).

`test_cache_equivalence` cannot apply here: quantization is lossy, so INT8 will
never reproduce FP16 tokens bit-for-bit. This is its replacement at the cache
level — a *bounded-error* gate instead of an exact-equality gate:

  1. round-trip reconstruction error stays under a stated bound (per-token
     symmetric int8 should give rel-L2 < ~2% / cosine > 0.999 on normal data),
  2. paging behaves exactly like PagedKVCache (shapes, prefill+decode, reset),
  3. the int8 footprint is ~0.5x the fp16 cache (the H3 memory lever),
  4. degenerate inputs (all-zero tokens) don't produce NaNs.

The *end-to-end* quality gate (logit drift / perplexity) lives with the
precision sweep, not here.
"""
import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache

N_LAYERS  = 2
N_HEADS   = 2
D_MODEL   = 64          # head_dim = 32 — realistic vector width for quant error
PAGE_SIZE = 4
NUM_PAGES = 16
SEQ_ID    = 0

# Error bounds for per-token symmetric int8 on ~N(0,1) data. Derivation: quant
# noise ~ uniform(-s/2, s/2), s = amax/127, giving rel-L2 ~ 0.7% — bounds are set
# a few x above that to stay robust without being vacuous.
REL_L2_MEAN_MAX = 0.02
REL_L2_MAX      = 0.05
COSINE_MIN      = 0.999


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


def make_int8(cfg):
    return INT8KVCache(
        n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
        block_size=PAGE_SIZE, num_blocks=NUM_PAGES,
        device="cpu", compute_dtype=torch.float16,
    )


def make_paged_fp16(cfg):
    return PagedKVCache(
        n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
        block_size=PAGE_SIZE, num_blocks=NUM_PAGES,
        device="cpu", dtype=torch.float16,
    )


def write_seq(cache, T, head_dim, seq_id=SEQ_ID, dtype=torch.float16):
    """Write T tokens across all layers (fp16, as the half() model produces)."""
    kvs = [(torch.randn(N_HEADS, T, head_dim, dtype=dtype),
            torch.randn(N_HEADS, T, head_dim, dtype=dtype))
           for _ in range(N_LAYERS)]
    for layer_idx, (k, v) in enumerate(kvs):
        cache.write(layer_idx, seq_id, k, v)
    cache.advance(seq_id, T)
    return kvs


def rel_l2(orig, recon):
    return (torch.norm((orig - recon).float(), dim=-1)
            / torch.norm(orig.float(), dim=-1).clamp(min=1e-8))   # per-token [H, T]


def cosine(orig, recon):
    return torch.nn.functional.cosine_similarity(
        orig.float(), recon.float(), dim=-1)                       # per-token [H, T]


# --- 1. bounded reconstruction error (the gate) ---

def test_roundtrip_error_bounded(cfg):
    cache = make_int8(cfg)
    kvs = write_seq(cache, T=10, head_dim=cfg.head_dim)
    for layer_idx, (k_orig, v_orig) in enumerate(kvs):
        k_r, v_r = cache.read(layer_idx, SEQ_ID)
        for orig, recon, name in [(k_orig, k_r, "K"), (v_orig, v_r, "V")]:
            err = rel_l2(orig, recon)
            cos = cosine(orig, recon)
            assert err.mean() < REL_L2_MEAN_MAX, f"{name} L{layer_idx} mean rel-L2 {err.mean():.4f}"
            assert err.max()  < REL_L2_MAX,      f"{name} L{layer_idx} max rel-L2 {err.max():.4f}"
            assert cos.min()  > COSINE_MIN,      f"{name} L{layer_idx} min cosine {cos.min():.5f}"


def test_zero_tokens_no_nan(cfg):
    cache = make_int8(cfg)
    z = torch.zeros(N_HEADS, 4, cfg.head_dim, dtype=torch.float16)
    cache.write(0, SEQ_ID, z, z)
    cache.advance(SEQ_ID, 4)
    k_r, v_r = cache.read(0, SEQ_ID)
    assert torch.isfinite(k_r).all() and torch.isfinite(v_r).all()
    assert torch.count_nonzero(k_r) == 0 and torch.count_nonzero(v_r) == 0


# --- 2. paging parity with PagedKVCache (shapes / prefill+decode / reset) ---

def test_read_shape_matches_paged(cfg):
    i8, fp = make_int8(cfg), make_paged_fp16(cfg)
    write_seq(i8, T=7, head_dim=cfg.head_dim)
    write_seq(fp, T=7, head_dim=cfg.head_dim)
    for layer_idx in range(N_LAYERS):
        k_i, v_i = i8.read(layer_idx, SEQ_ID)
        k_f, _   = fp.read(layer_idx, SEQ_ID)
        assert k_i.shape == k_f.shape == (N_HEADS, 7, cfg.head_dim)
        assert k_i.dtype == torch.float16


def test_prefill_then_decode(cfg):
    cache = make_int8(cfg)
    write_seq(cache, T=6, head_dim=cfg.head_dim)
    write_seq(cache, T=1, head_dim=cfg.head_dim)   # one decode step
    k_r, _ = cache.read(0, SEQ_ID)
    assert k_r.shape[1] == 7                        # spans a page boundary (page_size=4)


def test_reset_and_free(cfg):
    cache = make_int8(cfg)
    write_seq(cache, T=8, head_dim=cfg.head_dim)
    cache.reset()
    assert len(cache.page_table) == 0 and len(cache.free_pages) == NUM_PAGES
    write_seq(cache, T=4, head_dim=cfg.head_dim)
    cache.free(SEQ_ID)
    assert SEQ_ID not in cache.page_table


# --- 3. the H3 memory lever: int8 footprint ~0.5x fp16 ---

def test_memory_is_roughly_half_fp16(cfg):
    i8, fp = make_int8(cfg), make_paged_fp16(cfg)
    write_seq(i8, T=12, head_dim=cfg.head_dim)
    write_seq(fp, T=12, head_dim=cfg.head_dim)
    ratio = i8.memory_bytes() / fp.memory_bytes()
    # int8 data is 0.5x; fp16 scales add 2/head_dim. Should sit just above 0.5.
    expected = 0.5 + 2.0 / cfg.head_dim
    assert 0.5 < ratio < expected + 0.02, f"int8/fp16 memory ratio {ratio:.3f}, expected ~{expected:.3f}"
