"""Equivalence gate: paged and INT8 caches must reproduce the contiguous cache.

This is the gate the whole study rests on — no benchmark number counts unless
the layouts are interchangeable. Precision is held constant (fp32) so this tests
LAYOUT, not dtype; INT8 is tested against a tolerance instead.
"""
import random

import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache

N_LAYERS, N_HEADS, D_MODEL = 2, 2, 16
PAGE_SIZE, NUM_PAGES = 4, 64
BATCH = 3


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


def make_contiguous(cfg, max_seq_len):
    return ContiguousKVCache(cfg, batch_size=BATCH, max_seq_len=max_seq_len,
                             device="cpu", dtype=torch.float32)


def make_paged(cfg, block_size=PAGE_SIZE, num_blocks=NUM_PAGES):
    return PagedKVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
                        block_size=block_size, num_blocks=num_blocks,
                        batch_size=BATCH, device="cpu", dtype=torch.float32)


def make_paged_fp16(cfg, block_size=PAGE_SIZE):
    return PagedKVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
                        block_size=block_size, num_blocks=NUM_PAGES,
                        batch_size=BATCH, device="cpu", dtype=torch.float16)


def make_int8(cfg, block_size=PAGE_SIZE):
    return INT8KVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
                       block_size=block_size, num_blocks=NUM_PAGES,
                       batch_size=BATCH, device="cpu", compute_dtype=torch.float32)


def rand_kv_all_layers(T, head_dim):
    return [(torch.randn(BATCH, N_HEADS, T, head_dim),
             torch.randn(BATCH, N_HEADS, T, head_dim)) for _ in range(N_LAYERS)]


def zero_kv(T, head_dim):
    """Values are irrelevant to fragmentation accounting; zeros keep the
    length-sweep tests (hundreds of writes) cheap."""
    z = torch.zeros(BATCH, N_HEADS, T, head_dim)
    return z, z


def forward_step(cache, kvs):
    """Mirrors model.py: write every layer, then advance once."""
    T = kvs[0][0].size(2)
    for layer_idx, (k, v) in enumerate(kvs):
        cache.write(layer_idx, k, v)
    cache.advance(T)


def assert_layers_match(a, b, atol=0.0, label=""):
    for layer_idx in range(N_LAYERS):
        k_a, v_a = a.read(layer_idx)
        k_b, v_b = b.read(layer_idx)
        assert k_a.shape == k_b.shape, f"{label} K shape mismatch at layer {layer_idx}"
        assert torch.allclose(k_a, k_b, atol=atol), f"{label} K mismatch at layer {layer_idx}"
        assert torch.allclose(v_a, v_b, atol=atol), f"{label} V mismatch at layer {layer_idx}"


# --- layout equivalence: paged must equal contiguous exactly ---

def test_prefill_matches(cfg):
    torch.manual_seed(0)
    c, p = make_contiguous(cfg, 32), make_paged(cfg)
    kvs = rand_kv_all_layers(6, cfg.head_dim)
    forward_step(c, kvs); forward_step(p, kvs)
    assert_layers_match(c, p, label="prefill")


def test_prefill_then_decode_matches(cfg):
    torch.manual_seed(1)
    c, p = make_contiguous(cfg, 32), make_paged(cfg)
    forward_step(c, (kvs := rand_kv_all_layers(6, cfg.head_dim))); forward_step(p, kvs)
    for _ in range(5):   # cross several page boundaries
        forward_step(c, (d := rand_kv_all_layers(1, cfg.head_dim))); forward_step(p, d)
    assert c.read(0)[0].shape[2] == 11
    assert_layers_match(c, p, label="decode")


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
def test_matches_across_block_sizes(cfg, block_size):
    torch.manual_seed(2)
    c, p = make_contiguous(cfg, 32), make_paged(cfg, block_size)
    forward_step(c, (kvs := rand_kv_all_layers(7, cfg.head_dim))); forward_step(p, kvs)
    for _ in range(4):
        forward_step(c, (d := rand_kv_all_layers(1, cfg.head_dim))); forward_step(p, d)
    assert_layers_match(c, p, label=f"block_size={block_size}")


def test_reset_then_reuse_matches(cfg):
    torch.manual_seed(3)
    c, p = make_contiguous(cfg, 32), make_paged(cfg)
    forward_step(c, (kvs := rand_kv_all_layers(4, cfg.head_dim))); forward_step(p, kvs)
    c.reset(); p.reset()
    forward_step(c, (kvs2 := rand_kv_all_layers(4, cfg.head_dim))); forward_step(p, kvs2)
    assert_layers_match(c, p, label="after reset")


def test_batch_elements_are_independent(cfg):
    """Regression guard for the batched rewrite: seq b must not read seq b'.

    The v1 per-sequence loop made this impossible by construction; the vectorised
    page-table gather makes it a real failure mode worth pinning.
    """
    torch.manual_seed(4)
    p = make_paged(cfg)
    k = torch.arange(BATCH, dtype=torch.float32).view(BATCH, 1, 1, 1).expand(
        BATCH, N_HEADS, 5, cfg.head_dim).contiguous()
    for layer_idx in range(N_LAYERS):
        p.write(layer_idx, k, k)
    p.advance(5)
    out, _ = p.read(0)
    for b in range(BATCH):
        assert torch.all(out[b] == b), f"sequence {b} contaminated by another sequence"


# --- INT8: same layout, lossy precision -> tolerance, not exactness ---

def test_int8_matches_contiguous_within_quant_error(cfg):
    torch.manual_seed(5)
    c, q = make_contiguous(cfg, 32), make_int8(cfg)
    forward_step(c, (kvs := rand_kv_all_layers(6, cfg.head_dim))); forward_step(q, kvs)
    for layer_idx in range(N_LAYERS):
        k_c, v_c = c.read(layer_idx)
        k_q, v_q = q.read(layer_idx)
        # Two error sources, both required:
        #   (a) rounding to the int8 grid: <= scale/2 = amax/254
        #   (b) the SCALE ITSELF is stored fp16, so a code near +-127 picks up
        #       another ~2^-11 relative error. Bounding only (a) fails.
        def tol(ref):
            amax = ref.abs().amax(dim=-1, keepdim=True)
            return amax / 254.0 + amax * 2 ** -10
        assert torch.all((k_c - k_q).abs() <= tol(k_c)), f"K quant error too large, layer {layer_idx}"
        assert torch.all((v_c - v_q).abs() <= tol(v_c)), f"V quant error too large, layer {layer_idx}"


def test_int8_memory_ratio_is_0516(cfg):
    """The H3 memory lever: (1 + 2/head_dim)/2 exactly.

    The ratio is defined against an FP16 baseline. Comparing against an FP32
    paged cache silently returns a different number -- pin the dtype.
    """
    p, q = make_paged_fp16(cfg), make_int8(cfg)
    kvs = rand_kv_all_layers(6, cfg.head_dim)
    forward_step(p, [(k.half(), v.half()) for k, v in kvs])
    forward_step(q, kvs)
    expected = (1 + 2 / cfg.head_dim) / 2
    assert q.memory_bytes() / p.memory_bytes() == pytest.approx(expected)


# --- fragmentation semantics ---

@pytest.mark.parametrize("seq_len,block_size,expected", [
    (16, 16, 0.0),        # exactly one page
    (17, 16, 15 / 32),    # was 15/16 under last-block (2 pages -> 2x overstated)
    (24, 16, 8 / 32),     # was 0.5
    (8, 8, 0.0),
    (9, 8, 7 / 16),       # was 7/8
])
def test_paged_fragmentation_is_waste_over_allocation(cfg, seq_len, block_size, expected):
    p = make_paged(cfg, block_size)
    k, v = rand_kv_all_layers(seq_len, cfg.head_dim)[0]
    p.write(0, k, v)
    assert p.fragmentation_ratio() == pytest.approx(expected)


def test_contiguous_fragmentation_is_waste_over_allocation(cfg):
    c = make_contiguous(cfg, 100)
    k, v = rand_kv_all_layers(60, cfg.head_dim)[0]
    c.write(0, k, v)
    assert c.fragmentation_ratio() == pytest.approx(0.4)


def test_empty_caches_report_allocation_not_cursor(cfg):
    """The zero guard is on ALLOCATION, not seq_len.

    Paged holds no pages before the first write, so there is nothing to waste
    (0.0). Contiguous holds its whole reservation from construction, so all of
    it is waste (1.0). A `seq_len == 0` guard would collapse both to 0.0 and
    erase the reservation-waste axis the study is about. Same after free().
    """
    p, c = make_paged(cfg, 16), make_contiguous(cfg, 100)
    assert p.n_pages == 0 and p.memory_bytes() == 0
    assert c.memory_bytes() > 0                      # reservation exists already
    assert p.fragmentation_ratio() == pytest.approx(0.0)
    assert c.fragmentation_ratio() == pytest.approx(1.0)

    for cache in (p, c):
        forward_step(cache, rand_kv_all_layers(20, cfg.head_dim))
        cache.free()
    assert p.fragmentation_ratio() == pytest.approx(0.0)
    assert c.fragmentation_ratio() == pytest.approx(1.0)


def test_paged_fragmentation_counts_pages_held_before_advance(cfg):
    """Mid-forward, write() has taken pages and filled them but advance() has
    not run, so seq_len is still 0 while two pages are held. A `seq_len == 0`
    guard would report 0.0 here — the counter-crossing bug that the
    written-extent definition exists to prevent."""
    p = make_paged(cfg, 16)
    k, v = rand_kv_all_layers(17, cfg.head_dim)[0]
    p.write(0, k, v)
    assert p.seq_len == 0 and p.n_pages == 2         # pages held, cursor unmoved
    assert p.fragmentation_ratio() == pytest.approx(15 / 32)


def test_paged_absolute_waste_is_under_one_page(cfg):
    """The property that actually holds at every length: waste is the padding to
    the next page boundary, so it is strictly less than block_size.

    The RATIO is a sawtooth in L, not a decreasing function of it — 31 -> 1/32,
    32 -> 0.0, 33 -> 15/48. Any length grid whose entries share a residue mod
    block_size holds the numerator fixed and moves only the denominator, which
    manufactures a smooth decay that is an artifact of the grid. (The exp1 grid
    does exactly this; see the note at SEQ_LENS in run_cache_comparison.py.)
    """
    bs = 16
    rng = random.Random(0)
    lengths = [1, 15, 16, 17, 31, 32, 33] + [rng.randint(100, 3000) for _ in range(12)]
    cache = make_paged(cfg, bs, num_blocks=600)   # holds 3000 tokens x BATCH
    for L in lengths:
        cache.free()
        cache.write(0, *zero_kv(L, cfg.head_dim))
        waste = cache.fragmentation_ratio() * cache.n_pages * bs   # slots per seq
        assert 0 <= waste < bs, (L, waste)
        assert waste == pytest.approx((-L) % bs), L


def test_paged_waste_ratio_envelope_decays(cfg):
    """What decays with length is the upper ENVELOPE of the ratio, not the ratio.

    Waste is bounded by one page while the allocation grows with L, so the worst
    case over a window falls as the window moves out. Scanning every length in
    [n, 2n] (not sampling, and not one residue class) makes the max a real worst
    case. Asserting pointwise monotonicity here would be asserting something
    false; asserting it over a single residue class would pass for the wrong
    reason.
    """
    bs = 16
    cache = make_paged(cfg, bs, num_blocks=(2048 // bs) * BATCH)
    worst = []
    for n in (16, 64, 256, 1024):
        peak = 0.0
        for L in range(n, 2 * n + 1):
            cache.free()
            cache.write(0, *zero_kv(L, cfg.head_dim))
            peak = max(peak, cache.fragmentation_ratio())
        worst.append(peak)
    assert all(b < a for a, b in zip(worst, worst[1:])), worst
