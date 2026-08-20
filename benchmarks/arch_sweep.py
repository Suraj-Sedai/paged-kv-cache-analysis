"""Does the memory model hold across architectures, or only at 8L/512d?

peak_MB = slope * (batch*seq_len) + intercept was fit at ONE model shape. If the
slope is real it must decompose as

    slope = 2 * n_layers * d_model * bytes_per_elem   (KV cache -- derivable)
          + activation cost per token                 (scales with d_model, d_ff)

Note n_heads * head_dim == d_model in ModelConfig, so the cache term collapses to
2 * n_layers * d_model * bytes. This sweep varies n_layers, d_model and d_ff
independently and checks the measured cache slope against the prediction.

Uses the established peak = f(batch*seq_len) invariance: seq_len is fixed and
batch varies to hit target token counts, rather than sweeping both.

Memory only, no timing. Run:  python -m benchmarks.arch_sweep
"""
import gc
import json
import os

import torch

from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.kv_cache.paged import PagedKVCache

SEQ_LEN, BLOCK_SIZE, DECODE_LEN, SEED = 1024, 16, 8, 0
BUDGET_GB = 9.0
TARGET_PEAKS_MB = [1500, 2500, 3500, 4500]   # 4 points per arch -> real fit + residuals

# (label, n_layers, d_model, n_heads, d_ff). Each row changes ONE axis from the
# baseline so the contribution of that axis is identifiable.
ARCHS = [
    ("baseline    8L/512d/2048ff",   8,  512,  8, 2048),
    ("2x layers  16L/512d/2048ff",  16,  512,  8, 2048),
    ("half layers 4L/512d/2048ff",   4,  512,  8, 2048),
    ("2x d_model  8L/1024d/2048ff",  8, 1024, 16, 2048),
    ("2x d_ff     8L/512d/4096ff",   8,  512,  8, 4096),
    ("gpt2-small 12L/768d/3072ff", 12, 768, 12, 3072)
]
OUT = os.path.join("experiments", "results", "arch_sweep.json")


def cache_kb_per_token(n_layers, d_model, bytes_per=2):
    return 2 * n_layers * d_model * bytes_per / 1024   # 2 = K and V


def set_budget(gb):
    total = torch.cuda.get_device_properties(0).total_memory
    assert gb * 1024**3 < total, "budget must be under device total or the driver hides OOM"
    torch.cuda.set_per_process_memory_fraction(gb * 1024**3 / total)


def cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def measure(n_layers, d_model, n_heads, d_ff, batch, device):
    cleanup()
    total = SEQ_LEN + DECODE_LEN
    cfg = ModelConfig(vocab_size=50257, max_seq_len=total, n_layers=n_layers,
                      n_heads=n_heads, d_model=d_model, d_ff=d_ff)
    model = cache = None
    try:
        torch.manual_seed(SEED)
        model = GPTModel(cfg).to(device).half().eval()
        cache = PagedKVCache(n_layers=n_layers, n_heads=n_heads, dim_head=cfg.head_dim,
                             block_size=BLOCK_SIZE,
                             num_blocks=((total + BLOCK_SIZE - 1)//BLOCK_SIZE)*batch,
                             batch_size=batch, device=device, dtype=torch.float16)
        ids = torch.randint(0, cfg.vocab_size, (batch, SEQ_LEN), device=device)
        with torch.no_grad():
            logits = model(ids, kv_cache=cache, use_cache=True, start_pos=0)
            cache.advance(SEQ_LEN)
            for s in range(DECODE_LEN):
                logits = model(logits[:, -1:].argmax(-1), kv_cache=cache,
                               use_cache=True, start_pos=SEQ_LEN + s)
                cache.advance(1)
        return torch.cuda.max_memory_allocated() / 1024**2
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            return None
        raise
    finally:
        del model, cache
        cleanup()


def fit(xs, ys):
    """Least squares y = a*x + b. Returns (slope, intercept, max abs resid)."""
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    a = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    b = my - a*mx
    return a, b, max(abs(a*x + b - y) for x, y in zip(xs, ys))


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    set_budget(BUDGET_GB)
    print(f"{torch.cuda.get_device_name()}   seq_len {SEQ_LEN}, budget {BUDGET_GB} GB\n")

    results = []
    for label, L, D, H, FF in ARCHS:
        pred_cache = cache_kb_per_token(L, D)
        # rough total slope for picking batch sizes: cache + activations (~11 KB at
        # d_model 512, assumed to scale with d_model and d_ff)
        est = pred_cache + 11.0 * (D/512) * (0.5 + 0.5*FF/2048)
        xs, ys = [], []
        for target in TARGET_PEAKS_MB:
            batch = max(1, round((target - 170) * 1024 / est / SEQ_LEN))
            peak = measure(L, D, H, FF, batch, "cuda")
            if peak is None:
                print(f"  {label}: OOM at batch {batch}, skipping point"); continue
            xs.append(batch * SEQ_LEN); ys.append(peak)
        if len(xs) < 3:
            print(f"  {label}: too few points to fit"); continue

        slope_kb, intercept, resid = fit(xs, ys)
        slope_kb *= 1024   # MB/token -> KB/token
        act = slope_kb - pred_cache
        print(f"{label}")
        print(f"   measured slope {slope_kb:7.2f} KB/token   intercept {intercept:6.0f} MB   "
              f"max resid {resid:5.1f} MB")
        print(f"   predicted cache {pred_cache:6.2f}   ->  activation residual "
              f"{act:6.2f} KB/token   ({100*act/slope_kb:.0f}% of slope)")
        results.append(dict(label=label, n_layers=L, d_model=D, n_heads=H, d_ff=FF,
                            tokens=xs, peaks=ys, slope_kb_per_token=round(slope_kb, 3),
                            intercept_mb=round(intercept, 1), max_resid_mb=round(resid, 2),
                            predicted_cache_kb=round(pred_cache, 3),
                            activation_kb=round(act, 3)))
        print()

    torch.cuda.set_per_process_memory_fraction(1.0)

    print("=" * 72)
    print("What to look for:")
    print("  - activation residual should be ~EQUAL for baseline / 2x layers / half")
    print("    layers (transients are freed per layer, so they should not scale with L)")
    print("  - activation residual should RISE with d_model and with d_ff")
    print("  - measured slope minus predicted cache should be positive everywhere;")
    print("    if it is negative the cache prediction is wrong, not the activations")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
