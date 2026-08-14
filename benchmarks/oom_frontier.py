"""Find the memory budget at which the FP16 KV cache OOMs and INT8 survives.

H3 claims INT8 quantization shifts the OOM frontier outward. The precision
sweep shows INT8 taking ~26% off peak at high cache_frac, but nothing in that
grid OOMs, so the literal claim was untested.

WHY A SOFTWARE BUDGET RATHER THAN PHYSICAL VRAM: on Windows (WDDM), NVIDIA's
CUDA sysmem fallback is enabled by default, so an allocation that exceeds VRAM
silently spills to host RAM over PCIe instead of raising. A first pass of this
script reported 14009 MB allocated as "ok" on an 11.9 GB device -- no OOM is
reachable on this machine. set_per_process_memory_fraction makes PyTorch's own
allocator raise before the driver fallback engages.

This is also better methodology than relying on the device: "the frontier under
a stated budget of B GB" reproduces on any card, where "the frontier on my
laptop" does not. Sweeping the budget turns a single anecdote into a curve.

Memory only -- no timing loop. Peak is reached during prefill (activations plus
a nearly-full cache, since _ensure_capacity allocates prefill pages up front),
so prefill plus a few decode steps is enough.

Run:  python -m benchmarks.oom_frontier
"""
import gc
import json
import os

import torch

from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache

N_LAYERS, N_HEADS, D_MODEL, D_FF = 8, 8, 512, 2048
BLOCK_SIZE, DECODE_LEN, SEED = 16, 8, 0

# Ladder. Two directions out of the last known-good cell (8192 x 32):
# grow the batch (activations grow too) and grow the context at fixed batch
# (shifts the ledger toward cache-dominated, which is where INT8 should win).
LADDER = [(8192, 32), (8192, 48), (8192, 64), (16384, 16), (16384, 32)]

# Budgets in GB. The interesting window from the unbudgeted run is 10-14 GB:
# fp16 peaks at 14009 MB and int8 at 10329 MB on the two frontier cells, so
# fp16 should fail and int8 survive between them, and both should fail below.
BUDGETS_GB = [8.0, 9.0, 10.0, 11.0, 11.5]

OUT = os.path.join("experiments", "results", "oom_frontier.json")


def set_budget(gb):
    """Cap the PyTorch allocator so OOM is raised in-process.

    Must be called after any prior allocation is freed; the fraction applies to
    total device memory, not to what is currently free.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    frac = gb * 1024 ** 3 / total
    if frac >= 1.0:
        raise ValueError(f"budget {gb} GB >= device total {total/1024**3:.1f} GB; "
                         "the driver fallback would hide the OOM again")
    torch.cuda.set_per_process_memory_fraction(frac)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_cell(seq_len, batch, quantized, device):
    """Return (status, peak_mb, cache_mb). status in {ok, oom, pool_oom}."""
    cleanup()
    total = seq_len + DECODE_LEN
    per_seq = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    cfg = ModelConfig(vocab_size=50257, max_seq_len=total, n_layers=N_LAYERS,
                      n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
    model = cache = None
    try:
        torch.manual_seed(SEED)
        model = GPTModel(cfg).to(device).half().eval()
        kw = dict(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
                  block_size=BLOCK_SIZE, num_blocks=per_seq * batch,
                  batch_size=batch, device=device)
        cache = (INT8KVCache(**kw, compute_dtype=torch.float16) if quantized
                 else PagedKVCache(**kw, dtype=torch.float16))

        ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
        with torch.no_grad():
            logits = model(ids, kv_cache=cache, use_cache=True, start_pos=0)
            cache.advance(seq_len)
            for s in range(DECODE_LEN):
                nxt = logits[:, -1:].argmax(-1)
                logits = model(nxt, kv_cache=cache, use_cache=True, start_pos=seq_len + s)
                cache.advance(1)

        peak = torch.cuda.max_memory_allocated() / 1024**2
        cm = cache.memory_bytes() / 1024**2
        return "ok", round(peak, 1), round(cm, 1)

    except Exception as e:
        if is_oom(e):
            return "oom", None, None
        if "no free pages" in str(e):
            # pool exhaustion, not device OOM -- a harness bug, not a result
            return "pool_oom", None, None
        raise
    finally:
        del model, cache
        cleanup()


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    dev = "cuda"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{torch.cuda.get_device_name()}  ({total_gb:.1f} GB physical)")
    print("budgets enforced in-process; the driver sysmem fallback would "
          "otherwise hide every OOM\n")

    results = []
    for gb in BUDGETS_GB:
        set_budget(gb)
        print(f"--- budget {gb:g} GB " + "-" * 46)
        print(f"{'seq':>7} {'batch':>6} {'fp16':>22} {'int8':>22}   verdict")
        for seq_len, batch in LADDER:
            row = {"budget_gb": gb, "seq_len": seq_len, "batch": batch}
            cells = []
            for label, q in (("fp16", False), ("int8", True)):
                st, peak, cm = try_cell(seq_len, batch, q, dev)
                row[label] = {"status": st, "peak_mb": peak, "cache_mb": cm}
                cells.append(f"{st:>4}" + (f" peak {peak:>7.0f} cache {cm:>6.0f}"
                                           if peak else " " * 21))
            f_st, i_st = row["fp16"]["status"], row["int8"]["status"]
            verdict = ("FRONTIER" if (f_st, i_st) == ("oom", "ok")
                       else "both ok" if (f_st, i_st) == ("ok", "ok")
                       else "both OOM" if (f_st, i_st) == ("oom", "oom") else "?")
            row["verdict"] = verdict
            print(f"{seq_len:>7} {batch:>6} {cells[0]:>22} {cells[1]:>22}   {verdict}")
            results.append(row)
        print()

    torch.cuda.set_per_process_memory_fraction(1.0)   # release the cap

    fr = [r for r in results if r["verdict"] == "FRONTIER"]
    if fr:
        print("H3 result -- cells where FP16 OOMs and INT8 survives:")
        for r in fr:
            print(f"  budget {r['budget_gb']:g} GB   seq {r['seq_len']} x batch {r['batch']}"
                  f"   int8 peak {r['int8']['peak_mb']:.0f} MB")
    else:
        print("No cell separates the two dtypes. If both OOM everywhere the cache is "
              "not the binding constraint -- prefill activations are. Report that.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()