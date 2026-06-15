"""Experiment 3b — quality cost of INT8 KV on REAL weights (HF GPT-2).

Why not the study's GPTModel: it is randomly initialized (fine for memory/latency,
useless for perplexity) and is NOT architecturally GPT-2-compatible (no final
LayerNorm, exact-GELU), so real weights cannot be loaded into it without editing
frozen model_core (CLAUDE.md §5). So the quality axis runs on HF GPT-2 directly,
and is labeled as a separate model from the memory-frontier sweep (3a).

How INT8 KV is injected: a DynamicCache subclass that quantizes each new key/value
with the SAME per-token symmetric int8 scheme as src/kv_cache/int8_paged.py
(scale = amax/127 over head_dim) before storing, and returns the dequantized
tensors attention actually uses. Because the scheme is PER-TOKEN (each position's
scale depends only on its own vector), fake-quantizing the full sequence in one
teacher-forced forward is mathematically identical to incrementally quantizing each
token as it is cached — so a single forward yields the exact int8-cache perplexity.

Metric: delta perplexity, PPL(int8) - PPL(fp16), on a fixed corpus. The absolute
PPL is not the claim; the delta (and % increase) is. Caveat for the paper: this
measures the per-token-symmetric scheme on GPT-2's activations; KIVI/SmoothQuant
show real models have outlier channels that make naive symmetric quant worse than
on well-behaved layers — report this honestly, do not overclaim INT8 is "free".
"""
import argparse
import csv
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

DEFAULT_TEXT = os.path.join("experiments", "data", "eval_text.txt")
OUTPUT_PATH  = os.path.join("experiments", "results", "exp3_perplexity.csv")
FIELDNAMES   = ["model", "ctx", "n_tokens", "ppl_fp16", "ppl_int8",
                "ppl_delta", "ppl_pct_increase", "top1_agreement"]


def fake_quant_per_token(x):
    """Per-token symmetric int8 round-trip, BIT-IDENTICAL to INT8KVCache (codes
    from an fp32 scale; scale then stored/dequantized in x's dtype, i.e. fp16).
    x: [..., head_dim], scale over last dim. All-zero tokens -> scale 0 -> 0."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-8)                      # fp32 for the codes
    codes = torch.round(xf / scale).clamp(-127, 127).to(torch.int8)
    return codes.to(x.dtype) * scale.to(x.dtype)               # fp16 dequant, as the cache does


class Int8KVQuantCache(DynamicCache):
    """DynamicCache that stores K/V as if quantized to per-token int8."""
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return super().update(
            fake_quant_per_token(key_states),
            fake_quant_per_token(value_states),
            layer_idx, *args, **kwargs,
        )


@torch.no_grad()
def eval_corpus(model, input_ids, ctx, cache_factory):
    """Teacher-forced NLL over non-overlapping windows of length `ctx`, using a
    fresh cache per window (use_cache=True so K/V flow through cache.update).
    Returns (perplexity, per-position argmax tokens) for top-1 agreement."""
    device = next(model.parameters()).device
    total_nll, total_tok = 0.0, 0
    argmaxes = []
    for start in range(0, input_ids.size(1) - 1, ctx):
        window = input_ids[:, start:start + ctx + 1].to(device)
        if window.size(1) < 2:
            break
        inp, tgt = window[:, :-1], window[:, 1:]
        logits = model(inp, past_key_values=cache_factory(), use_cache=True).logits
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += tgt.numel()
        argmaxes.append(logits.argmax(-1).reshape(-1))
    return math.exp(total_nll / total_tok), torch.cat(argmaxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id (gpt2, gpt2-medium, ...)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="eval corpus (plain text)")
    ap.add_argument("--ctx", type=int, default=512, help="context window for NLL")
    ap.add_argument("--max-tokens", type=int, default=0, help="cap tokens (0 = all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device {device} | model {args.model} | dtype {dtype} | ctx {args.ctx}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt").input_ids
    if args.max_tokens:
        ids = ids[:, :args.max_tokens]
    print(f"Eval tokens: {ids.size(1)}")

    ppl_fp16, am_fp16 = eval_corpus(model, ids, args.ctx, DynamicCache)
    ppl_int8, am_int8 = eval_corpus(model, ids, args.ctx, Int8KVQuantCache)

    top1 = (am_fp16 == am_int8).float().mean().item()
    delta = ppl_int8 - ppl_fp16
    pct = 100.0 * delta / ppl_fp16

    row = {"model": args.model, "ctx": args.ctx, "n_tokens": ids.size(1),
           "ppl_fp16": round(ppl_fp16, 4), "ppl_int8": round(ppl_int8, 4),
           "ppl_delta": round(delta, 4), "ppl_pct_increase": round(pct, 3),
           "top1_agreement": round(top1, 4)}
    print(row)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
