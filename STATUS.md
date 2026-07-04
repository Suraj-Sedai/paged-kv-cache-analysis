# STATUS — volatile experiment state

## Canonical CSVs

| exp | file | status |
|-----|------|--------|
| 1 — layout | `experiments/results/exp1_layout_comparison.csv` | done (real GPU) |
| 2 — block size (H2) | `experiments/results/exp2_block_size_sweep.csv` | done; small block (8–16) wins frag+mem. Throughput ordering NOT reproducible across reruns (launch-bound) — claim withdrawn |
| 3a — precision memory frontier (H3) | `experiments/results/exp3_precision_sweep.csv` | done; study GPTModel, paged FP16 vs INT8 |
| 3b — precision quality (H3) | `experiments/results/exp3_perplexity.csv` | done; HF GPT-2 / gpt2-medium on WikiText-2 test |

Corpus for 3b: `experiments/data/wikitext2_test.txt` (WikiText-2 raw test, 284,614 GPT-2 tokens).

All experiments now have real GPU numbers — per CLAUDE.md §7 the critical path is
satisfied. Remaining work is synthesis (plots + writeup), not more sweeps.

---

## Done since last update (2026-06-15, commit f7826fa)

- Exp 3b run on real corpus (WikiText-2 test, 284k tok) for gpt2 and gpt2-medium.
  Dropped the throwaway 1,135-token smoke row (had wrong-sign delta, misleading).
- Fixed `0.53x` → `0.516x` docstrings in `run_precision_sweep.py` and `int8_paged.py`.
- Inspected `INT8KVCache.read()` for invariant #5 — confirmed present; decided caveat
  not refactor (see Latency below).
- Created this `STATUS.md`. Rewrote `README.md` (in-depth, real numbers, honest negatives).
- Nothing committed yet — all working-tree edits.

---

## Experiment 1 findings (H1: layout crossover)

Contiguous vs paged on the study model. On this GPU contiguous wins decode throughput
across the tested grid (e.g. 128×16: contiguous ~1368 vs paged ~438 tok/s). Two reasons,
both honest caveats not clean wins: (a) paged read copies (invariant #4), taxing TPOT;
(b) uniform-length batched forward means paging's memory flexibility never surfaces
(invariants #6/#7). So Exp 1 characterizes **the overhead cost of paged indirection when
the memory benefit doesn't apply** — a defensible framing (§7), NOT "paged is worse".

## Experiment 2 findings (H2: block-size optimum) — MECHANISM SUPPORTED, THROUGHPUT CLAIM WITHDRAWN

Block size ∈ {8,16,32,64}, frag_ratio varies correctly (0.0–0.875). Robust, deterministic
half holds: small blocks (8–16) cut fragmentation/memory; large blocks lower TPOT (fewer
per-token lookups). E.g. seq 1100 × batch 16: last-block waste ~0.0 (bs8) → 0.875 (bs64).

Throughput ordering does NOT reproduce: an earlier sweep had bs8 ~400 / bs64 ~240 tok/s
(small wins); a later rerun on the same model reversed it (bs32 227 → 495 tok/s, swings
>2×). Decode is launch-bound, so throughput is too noisy to crown a block size. The
"small blocks win throughput / use ≤16" headline is withdrawn — not reproducible.

## Experiment 3 findings (H3: does precision shift the crossover?)

**Memory (clean, deterministic):** INT8 cache = **0.516×** FP16 = `(1 + 2/head_dim)/2`
at head_dim 64. Holds exactly across every (seq,batch) cell.

**Quality (clean):** INT8 KV is effectively lossless on WikiText-2 test —
| model | Δppl | % | top1 |
|-------|------|---|------|
| gpt2 | +0.0007 | +0.002% | 0.9854 |
| gpt2-medium | +0.0016 | +0.006% | 0.9883 |
Per-token-symmetric scheme holds at least to gpt2-medium; outlier-channel concern
(KIVI/SmoothQuant) does not bite at this scale. Do not generalize past measured sizes.

**OOM frontier (NEGATIVE — H3 falsified on this GPU):** INT8 OOMs at the *same* cells
as FP16 (8192×16, 8192×32). Cache is ~10% of peak memory (e.g. 4096×16: cache 1040 MB
vs peak 9872 MB); the OOM is set by prefill activation memory (batch×heads×seq²), which
is cache-precision-independent. Halving 10% can't move the frontier.

**Latency (CAVEAT, not a clean finding):** INT8 TPOT runs ~2–3× FP16. This is
contaminated and NOT an intrinsic INT8 cost:
- `INT8KVCache.read()` returns full history and re-dequantizes it every decode step
  (CLAUDE.md §4 invariant #5; inherited from PagedKVCache's full-history read).
- The ratio does NOT climb with seq_len — it falls (~3.1x at seq 1024 to 1.33x at
  8192). History-size dequant would make it grow; a fixed per-step cost shrinks as a
  fraction as FP16's own per-step cost rises. So the tax is extra per-step quant/dequant
  **kernel launches** in a launch-bound regime, not history-size dequant arithmetic.
- A fused dequant-in-attention path would remove most of it. Report as harness/regime
  artifact, NOT "INT8 costs 2–3× decode latency". `read()` deliberately NOT refactored
  (§7: data is the critical path; refactor doesn't change memory/quality findings).

**Paper line for H3:** favorable memory↔quality (0.516× at ~0 quality cost), unfavorable
memory↔latency, no OOM-frontier shift on this GPU. Keep INT8 a *sub-result* of the
block-size paper, not a co-headline (the frontier-shift story we can't tell well here).

---

### Notes / known non-issues

- frag_ratio in 3a is constant 0.25 — informationless there (all seq_len multiples of
  block_size 16, +60 decode → wasted=4/16). Correctly computed, NOT hardcoded; just
  carries no signal in that aligned sweep. Drop it from any Exp-3 frag claim. (Exp 2's
  frag_ratio does vary, 0.0–0.875, and is the valid one.)
- README motivation paragraph is Claude's reconstruction of *why* — Suraj to verify it
  matches his actual reasoning before the paper leans on it.
