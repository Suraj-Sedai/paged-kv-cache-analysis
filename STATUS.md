# STATUS — working notes, 2026-08-13

Memory is the whole study now. The OOM frontier run (e414dee) is the last piece of
new data: under enforced budgets INT8 survives three cells FP16 cannot, and peak
memory turns out to be a function of `batch × seq_len` alone, fit to under 0.5% on
held-out cells. Latency is out — decode is launch-bound at this model size and no
timing claim has reproduced. Next is the architecture sweep, which is the one thing
that would turn the memory model from a curve fit into a prediction.

## Verified

- Peak = f(batch × seq_len). 8192×32 vs 16384×16 within 0.058% (fp16), 8192×64 vs
  16384×32 within 0.038% (int8) — `experiments/results/oom_frontier.json`.
- Fits: fp16 27.15 KB/token + 168.0 MB, int8 19.93 + 164.0, R² ≥ 0.99997 on
  `exp3_precision_sweep.csv`; held-out error +0.18% to +0.46% on the frontier cells.
- Cache term matches `2·n_layers·n_heads·head_dim·bytes` = 16 KB/token fp16,
  8.25 int8. Measured 16.125 at seq 8192 (surcharge is the 60 decode tokens).
- H3 holds under a budget: FP16 OOM / INT8 ok at 8192×48 (9, 10 GB), 8192×64 and
  16384×32 (11, 11.5 GB) — `oom_frontier.json` verdicts.
- INT8 cache ratio 0.515627–0.515639 vs predicted 0.515625 —
  `tests/test_cache_equivalence.py::test_int8_memory_ratio_is_0516`.
- INT8 quality: +0.002% ppl (gpt2), +0.006% (gpt2-medium), 284,614 tokens —
  `exp3_perplexity.csv`.
- Equivalence gate green: `python -m pytest -q` → 56 passed.

## Next

1. Architecture sweep. Fit the memory model at 3–4 model shapes (vary n_layers,
   n_heads, head_dim independently) and check the measured slope against
   `2·n_layers·n_heads·head_dim·bytes`. One shape is a fit; four is a model.
2. Isolate the 7.5% of INT8 cache savings that never reach peak (0.58 KB/token,
   steady across cells). Suspect the fp32/fp16 temporaries in `_quantize`/`read`.
   Snapshot allocator stats per phase rather than guessing.
3. Re-run exp2 under the current harness so the block-size fragmentation numbers
   are on the current definition.
4. Interleave cell execution order in the sweep scripts (see below) before any
   re-run that will be quoted.

## Known broken / not done

- `benchmarks/validity_gate.py` and `benchmarks/regen_v2.py` are on the v1 cache
  API — positional `ContiguousKVCache(config, max_seq_len, device)` without
  `batch_size`, and per-sequence `write(layer, b, k, v)` / `free(b)`. Both crash on
  the batched interface. Not deleted; not run since fceab20.
- No trace harness. It is the prerequisite for a real reservation policy
  (Oracle / FixedMax / Percentile), referenced in `run_cache_comparison.py:87` and
  `analysis/make_figures.py:73`.
- Cell execution is not interleaved. All three sweeps loop cache_type (or
  block_size) outermost, so one arm runs to completion before the other starts and
  thermal drift lands entirely on the arm that runs last. Clocks are not locked.
  This is a live confound for any A/B timing claim in these CSVs.
- `exp2_block_size_sweep.csv` has not been regenerated since e383030 (2026-06-21).
  It predates the batched interface, the SDPA switch, the lm_head slice and the
  fragmentation redefinition. Its timing, peak_memory and frag_ratio columns are on
  old semantics. Only cache_memory_mb carries over.
- `analysis/notebooks/03_precision_analysis.ipynb` still prints "H3 NOT SUPPORTED".
  It reads the sweep grid, where nothing OOMs; `oom_frontier.json` supersedes it.
- `exp1_layout_comparison.csv` carries commit `fb29cd5-dirty` although it ships in
  237ad2d. The stamp is the parent plus a dirty tree, not the commit that has it.
- The GPU model is not recorded in any output file. `oom_frontier.py` prints the
  device name but does not write it to the JSON.

## Withdrawn claims

- 2026-06-21 — "small blocks win throughput." Two runs of the same script gave
  226.6 and 495.3 tok/s at block 32, 1100×16. Launch-bound; not reproducible.
- 2026-08-12 — "contiguous fragments less than paged." Contiguous reserves the exact
  final length, so 0.0 is the setup restated. Deferred to a reservation policy.
- 2026-08-12 — H1, "paged overtakes contiguous somewhere in the grid." No crossover:
  0.32×–0.77× throughput across 20 cells, peak equal within 0.9%. The grid cannot
  show one — lockstep batch, copying read.
- 2026-08-13 — "H3 is falsified on this GPU / the cache is ~10% of peak." That came
  from the pre-lm_head-slice sweep, where an unsliced logits tensor was most of
  peak. Cache share is 50–58% at the large cells and the frontier does move.
