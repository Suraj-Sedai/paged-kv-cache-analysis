# STATUS — working notes, 2026-08-20

Memory is the whole study. The architecture sweep (2eab193) closes the last gap: the
peak-memory model was fit at one model shape, and it now holds at six. Given
`(n_layers, d_model, d_ff, dtype, batch, seq_len)` it predicts peak memory to about
1% on a shape nothing was fitted on. That turns the curve fit into a model, and it is
the result the paper is built around. Latency stays out — decode is launch-bound at
this model size and no timing claim has reproduced.

## Verified

- Memory model transfers across shapes. Six architectures, one axis moved at a time,
  all linear in resident tokens (worst residual 2.4 MB over a 3 GB span). Slope splits
  into a derivable cache term and a measured activation term, positive everywhere —
  `experiments/results/arch_sweep.json`, notebook 05.
- Activations are invariant in depth (3.6% across a 4× change in `n_layers`, while the
  cache term moves 4×) and linear in width: +29% for 2× `d_model`, +71% for 2× `d_ff`.
- Held out: gpt2-small (12L/768d/3072ff) slope predicted within 0.37%, full peak
  within 1.1%, with no parameter fitted on that shape.
- Scope of that result: the sweep moves `n_layers`, `d_model`, `d_ff`. It does NOT
  move `n_heads` and `head_dim` independently — `head_dim` is pinned at 64 and
  `n_heads` moves with `d_model`, so the cache term is only ever confirmed as the
  product `2·n_layers·d_model·bytes`. The factorization into heads × head_dim is
  untested, and the model as written would not transfer to GQA/MQA, where the KV
  head count is decoupled from `d_model`. Say this before a reviewer does.
- Baseline slope replicates across scripts: 27.25 KB/token in the arch sweep vs 27.15
  in the precision sweep (+0.36%), different harness path.
- Peak = f(batch × seq_len). 8192×32 vs 16384×16 within 0.058% (fp16), 8192×64 vs
  16384×32 within 0.038% (int8) — `experiments/results/oom_frontier.json`.
- Fits: fp16 27.15 KB/token + 168.0 MB, int8 19.93 + 164.0, R² ≥ 0.99997 on
  `exp3_precision_sweep.csv`; held-out error +0.18% to +0.46% on the frontier cells.
- Cache term matches `2·n_layers·n_heads·head_dim·bytes` = 16 KB/token fp16,
  8.25 int8. Measured 16.125 at seq 8192 (surcharge is the 60 decode tokens).
- Intercepts are parameter bytes plus 9–25 MB of workspace at all six shapes.
- Exp 2 is on the current harness (re-run in ae0cb58, 60 paged cells). `frag_ratio`
  is on the waste-over-total-allocation definition and rises weakly with block size
  at every seq_len. Peak agrees with the memory model — 1022.86 MB measured at
  2000×16 against 1016.4 predicted, +0.6% — which only holds post-lm_head-slice.
- H3 holds under a budget: FP16 OOM / INT8 ok at 8192×48 (9, 10 GB), 8192×64 and
  16384×32 (11, 11.5 GB) — `oom_frontier.json` verdicts.
- INT8 cache ratio 0.515627–0.515639 vs predicted 0.515625 —
  `tests/test_cache_equivalence.py::test_int8_memory_ratio_is_0516`.
- INT8 quality: +0.002% ppl (gpt2), +0.006% (gpt2-medium), 284,614 tokens —
  `exp3_perplexity.csv`.
- Equivalence gate green: `python -m pytest -q` → 56 passed.

## Next

0. Clear the conflict markers out of `paper/figures/*_data.csv` and point the H3
   figure at `oom_frontier.json`. Nothing else on this list matters while the figure
   data in the repo is unparseable and the H3 figure contradicts the H3 result.
1. Record the GPU name and the commit in every output file. Three of the results
   files carry neither, and the paper quotes all three. Cheapest fix here, do it
   before anything is re-run.
2. Repeat the arch sweep in INT8. The transfer result is FP16 only, so the paper
   currently claims transfer for one dtype and a precision effect at one shape.
   Same script, one flag.
3. Isolate the 7.5% of INT8 cache savings that never reach peak (0.58 KB/token,
   steady across cells). Suspect the fp32/fp16 temporaries in `_quantize`/`read`.
   Snapshot allocator stats per phase rather than guessing.
4. Decide what Exp 2 is for. The re-run is done and clean, but with uniform-length
   batches `frag_ratio` is closed-form (see below), so the sweep as it stands cannot
   carry a fragmentation finding. Either give it variable lengths and a reservation
   policy, or demote it to a worked example and stop calling it an experiment.
5. Interleave cell execution order in the sweep scripts (see below) before any
   re-run that will be quoted.

## Known broken / not done

- **All four `paper/figures/*_data.csv` files contain unresolved git conflict
  markers** — `<<<<<<< HEAD` / `=======` / `>>>>>>> 6e17cd9`, committed in the merge
  66e2af7 and still in the tree. These are the per-figure data dumps, the exact
  artifact a reviewer opens to check a figure. Fix first; it costs one re-run of
  `make_paper_figures.py`.
- The H3 figure is built from the wrong file. `figure4()` reads
  `exp3_precision_sweep.csv`, where post-lm_head-slice every cell is `ok`, so it
  prints "cells INT8 rescues: []" and draws two identical grids. The actual H3
  evidence is `oom_frontier.json`, which no figure script reads. Same root cause as
  the notebook 03 problem below.
- The headline result has no figure. Neither `make_figures.py` nor
  `make_paper_figures.py` reads `arch_sweep.json` or `oom_frontier.json`. The memory
  model, the frontier and the transfer result exist only inside notebooks 04 and 05.
- `make_paper_figures.py`'s docstring pins its inputs to commit e383030 (2026-06-21).
  exp1 was regenerated 2026-08-12 and exp2 2026-08-13, so the header describes files
  that no longer exist.
- Paged pre-allocates the whole pool. `PagedKVCache.__init__` does `torch.zeros` over
  `num_blocks`, and every benchmark sizes `num_blocks = ceil(final_len/block)*batch`,
  the exact worst case. `memory_bytes()` correctly reports live pages (invariant 3
  holds for that metric), but `peak_memory_mb` sees the full pool from construction.
  On-demand allocation therefore cannot show up in peak at all — which is the real
  reason H1's "peak equal within 0.9%" was never a finding, deeper than the lockstep
  batch. Any future memory claim for paged needs the pool sized above the workload.
- Frozen files have been modified. CLAUDE.md §5 freezes `src/model_core/`,
  `src/inference/`, `src/profiling/`. `src/model_core` changed in 2a86eeb, fceab20
  and 237ad2d (the lm_head slice), `src/inference` in 2a86eeb, 0d029d6 and fceab20.
  The changes look necessary and are documented, but the contract no longer matches
  the repo — either re-scope §5 or record these as accepted exceptions.
- The merge 66e2af7 duplicated commits: b7a83be/e414dee, 6e17cd9/ae0cb58 and
  98d0e75/237ad2d are pairs with identical messages. Commit stamps quoted in notes
  may name the twin that is not on the current history.
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
- Exp 2's `frag_ratio` is arithmetic, not measurement. All 60 values reproduce
  exactly from `ceil((seq_len+60)/block)*block` — zero mismatches, no GPU needed.
  Every sequence in a batch has the same length, so the waste is fixed by the
  remainder and the run cannot tell you anything the formula does not. The column is
  correct and on the right definition; it just is not evidence. A fragmentation
  finding needs variable lengths, where the quantity becomes a distribution.
- `exp2_block_size_sweep.csv` carries no provenance columns. exp1 writes
  `alloc_basis, dtype, attn_impl, commit`; exp2 writes none of them, so the file
  cannot be tied to a commit from its own contents.
- The `run_block_size_sweep.py` docstring still says the CSV in the repo predates
  the port. It does not — the same commit regenerated both.
- `analysis/notebooks/03_precision_analysis.ipynb` still prints "H3 NOT SUPPORTED".
  It reads the sweep grid, where nothing OOMs; `oom_frontier.json` supersedes it.
- `exp1_layout_comparison.csv` carries commit `fb29cd5-dirty` although it ships in
  237ad2d. The stamp is the parent plus a dirty tree, not the commit that has it.
- The GPU model is not recorded in any output file. `oom_frontier.py` and
  `arch_sweep.py` print the device name but do not write it. `arch_sweep.json`
  records no commit either.
- Two figure generators write into `paper/figures` with different names and
  contradicting stories. `make_figures.py` still produces
  `fig1_block_throughput` under the withdrawn "block-size optimum" framing;
  `make_paper_figures.py` is the current one. Both sets were regenerated in 6632086
  and both are sitting in the directory. Delete or retire the old script before the
  figures go anywhere near a submission.
- The arch sweep is FP16 only, four points per shape, and each width axis moves by
  exactly one doubling. Enough to rule out curvature that would matter, not enough
  for a confidence interval. Batch is chosen per shape to hit a target peak, so no
  two shapes are compared at an identical workload — fine for a slope, not for a
  like-for-like memory comparison.

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
