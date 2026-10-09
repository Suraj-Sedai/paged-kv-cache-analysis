# Status — 2026-10-05

What is current, what is broken, and what is still open. Results are in the [README](README.md).

## Which outputs are current

All files are in `experiments/results/` unless noted.

| File | Used for | State |
|---|---|---|
| `arch_sweep.json` | Memory model | Current. No commit or GPU recorded. Its `activation_kb` field subtracts the uncorrected 16 KB cache term; subtract the corrected term (× 1040/1024) to get 11.00 KB. |
| `exp3_precision_sweep.csv` | Cache share, INT8 savings, FP16/INT8 fits | Current |
| `oom_frontier.json` | Memory-budget frontier | Current. No GPU recorded. |
| `exp1_layout_comparison.csv` | Paged vs contiguous (withdrawn H1) | Current. Commit stamp reads `fb29cd5-dirty`. |
| `exp2_block_size_sweep.csv` | Block-size sweep | Current, but `frag_ratio` reproduces from arithmetic alone, and the file has no provenance columns |
| `exp3_perplexity.csv`, rows 1–2 | INT8 quality | Current (WikiText-2 test, 284,614 tokens) |
| `exp3_perplexity.csv`, row 3 | — | 1,135-token test run. Remove. |
| `exp1_layout_comparison1.csv`, `exp1_layout_comparison2.csv` | — | Old schema, superseded. Remove. |
| `analysis/notebooks/04`, `05` | Memory model, frontier | Current data, but 05 still computes the uncorrected model (0.37% error on GPT-2) |
| `analysis/notebooks/01`–`03` | — | Pre-fix analyses. 03 still prints "H3 NOT SUPPORTED". |

## Settled numbers

Earlier notes quoted 3.6%, 0.37%, 1.1% and "about 1%" for the memory model. Those came from subtracting a 16 KB cache term that ignored decode tokens and page rounding. The corrected values:

| Quantity | Value | Source |
|---|---|---|
| Per-token error, GPT-2 small (held out) | 0.03% | `arch_sweep.json`, corrected cache term |
| Worst per-token error, six shapes | 0.08% | same |
| Activation term at 4 / 8 / 16 layers | 11.01 / 11.00 / 11.04 KB/token | same |
| Same total tokens, different shape | FP16 0.058% at 262k tokens; INT8 0.17% at 262k, 0.038% at 524k | `oom_frontier.json` |
| FP16 / INT8 fits | 27.15 KB/token + 168 MB / 19.93 + 164 | `exp3_precision_sweep.csv` |
| Paged / contiguous throughput | 0.32–0.77× | `exp1_layout_comparison.csv` |
| Cache share before the LM-head fix | 4–14% across the grid | `exp3_precision_sweep.csv` at e383030 |

## Known broken

- All four `paper/figures/*_data.csv` files contain unresolved merge-conflict markers from merge 66e2af7.
- The H3 figure reads `exp3_precision_sweep.csv`, where nothing runs out of memory, so it shows no frontier. It should read `oom_frontier.json`.
- The memory model and the frontier have no figure script. They exist only in notebooks 04 and 05.
- `analysis/make_figures.py` still draws withdrawn claims (block-size optimum), and its figures sit in `paper/figures` next to the current ones.
- `benchmarks/validity_gate.py` and `benchmarks/regen_v2.py` use the old per-sequence cache API and crash.
- The paged cache allocates its whole page pool up front, so peak memory cannot show on-demand allocation. Any paging memory claim needs a pool larger than the workload.
- Sweeps run one arm to completion before the other, with unlocked clocks, so thermal drift lands on whichever runs last. Don't quote timing comparisons from these CSVs.
- About 7.5% of the INT8 cache saving (≈ 0.58 KB/token) never reaches peak. Cause unknown.
- Merge 66e2af7 duplicated three commits (b7a83be/e414dee, 6e17cd9/ae0cb58, 98d0e75/237ad2d). Older notes may cite the twin.
- Docstrings in `make_paper_figures.py` and `run_block_size_sweep.py` describe files that no longer exist.

## Open, in order

1. Fix the figure data. Point the H3 figure at `oom_frontier.json`. Add figures for the memory model and cache share.
2. Add the corrected cache term to notebook 05 and to `arch_sweep.py`'s output.
3. Write the GPU name and commit into every output file. Add a `corpus` column to the perplexity CSV.
4. Delete the stale scripts, old figures, superseded CSVs, and notebook 03's verdict.
5. Rerun the architecture sweep in INT8. Transfer is shown for FP16 only.
6. Snapshot allocator stats per phase to explain the INT8 shortfall.
7. Add variable-length sequences and a reservation policy. Needed before any paging or fragmentation claim.
8. Interleave cell order before rerunning anything timed.

## Withdrawn claims

- 2026-06-21 — "Small blocks win throughput." Two runs of the same config gave 226.6 and 495.3 tok/s at block 32, 1100×16.
- 2026-08-12 — "Contiguous fragments less than paged." Contiguous reserved the exact final length, so 0.0 was the setup restated.
- 2026-08-12 — H1, "paged overtakes contiguous somewhere in the grid." No crossover in 20 cells. The grid can't show one: lockstep batch, full pool allocated up front, copying read.
- 2026-08-13 — "H3 is falsified; the cache is a small share of peak." Caused by the LM head running over every prefill position. With that fixed, the cache is 50–58% of peak at large cells and INT8 moves the frontier.
