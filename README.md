# Measuring the Wrong Ledger

Where KV cache memory goes during transformer inference, measured on one consumer GPU, and four measurement bugs that each gave me a confident wrong answer first.


**Main point:** a memory optimization only pays off when the thing it shrinks is a large share of memory. INT8 halves the KV cache in every workload here, but that cuts peak memory by anywhere from 2.7% to 26%. The technique doesn't change. The cache's share of memory does.

## INT8 is worth as much as the cache's share of memory

| seq_len | batch | Cache share of peak | INT8 cuts peak by |
|--:|--:|--:|--:|
| 512 | 1 | 5.2% | 2.7% |
| 1024 | 8 | 35.5% | 15.9% |
| 2048 | 16 | 50.8% | 22.8% |
| 8192 | 32 | 58.0% | 26.0% |

The INT8 cache is 48.4% smaller in every row. Under a fixed memory budget, 36% more tokens fit (about 303k → 412k at 8 GB). On real GPT-2 weights, perplexity changes by less than 0.01%, but 1.2–1.5% of next-token predictions flip.

## A per-token memory model

Peak memory depends only on total tokens (batch × seq_len). Each token costs:

```
bytes per token = ( 2 · n_layers · d_model · c   +   3 · d_model + 2 · d_ff ) × bytes per element
                    └──────── KV cache ────────┘     └──── activations ────┘
```

The cache is kept for every layer. Activations are not: each layer's temporaries are freed before the next layer runs, so the activation term stays flat as the model gets deeper (11.01, 11.00, 11.04 KB/token at 4, 8, 16 layers). `c = 1040/1024` accounts for decode tokens and 16-token pages.

Built from five model shapes, the formula predicts GPT-2 small, a shape it never saw, at 53.06 KB/token. Measured: 53.08 (0.03% off). Worst error across all six shapes: 0.08%.

## Four bugs, four wrong conclusions

None raised an error. Each returned a believable number.

| Bug | What I wrongly concluded | How to spot it in your own setup |
|---|---|---|
| LM head ran on every prefill position, not just the last | KV cache is at most 14% of peak, so INT8 can't help | Measured peak is bigger than your hand-computed sum of tensors |
| Attention looped over the batch in Python | Decode is launch-bound on this GPU | Latency barely moves across 16× changes in batch and seq_len |
| Contiguous cache reserved exactly the final length | Contiguous never fragments | Fragmentation is exactly 0 in every cell |
| Windows driver spilled to host RAM instead of failing | This GPU never runs out of memory | 14,009 MB "allocated" on an 11.9 GB card |

## What didn't hold up

- **Paged beats contiguous somewhere.** Not in any of 20 cells: paged throughput is 0.32–0.77× contiguous, and peak memory is within 0.9%. This setup can't show paging's benefit, because all sequences are the same length and the page pool is allocated at full size up front.
- **Contiguous fragments less than paged.** Only because of bug 3.
- **Small block sizes are fastest.** Two runs of the same config gave 227 and 495 tok/s. Too noisy to rank.

## Setup and limits

- Study model: 8 layers, d_model 512, d_ff 2048, FP16, random weights. Random weights are fine for memory, which depends on tensor shapes, not values. Quality is measured on real GPT-2 and GPT-2-medium weights (WikiText-2, 284,614 tokens).
- One GPU: RTX 5070 Ti Laptop, 11.9 GB, Windows. Memory budgets are enforced in software.
- Standard multi-head attention and MLP only. Grouped-query attention changes the cache term and SwiGLU changes the activation term, so Llama-style models need the coefficients re-derived.
- The paged cache copies blocks into one tensor before attention (no fused kernel), so its speed is not the speed of paging in a real server.
- No latency claims. At this model size a decode step is mostly kernel-launch overhead.

## Reproduce

Needs an NVIDIA GPU. Install PyTorch with CUDA first ([pytorch.org](https://pytorch.org/get-started/locally/); tested on 2.11 with CUDA 12.8), then:

```
pip install -r requirements.txt
python -m pytest -q        # 56 tests; paged and contiguous must produce identical tokens
```

| Command | Writes to `experiments/results/` |
|---|---|
| `python -m benchmarks.arch_sweep` | `arch_sweep.json` (memory model) |
| `python -m benchmarks.run_precision_sweep` | `exp3_precision_sweep.csv` (FP16 vs INT8) |
| `python -m benchmarks.oom_frontier` | `oom_frontier.json` (memory budgets) |
| `python -m benchmarks.run_cache_comparison` | `exp1_layout_comparison.csv` (paged vs contiguous) |
| `python -m benchmarks.run_block_size_sweep` | `exp2_block_size_sweep.csv` |
| `python -m benchmarks.run_perplexity_eval --model gpt2 --text experiments/data/wikitext2_test.txt` | `exp3_perplexity.csv` |

Notebooks 04 and 05 work through the memory model. [STATUS.md](STATUS.md) lists which files are current and what is still open.
