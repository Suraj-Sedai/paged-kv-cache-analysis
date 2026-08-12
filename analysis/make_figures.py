"""Regenerate the paper figures from the result CSVs.

Reproducible: `python analysis/make_figures.py` reads experiments/results/*.csv and
writes paper/figures/*.{pdf,png}. PDF is for the camera-ready, PNG for quick preview.
Four figures, mapping to the paper:
  fig1_block_throughput  - block-size optimum (H2, headline)   [Exp 2]
  fig2_fragmentation     - internal waste vs block size (H2)   [Exp 2]
  fig3_layout_throughput - contiguous vs paged (H1 overhead)   [Exp 1]
  fig4_precision         - INT8 memory / cache share / quality [Exp 3]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM = ["ttft_ms", "tpot_ms", "throughput", "throughput_cv",
       "peak_memory_mb", "cache_memory_mb", "frag_ratio"]

# Mirrors DECODE_LEN in the benchmark scripts. Needed because frag_ratio is now
# waste/allocation, so recovering absolute wasted slots requires the allocation,
# and allocation depends on the final length (seq_len + DECODE_LEN). Keep in sync.
DECODE_LEN = 60


def find_results_dir():
    for base in [Path.cwd(), *Path.cwd().parents]:
        cand = base / "experiments" / "results"
        if cand.exists():
            return cand
    sys.exit("experiments/results not found")


def load(results, name):
    df = pd.read_csv(results / name)
    for c in NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ok"] = df["status"].astype(str).str.lower().eq("ok")
    return df


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def fig1_block_throughput(exp2, outdir):
    ok = exp2[exp2.ok]
    blocks = sorted(ok.block_size.unique())
    piv = ok.pivot_table(index=["seq_len", "batch"], columns="block_size", values="throughput")
    norm = piv.div(piv.max(axis=1), axis=0)            # 1.0 = best block in that cell
    mean, std = norm.mean(), norm.std()
    best = int(mean.idxmax())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar([str(b) for b in mean.index], mean.values, yerr=std.values,
                marker="o", capsize=4, color="#d62728", linewidth=2)
    ax.axvline(list(mean.index).index(best), color="gray", ls=":", alpha=0.7)
    ax.set(xlabel="block size", ylabel="throughput / best-in-cell",
           title=f"Throughput optimum at block_size={best}")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    save(fig, outdir, "fig1_block_throughput")


def fig2_fragmentation(exp2, outdir):
    """Paged-only, single series (Exp 2 sweeps block size on the paged cache).

    TODO (Prompt 3, trace harness): there is deliberately NO contiguous series
    here. Under oracle reservation (max_seq_len == final length) contiguous
    fragmentation is 0.0 by construction, so a contiguous-vs-paged comparison
    would be vacuous, not favourable. Add it only once ReservationPolicy
    (Oracle / FixedMax / Percentile) makes the reservation a real variable.
    """
    ok = exp2[exp2.ok].copy()
    # frag_ratio is waste/allocation (not waste/block), so recover absolute
    # wasted slots by multiplying through the allocation, not the block size.
    final_len = ok["seq_len"] + DECODE_LEN
    n_pages = np.ceil(final_len / ok["block_size"])
    ok["wasted_slots"] = ok["frag_ratio"] * n_pages * ok["block_size"]
    blocks = sorted(ok.block_size.unique())
    waste = ok.groupby("block_size")["wasted_slots"].agg(["mean", "std"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(blocks)))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([str(b) for b in waste.index], waste["mean"], yerr=waste["std"],
           capsize=4, color=colors, edgecolor="black")
    ax.plot(range(len(blocks)), [b / 2 for b in blocks], "k--", marker="o",
            label="~block_size / 2 (uniform expectation)")
    ax.set(xlabel="block size", ylabel="wasted slots per sequence",
           title="Internal fragmentation grows with block size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, outdir, "fig2_fragmentation")


def fig3_layout_throughput(exp1, outdir):
    ok = exp1[exp1.ok]
    batches = [1, 8, 16]
    fig, axes = plt.subplots(1, len(batches), figsize=(12, 3.8), sharey=True)
    for ax, batch in zip(axes, batches):
        for ctype, color in [("contiguous", "#1f77b4"), ("paged", "#ff7f0e")]:
            sub = ok[(ok.cache_type == ctype) & (ok.batch == batch)].sort_values("seq_len")
            ax.plot(sub.seq_len, sub.throughput, marker="o", label=ctype, color=color)
        ax.set(xlabel="sequence length", title=f"batch {batch}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("throughput (tok/s)")
    axes[0].legend()
    fig.suptitle("Contiguous vs paged throughput (paged indirection overhead)")
    fig.tight_layout()
    save(fig, outdir, "fig3_layout_throughput")


def fig4_precision(exp3, ppl, outdir):
    fp16 = exp3[exp3.cache_type == "paged_fp16"]
    int8 = exp3[exp3.cache_type == "paged_int8"]
    m = fp16.merge(int8, on=["seq_len", "batch"], suffixes=("_fp16", "_int8"))
    expected = (1 + 2 / 64) / 2

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) memory ratio
    a1.scatter(m.cache_memory_mb_fp16, m.cache_memory_mb_int8, color="#1f77b4")
    lim = m.cache_memory_mb_fp16.max() * 1.05
    a1.plot([0, lim], [0, lim], "k--", alpha=0.4, label="1:1")
    a1.plot([0, lim], [0, lim * expected], "r-", label=f"{expected:.3f}x")
    a1.set(xlabel="FP16 cache (MB)", ylabel="INT8 cache (MB)", title="Cache memory: INT8 vs FP16")
    a1.legend()
    a1.grid(alpha=0.3)

    # (b) cache share of peak
    okf = fp16[fp16.ok].copy()
    okf["share"] = 100 * okf.cache_memory_mb / okf.peak_memory_mb
    share = okf.groupby("seq_len")["share"].mean()
    a2.bar([str(s) for s in share.index], share.values, color="#2ca02c", edgecolor="black")
    a2.set(xlabel="sequence length", ylabel="cache % of peak memory",
           title="Cache is a small share of peak")
    a2.grid(axis="y", alpha=0.3)

    # (c) perplexity cost
    a3.bar(ppl.model, ppl.ppl_pct_increase, color="#9467bd", edgecolor="black")
    a3.set(xlabel="model", ylabel="perplexity increase (%)",
           title="INT8 quality cost (WikiText-2)")
    a3.grid(axis="y", alpha=0.3)

    fig.suptitle("INT8 KV: free quality, 0.516x memory, but cache is not the bottleneck")
    fig.tight_layout()
    save(fig, outdir, "fig4_precision")


def main():
    results = find_results_dir()
    outdir = results.parent.parent / "paper" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results: {results}\nfigures: {outdir}")

    exp1 = load(results, "exp1_layout_comparison.csv")
    exp2 = load(results, "exp2_block_size_sweep.csv")
    exp3 = load(results, "exp3_precision_sweep.csv")
    ppl = pd.read_csv(results / "exp3_perplexity.csv")

    fig1_block_throughput(exp2, outdir)
    fig2_fragmentation(exp2, outdir)
    fig3_layout_throughput(exp1, outdir)
    fig4_precision(exp3, ppl, outdir)
    print("done.")


if __name__ == "__main__":
    main()
