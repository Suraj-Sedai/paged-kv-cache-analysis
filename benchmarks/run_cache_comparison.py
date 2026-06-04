import csv
import os
import statistics

import torch

from src.inference.controller import InferenceController
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

SEQ_LENS    = [128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 4, 8, 16]
BLOCK_SIZE  = 16
DECODE_LEN  = 60
N_WARMUP    = 3
N_MEASURE   = 5

OUTPUT_PATH = os.path.join("experiments", "results", "exp1_layout_comparison2.csv")
FIELDNAMES  = ["cache_type", "seq_len", "batch", "block_size",
               "ttft_ms", "tpot_ms", "throughput", "memory_mb", "frag_ratio"]



def make_cache(cache_type, model_config, batch_size, max_seq_len, device, io_dtype):
    """Both caches now implement the per-seq_id BaseKVCache interface that
    attention/controller drive directly (write/read/advance/free keyed by the
    batch index), so no adapter is needed.

    Contiguous matches the model dtype so attention's float matmuls line up.
    Paged is fixed to float16 page storage (its live read tensors carry the
    model dtype), and is given a shared pool sized for all `batch_size` seqs.
    """
    if cache_type == "contiguous":
        return ContiguousKVCache(model_config, max_seq_len, device, dtype=io_dtype)
    per_seq_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return PagedKVCache(
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        dim_head=model_config.head_dim,
        block_size=BLOCK_SIZE,
        num_blocks=per_seq_blocks * batch_size,  # shared pool across all seqs
        device=device,
        dtype=torch.float16,
    )


def theoretical_frag(cache_type, length, max_seq_len):
    """Fragmentation at a given filled length, mirroring each cache's own
    fragmentation_ratio(). Computed analytically because controller.generate()
    frees every seq during cleanup, so the cache is empty by the time we report.
    """
    if cache_type == "contiguous":
        return (max_seq_len - length) / max_seq_len  # reserved-but-unused waste
    rem = length % BLOCK_SIZE                          # last-page internal waste
    return 0.0 if rem == 0 else (BLOCK_SIZE - rem) / BLOCK_SIZE


def run_one(model, model_config, cache, prompt_ids):
    controller = InferenceController(model, model_config)
    return controller.generate(prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []

    for cache_type in ["contiguous", "paged"]:
        for seq_len in SEQ_LENS:
            for batch in BATCH_SIZES:
                max_seq_len = seq_len + DECODE_LEN
                model_config = ModelConfig(max_seq_len=max_seq_len)
                model = GPTModel(model_config).to(device)
                io_dtype = next(model.parameters()).dtype
                prompt_ids = torch.randint(
                    0, model_config.vocab_size, (batch, seq_len), device=device
                )

                try:
                    for _ in range(N_WARMUP):
                        cache = make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype)
                        run_one(model, model_config, cache, prompt_ids)

                    metrics_list = []
                    for _ in range(N_MEASURE):
                        cache = make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype)
                        result = run_one(model, model_config, cache, prompt_ids)
                        metrics_list.append(result.metrics)

                    row = {
                        "cache_type": cache_type,
                        "seq_len":    seq_len,
                        "batch":      batch,
                        "block_size": BLOCK_SIZE if cache_type == "paged" else "N/A",
                        "ttft_ms":    round(statistics.median(m.ttft_ms for m in metrics_list), 3),
                        "tpot_ms":    round(statistics.median(m.tpot_avg_ms for m in metrics_list), 3),
                        "throughput": round(statistics.median(m.throughput_tokens_per_sec for m in metrics_list), 1),
                        "memory_mb":  round(statistics.median(m.peak_memory_mb for m in metrics_list), 2),
                        "frag_ratio": round(theoretical_frag(cache_type, seq_len + DECODE_LEN, max_seq_len), 4),
                    }

                except torch.cuda.OutOfMemoryError:
                    row = {
                        "cache_type": cache_type, "seq_len": seq_len, "batch": batch,
                        "block_size": BLOCK_SIZE if cache_type == "paged" else "N/A",
                        "ttft_ms": "OOM", "tpot_ms": "OOM", "throughput": "OOM",
                        "memory_mb": "OOM", "frag_ratio": "OOM",
                    }

                results.append(row)
                print(row)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
