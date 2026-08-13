"""Which SDPA backend is actually running, and does it change peak memory?

Motivation: exp1 shows the KV cache at ~13% of peak memory even after the SDPA
switch, with ~3.25 GB unaccounted for at seq 2048 / batch 16. A [16,8,2048,2048]
fp16 score matrix is 1.07 GB. If SDPA is falling back to the MATH backend on this
GPU (Blackwell / sm_120), the score matrix is still being materialised and the
switch bought nothing on memory -- which is H3's precondition.

Run:  python -m benchmarks.sdpa_backend_check
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B, H, T, HD = 16, 8, 2048, 64
DEV = "cuda"


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        fn()
    except Exception as e:
        return None, type(e).__name__ + ": " + str(e)[:90]
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1024**2, None


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    print(f"device: {torch.cuda.get_device_name()}   torch {torch.__version__}")
    print(f"shape:  [B={B}, H={H}, T={T}, Hd={HD}] fp16")
    print(f"a materialised [B,H,T,T] score matrix would be "
          f"{B*H*T*T*2/1024**3:.2f} GB\n")

    q, k, v = (torch.randn(B, H, T, HD, device=DEV, dtype=torch.float16) for _ in range(3))

    def run(backend=None):
        def inner():
            with torch.no_grad():
                if backend is None:
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    with sdpa_kernel(backend):
                        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return inner

    results = {}
    for label, backend in [
        ("default (what exp1 used)", None),
        ("MATH", SDPBackend.MATH),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ]:
        mb, err = peak_mb(run(backend))
        results[label] = mb
        print(f"  {label:<26} {'unavailable — ' + err if mb is None else f'{mb:8.1f} MB'}")

    d, m = results["default (what exp1 used)"], results["MATH"]
    print()
    if d is not None and m is not None:
        if abs(d - m) / m < 0.05:
            print("VERDICT: default matches MATH -> the score matrix is still being")
            print("materialised. The SDPA switch did NOT change the memory ledger,")
            print("and H3's precondition is still broken.")
        else:
            print(f"VERDICT: default is {m/d:.1f}x cheaper than MATH -> a memory-efficient")
            print("backend is running. The ~3.25 GB gap in exp1 has another cause.")


if __name__ == "__main__":
    main()
