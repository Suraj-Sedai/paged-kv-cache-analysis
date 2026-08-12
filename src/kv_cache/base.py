"""Batched KV cache interface.

The batch is the unit of operation. Every call touches all B sequences at once,
so a forward pass costs O(n_layers) cache ops instead of O(B * n_layers) — the
per-sequence Python loop was the dominant cost in the v1 harness and made decode
launch-bound, which is what made every timing column unreadable.

Shape contract:
    write(layer_idx, k, v)   k, v: [B, H, T, Hd]
    read(layer_idx)       -> k, v: [B, H, n, Hd]   where n = filled length

Lengths are exposed separately rather than returned from read(), so the ragged
(variable-length) harness can pad to max and mask without a signature change.
"""
from abc import ABC, abstractmethod


class BaseKVCache(ABC):
    @abstractmethod
    def write(self, layer_idx, k, v):
        """Append k, v ([B, H, T, Hd]) at the current logical position."""

    @abstractmethod
    def read(self, layer_idx):
        """Return (k, v) as [B, H, n, Hd] covering all tokens written so far."""

    @abstractmethod
    def advance(self, amount=1):
        """Advance the logical length of every sequence by `amount`."""

    @abstractmethod
    def free(self):
        """Release all sequences (end of generation)."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def memory_bytes(self) -> int:
        pass

    @abstractmethod
    def fragmentation_ratio(self) -> float:
        """Internal fragmentation: allocated-but-unwritten fraction of this
        cache's own allocation, summed over the batch.

            (sum_b allocated_b - sum_b used_b) / sum_b allocated_b

        used_b is the WRITTEN extent (`lengths`), not the logical length from
        advance() — allocation is driven by write(), so both sides of the ratio
        track the same counter. Summing over the batch (rather than reporting
        sequence 0) makes this well-defined for ragged batches.

        Subclasses differ only in what counts as allocated_b: the up-front
        reservation for contiguous, pages actually held for paged. The
        `alloc_basis` CSV column labels which basis a row reports, so rows are
        never read as a like-for-like comparison across cache types.

        The zero guard is on ALLOCATION, not on seq_len: a cache holding nothing
        has nothing to waste (paged, 0.0), while a cache holding an unwritten
        reservation wastes all of it (contiguous, 1.0). Guarding on seq_len would
        cross counters and report 0.0 mid-forward, after write() has allocated
        and filled but before advance() has moved the cursor.

        SCOPE: internal fragmentation only. Pool blocks never handed to any
        sequence are over-provisioning, not fragmentation, and are excluded.
        """

    @property
    @abstractmethod
    def lengths(self):
        """Per-sequence written extent. Uniform today; ragged later."""
