"""BlockTable: per-sequence logical-to-physical block mapping (FR5, FR9).

This is the actual "paging" data structure: a sequence's KV cache is a
list of logical block indices ``0, 1, 2, ...`` each mapped to some
physical block id in a ``BlockPool`` -- not necessarily contiguous, not
necessarily allocated up front. ``block_ids[i]`` is the physical block
backing logical block ``i`` of this sequence's KV cache.

Growth is lazy and block-granular: ``grow_to(num_tokens)`` allocates only
as many *additional* physical blocks as are needed to cover
``num_tokens``, rounding up to the block boundary -- this is precisely
what lets a paged sequence avoid Phase 1's contiguous baseline's "reserve
prompt_len + max_new_tokens up front" (see
``docs/decisions/0002-contiguous-baseline-simulated-capacity.md``): a
paged sequence only ever holds blocks for tokens it has actually
generated, rounded up to ``block_size``. Core invariant #3 ("a sequence's
logical block count covers its KV length") is exactly
``capacity_tokens >= current_len``, which ``grow_to`` maintains by
construction after every call.
"""

from __future__ import annotations

from minipaged.kv.pool import BlockPool, BlockPoolExhausted


def blocks_needed_for(num_tokens: int, block_size: int) -> int:
    """How many blocks of size ``block_size`` are needed to cover
    ``num_tokens`` tokens (ceiling division). Shared by ``BlockTable``,
    ``PagedKVManager``, and Phase 3's scheduler admission ledger so the
    rounding rule is defined exactly once."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


class BlockTable:
    """One sequence's logical block list, backed by a shared ``BlockPool``."""

    def __init__(self, block_size: int, pool: BlockPool) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self._pool = pool
        self.block_ids: list[int] = []

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def capacity_tokens(self) -> int:
        """Total tokens this table's currently-allocated blocks can hold --
        always ``>=`` the sequence's actual KV length once ``grow_to`` has
        been called for it (core invariant #3)."""
        return self.num_blocks * self.block_size

    def covers(self, num_tokens: int) -> bool:
        return self.capacity_tokens >= num_tokens

    def grow_to(self, num_tokens: int) -> int:
        """Ensure this table has enough blocks to cover ``num_tokens``
        tokens, allocating only the shortfall from the pool. Returns the
        number of newly allocated blocks (0 if already sufficient -- most
        decode ticks, since growth only happens when crossing a
        ``block_size`` boundary).

        All-or-nothing: if the pool runs out partway through a multi-block
        growth, every block newly allocated by *this call* is freed back
        to the pool before ``BlockPoolExhausted`` propagates, so a failed
        growth never leaves the table (or the pool) in a partially-grown,
        inconsistent state."""
        needed = blocks_needed_for(num_tokens, self.block_size)
        delta = needed - self.num_blocks
        if delta <= 0:
            return 0
        newly_allocated: list[int] = []
        try:
            for _ in range(delta):
                newly_allocated.append(self._pool.allocate())
        except BlockPoolExhausted:
            for block_id in newly_allocated:
                self._pool.free(block_id)
            raise
        self.block_ids.extend(newly_allocated)
        return delta

    def free_all(self) -> int:
        """Release every block this table holds back to the pool (sole
        ownership -- Phase 2 has no sharing yet, see ``pool.py``'s
        docstring for why ``free`` rather than ``release`` is used here).
        Returns the number of blocks freed. Idempotent: calling this again
        on an already-empty table is a no-op returning 0."""
        n = len(self.block_ids)
        for block_id in self.block_ids:
            self._pool.free(block_id)
        self.block_ids.clear()
        return n

    def physical_block_for_token(self, token_index: int) -> int:
        """Which physical block id backs logical token ``token_index``
        (0-based) of this sequence's KV cache."""
        if not (0 <= token_index < self.capacity_tokens):
            raise IndexError(
                f"token_index {token_index} out of range for capacity_tokens="
                f"{self.capacity_tokens} ({self.num_blocks} blocks x {self.block_size})"
            )
        return self.block_ids[token_index // self.block_size]
