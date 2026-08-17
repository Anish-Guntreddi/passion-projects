# 0003 — Block size: 16 tokens/block (D1)

- **Status:** Accepted
- **Phase:** 2
- **Related spec sections:** S1.10 D1 ("Block size (tokens per block;
  relation to layers/heads)" → default: 16 tokens/block; ADR), FR5, FR9

## Context

Phase 2's paged allocator (`kv/{block,pool,table,manager}.py`) needs a
fixed number of tokens per physical block before any allocation code can
run. Real serving systems pick this by balancing two costs that both grow
with block size:

- **Internal fragmentation** (the thing Phase 2 exists to reduce, see
  `docs/kv-memory.md`): a sequence's last, partially-filled block wastes
  up to `block_size - 1` tokens on average. Smaller blocks waste less.
- **Bookkeeping/lookup overhead**: more, smaller blocks means a longer
  `BlockTable.block_ids` list per sequence and more individual
  `BlockPool.allocate()`/`free()` calls over a sequence's lifetime.
  Larger blocks amortize this. In a real system, block size also
  interacts with attention-kernel memory-access patterns (vLLM's default
  is 16, chosen for GPU memory-coalescing reasons) — out of scope for
  this simulated-capacity phase (Decision 0002 / D2), but the *name* of
  the constant is chosen to match that precedent rather than invent an
  unrelated one.

Per D2 (simulated capacity first, no real model until Phase 5), there is
no real attention kernel to tune against yet — this decision is about the
fragmentation/overhead tradeoff only, made with synthetic-trace evidence.

## Decision

`block_size = 16` tokens, the spec's stated default, applied uniformly
(`PagedKVManager(num_blocks, block_size=16)`,
`PagedKVEngine(num_blocks, block_size=16, ...)` — both default to it,
overridable per instance for tests that want tighter control, e.g.
`tests/unit/test_manager.py` uses `block_size=1` and `block_size=4` in
several tests specifically to make block-boundary-crossing behavior exact
and easy to reason about by hand).

Evidence this default is reasonable for this project's synthetic
workloads (`examples/phase2_paged_kv_demo.py`, prompt lengths 4-32,
`max_new_tokens` 8-96, 400-token-slot total capacity, seed 7 — identical
trace and capacity Phase 1's fragmentation demo uses): peak waste ratio
over the full replay is **31.5%** for the paged allocator versus **63.7%**
for Phase 1's contiguous baseline on the same trace — internal
(block-rounding) fragmentation at `block_size=16` is well under half of
what worst-case-reservation fragmentation was. `blocks_needed_for`
(`kv/table.py`) is the single function computing the rounding for every
caller (`BlockTable`, `PagedKVManager`, and Phase 3's admission ledger),
so this constant only needs to be right in one place.

## Consequences

- A sequence's last block is, on average, half-empty
  (`block_size / 2 = 8` tokens of unavoidable rounding waste per active
  sequence) — this is `PagedKVMetrics.waste`'s floor, not a bug; Decision
  0004 covers the *other*, avoidable source of paged waste (optimistic
  admission).
- Changing `block_size` is a single-parameter experiment (both
  `PagedKVManager` and `PagedKVEngine` accept it), useful for a Phase 7
  benchmark-study sweep (fragmentation vs. block size) without any other
  code change.
- This value is not tuned against a real GPU/attention-kernel memory
  layout — per Decision 0001/0002, that constant does not exist until
  Phase 5. If Phase 5+ integration reveals a different real-hardware
  sweet spot, this ADR's default should be revisited then, not assumed to
  transfer.
