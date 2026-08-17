# 0002 — Contiguous baseline: simulated capacity (D2)

- **Status:** Accepted
- **Phase:** 1
- **Related spec sections:** S1.10 D2 ("How faithfully physical KV tensors
  follow block tables in MVP" → default: simulated pages first, document
  gap), S1.6

## Context

Phase 1's exit criterion is "a variable-length trace demonstrates
fragmentation/waste numerically." To show that, some notion of "capacity"
and "reservation" is needed — but per Decision 0001, there is no real
model and no real KV tensor yet, so there is nothing to measure a real
per-token KV footprint (`layers x heads x head_dim x 2 x dtype_bytes`)
against.

## Decision

`ContiguousMemoryModel` (`minipaged.kv.contiguous`) measures capacity in
abstract "token slots" — one slot conceptually holds one token's K/V
vectors across all layers/heads, for a toy model where that constant
happens to be exactly 1. A running sequence reserves
`prompt_len + max_new_tokens` slots up front (the only size a *contiguous*
allocator can safely commit to, since it cannot grow a live span in place
without risking collision with a neighboring reservation — there are no
gaps between reservations by construction). `used` tracks the sequence's
actual current KV length (`prompt_len + tokens_generated`).
`reserved - used` is waste: capacity the allocator holds but nothing is
using, and nobody else can borrow either, since a contiguous span cannot
be subdivided.

This is *not* the same claim as "this many real GPU bytes are wasted" —
it is "this many token-slot-equivalents would be wasted, holding the
per-token KV footprint constant." Converting to real bytes requires a
real model's `layers x heads x head_dim x 2 x dtype_bytes` constant, which
does not exist until Phase 5.

## Consequences

- `ContiguousMemoryMetrics.waste` / `waste_ratio` are directly comparable
  *within* Phase 1-2 (contiguous vs. paged, same trace, same
  `capacity_tokens`), which is exactly what Phase 2's exit criterion asks
  for ("same trace runs paged and produces utilization statistics").
- They are *not* directly comparable to a real deployment's memory usage
  in GB until Phase 5-7 supply the real per-token-per-layer constant. Any
  benchmark report (Phase 7) that presents Phase 1-2 numbers as
  real-hardware memory savings without that conversion would violate
  agent execution rule #4.
- `CapacityError` (a request whose worst-case reservation exceeds *total*
  capacity) is distinct from a normal, retryable admission rejection
  (insufficient *free* capacity right now) — see
  `ContiguousMemoryModel.try_reserve`'s docstring. This maps directly onto
  D7's default (reject at admission for MVP; preemption is stretch): an
  unrecoverable request is rejected immediately rather than queued
  forever, while a temporarily-too-large request stays queued and is
  retried every tick.
