# Scan (Prefix Sum) Strategies (Phase 2)

Inclusive prefix sum of an `n`-element FP32 array. Two GPU strategies for
the per-block scan algorithm, sharing one multi-block orchestration
(`scan_common.cuh/.cu`) so an n larger than one block is handled
identically regardless of which strategy is used — the block-level
algorithm is the only variable being compared. Both validated against the
same CPU reference (`kernelforge::reference::inclusive_scan`,
`src/common/reference_ops.cpp`) before any timing is trusted.

| Strategy | File | Per-block algorithm | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| A: Hillis-Steele | `scan_hillis_steele.cuh/.cu` | Naive, step-efficient: every thread active every pass, double-buffered shared memory, `log2(block_size)` passes | O(n log n) work but simple, uniform, fewer sync points; may still win at GPU block-size scale. |
| B: Blelloch | `scan_blelloch.cuh/.cu` | Work-efficient: up-sweep (reduce) then down-sweep (distribute), `2*log2(block_size)` passes | O(n) work but more phases and a shrinking/growing active-thread count; the classic work-efficiency-vs-step-efficiency tradeoff. |

Both produce an **inclusive** scan (Blelloch's natural exclusive result is
converted with one extra add — see `scan_blelloch.cuh`), so both strategies
are directly comparable output-for-output against one reference and one
test file.

## Multi-block orchestration (shared, strategy-independent)

`scan_common.cuh/.cu` implements the standard "scan, scan-of-sums,
add-offsets" 3-pass decomposition: each block scans its own chunk and
records its total; the (small) array of block totals is itself scanned in
one block (by calling the *same* per-block kernel a second time); each
block's resulting exclusive carry-in is broadcast across its chunk. This
is shared plumbing (not part of either strategy's distinguishing
algorithm), the same way `apps/bench_cli.hpp` is shared across every
`bench_*_main.cpp` — see that file's header comment for the full
walk-through.

**Known, disclosed scope limit:** this supports n up to `block_size^2`
elements (block_size = 1024 → up to 1,048,576) because the level-2
scan-of-sums itself needs to fit in one block. Beyond that, a 3rd scan
level would be required — explicit future work, not silently worked
around: `scan_multilevel_launch` throws `std::runtime_error` rather than
truncating or producing a wrong answer (hard constraint 8).

## Correctness

`tests/test_scan.cpp` checks both strategies against
`reference::inclusive_scan` on: n = 0 (empty), n = 1, sizes within one
block (including non-power-of-two), sizes exactly at a block boundary,
sizes spanning multiple blocks (non-block-multiple), a size at the
2-level ceiling, and an explicit test that exceeding the ceiling throws
rather than silently misbehaving.

## Benchmarks

`benchmarks/configs/scan.json` sweeps both strategies across the same
size list (all `<= block_size^2` for the block size used); results land
in `benchmarks/raw/scan.jsonl`. See `benchmarks/methodology.md` §9 for the
as-run numbers (effective bandwidth only -- see below for `gflops`).

**`gflops` note:** `apps/bench_scan_main.cpp`'s `total_scan_ops` computes
each record's `gflops` from the strategy's *actual* per-block addition
count (Hillis-Steele: `B*log2(B) - (B-1)` per block; Blelloch: `2*(B-1)`
per block, work-efficient), not a single shared `flops = n` approximation
-- the two strategies are not equivalent in op count, so reusing one
formula for both understates/overstates one of them. The `gflops` values
already committed in `benchmarks/raw/scan.jsonl` predate this fix and were
computed with the old shared formula; `effective_bandwidth_gb_s` (§9.2's
metric) is unaffected, since it depends only on bytes moved, not op count.
Refreshing `gflops` in the committed file is deferred to the next
benchmark-collection pass (hard constraint 9: separate implementation
commits from benchmark-result commits).
