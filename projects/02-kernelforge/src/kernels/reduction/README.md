# Reduction Ladder (Phase 2)

Sum-reduction of an `n`-element FP32 array to one scalar. Five GPU variants,
each changing exactly one variable from the previous rung (hard constraint
4), all validated against the same CPU reference (`kernelforge::reference::
reduce_sum`, `src/common/reference_ops.cpp` — double-accumulated for
accuracy) before any timing is trusted (hard constraint 1). Every rung's
hypothesis below was written into that rung's `.cuh` header **before**
its `.cu` implementation, per `docs/optimization-method.md`'s loop and
hard constraint 10.

| Rung | File | What changes vs. previous rung | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| V0 `v0_naive_global_atomic` | `reduce_naive_global.cuh/.cu` | — (baseline) | Pure global-memory: no shared memory, no tree reduction. Every thread combines its single element with one global `atomicAdd`. All `n` additions serialize on the same address; this isolates "atomics contention is the bottleneck" as its own measurable mechanism (FR2's literal "naive global-memory" rung). Does not require a power-of-two block size (no tree step). |
| V1 `v1_naive_interleaved` | `reduce_naive_interleaved.cuh/.cu` | Introduces shared memory | Interleaved (`tid % (2*stride)`) shared-memory tree addressing + one global atomicAdd per block to combine, cutting global atomics from `n` down to `ceil(n/block_size)`. Still deliberately naive: warp divergence at every tree step (see the `.cuh` header for a worked-out note on why this specific pattern does *not* actually bank-conflict on this GPU's 32 FP32-word banks, despite the textbook expectation). |
| V2 `v2_sequential_addressing` | `reduce_sequential_addressing.cuh/.cu` | Addressing pattern only (`tid < stride` instead of `tid % (2*stride) == 0`) | Removes warp divergence for all but the last few tree steps; expect a clear win over V1, growing with block size. |
| V3 `v3_warp_shuffle` | `reduce_warp_shuffle.cuh/.cu` | Last-warp finalization only (warp-shuffle instead of shared memory + `__syncthreads()`) | ADR 0004-compliant `__shfl_down_sync` (full mask) replaces the last 5 shared-memory tree steps; expect a modest further win, more visible at larger block sizes / smaller n. |
| V4 `v4_vectorized_coarsened` | `reduce_vectorized_coarsened.cuh/.cu` | Load phase only (`float4` vectorized loads + grid-stride thread coarsening, capped block count) | Wider (128-bit) transactions + fewer/busier threads → fewer blocks → less atomicAdd contention on the shared accumulator; expect the gap to widen with n, possibly regressing at very small n. |

**Held constant across every GPU rung (so it never becomes a hidden second
variable):** the cross-block combine mechanism is a single global
`atomicAdd` from each block (or, for V0, each thread) into a pre-zeroed
scalar accumulator, for every one of V0-V4. This is a real, disclosed
bottleneck at large block counts (all `n/block_size` — or, for V4,
`min(n4/block_size, 4096)` — blocks serialize on one address), included on
purpose as a small-scale preview of the Phase 3 histogram/atomics-contention
lab, not fixed here. A two-pass (reduce-then-reduce) or cooperative-groups
combine that avoids this atomic entirely is explicit future work, not
silently included or omitted.

**Ladder history / naming note:** V1-V4's file and `--variant` identifiers
predate V0 and are kept unchanged (rather than renumbered to v1-v5) so the
already-committed, benchmarked `benchmarks/raw/reduction.jsonl` records for
those four stay traceable to the exact code that produced them (spec
§1.7: "every claimed speedup traceable to raw committed results"). V0 was
added afterward to close a real gap: the original four-rung ladder's V1
already combined FR2's "naive global-memory" and "shared-memory" stages
into one kernel (it used shared memory from the start), so there was no
preserved baseline demonstrating pure global-memory atomics contention in
isolation. V0 is that baseline. Its benchmark numbers are collected in a
separate pass (hard constraint 9: separate implementation commits from
benchmark-result commits) and are not yet in `benchmarks/raw/reduction.jsonl`
or methodology.md §9.1 as of this rung's introduction.

## Correctness

`tests/test_reduction.cpp` checks all five variants against
`reference::reduce_sum` on: n = 0 (empty; identity = 0), n = 1, small
non-power-of-two n, n exactly one block, n one more/less than a block
multiple, a large multi-block n, and a block-size sweep (plus a
non-power-of-two block-size case specific to V0, which — unlike V1-V4 —
does not require one). See that file for the exact cases and tolerances
used.

## Benchmarks

`benchmarks/configs/reduction.json` sweeps all five variants across the
same size list; results land in `benchmarks/raw/reduction.jsonl`. See
`benchmarks/methodology.md` §9 for the as-run numbers (filled in only
after `scripts/run_all_benchmarks.sh` has actually produced them — hard
constraint 3: never fabricate a benchmark number). As of V0's introduction,
its numbers are still pending a dedicated benchmark-collection pass (see
"Ladder history" above); §9.1 covers V1-V4 only until then.
