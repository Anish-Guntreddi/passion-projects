# Case Study 1: Transpose Naive → Tiled — Coalescing Confirmed, Occupancy Ruled Out, a Bank Conflict Found and Left Unfixed on Purpose

**Kernels:** `transpose_naive_kernel` (V1) vs. `transpose_tiled_kernel`
(V2) — `src/kernels/transpose/{transpose_naive,transpose_tiled}.cu`.
**Evidence:** `benchmarks/raw/transpose.jsonl` (Phase 1, already
committed), `profiling/occupancy/transpose.jsonl` (Phase 6),
`profiling/ptx-sass/transpose.sass.txt` (Phase 6),
`profiling/nsight-systems/01-transpose-{v1-naive,v2-tiled}.*` (Phase 6).

## 1. Hypothesis (written in Phase 1, before this profiling pass — `transpose_naive.cu`/`transpose_tiled.cuh`)

V1 reads `in` with a coalesced access pattern (consecutive `threadIdx.x`
→ consecutive input column → contiguous address) but writes `out` with a
pathological one (consecutive `threadIdx.x` → consecutive output *row* →
addresses `rows` floats apart, one memory transaction per thread instead
of one per warp). V2 changes exactly one thing: it stages each 32x32 tile
through shared memory so BOTH the global read and the global write are
coalesced. `transpose_tiled.cuh` additionally documents, **in advance and
un-fixed**, a second, separate prediction: reading the shared-memory tile
back out transposed (`tile[threadIdx.x][threadIdx.y]`) should cause a
shared-memory bank conflict, because the tile is deliberately *not*
padded — that fix (padding) is out of scope for the ladder as currently
implemented (V4 in the spec's ladder, not yet built). Phase 6's job here
is to check **both** predictions against real evidence: the coalescing
claim (already measured, Phase 1) against occupancy and instruction-level
evidence, and the bank-conflict claim (never previously measured at all)
against SASS.

## 2. Evidence before / after — wall-clock (Phase 1, `benchmarks/raw/transpose.jsonl`, 8192x8192, recomputed here to full precision)

| variant | median (ms) | effective BW (GB/s) | % of 1008.1 GB/s peak |
|---|---:|---:|---:|
| v1_naive | 1.9712 | 272.357 | 27.02% |
| v2_tiled | 0.757568 | 708.677 | 70.30% |

**v2_tiled is 2.60201x faster** (time ratio 1.9712/0.757568; bandwidth
ratio 708.677/272.357 agrees to 5 significant figures, as it must — both
are computed from the same two numbers). This confirms the Phase 1
finding this case study re-examines with new evidence, not a new
benchmark run.

## 3. New evidence — theoretical occupancy (`profiling/occupancy/transpose.jsonl`, both queried at their fixed 32x32=1024-thread block shape)

| variant | regs/thread | static shared/block | max active blocks/SM | achieved threads/SM | occupancy fraction |
|---|---:|---:|---:|---:|---:|
| v1_naive | 12 | 0 B | 1 | 1024 | 66.7% |
| v2_tiled | 14 | 4096 B | 1 | 1024 | 66.7% |

**Occupancy is identical between the two variants** — both are capped at
exactly 1 resident block/SM (1024 threads is already 66.7% of this GPU's
1536-thread/SM limit, and a 2nd block would need 2048 > 1536), regardless
of v2_tiled's extra shared-memory footprint (4096 B is nowhere near this
GPU's 49,152 B/block budget) or its 2 extra registers/thread. **This
rules out occupancy as the explanation for the 2.60x speedup** — both
kernels have exactly the same number of warps resident and eligible to
hide latency at any given moment. The grid itself is identical too
(65536 blocks either way, 100% grid utilization, since both variants use
the same 32x32 tile/launch geometry). Whatever explains the speedup has
to be visible somewhere other than "more concurrency" — which is exactly
what the coalescing hypothesis predicts (fewer, larger memory
transactions per warp, not more warps in flight).

## 4. New evidence — SASS (`profiling/ptx-sass/transpose.sass.txt`)

**V1 (`transpose_naive_kernel`)** compiles to exactly one global load and
one global store, no shared-memory instructions at all:
```
LDG.E.CONSTANT R7, [R6.64] ;                 // read: in[row*cols + col]
IMAD.WIDE R2, R4, c[0x0][0x170], R2 ;        // address = col*rows + row  (R4 = col, c[...][0x170] = rows)
STG.E [R4.64], R7 ;                          // write: out[col*rows + row] -- the pathological stride
```
The store address is built from `col * rows`, confirming in the compiled
code (not just the source) that consecutive threads' *store* addresses
are `rows` floats apart — the mechanism the hypothesis names.

**V2 (`transpose_tiled_kernel`)** compiles to one shared store, one
`BAR.SYNC`, one shared load, then one coalesced global load + coalesced
global store:
```
IMAD R11, R6, 0x20, R9 ;      // address = threadIdx.y*32 + threadIdx.x  (R6=ty, R9=tx)
STS  [R11.X4], R4 ;           // write into shared tile -- consecutive tx -> consecutive address: no conflict
BAR.SYNC.DEFER_BLOCKING 0x0 ;
IMAD R9, R9, 0x20, R6 ;       // address = threadIdx.x*32 + threadIdx.y  (the TRANSPOSED read)
LDS  R9, [R9.X4] ;            // read back out of shared tile
...
STG.E [R4.64], R9 ;           // write: out[out_row*rows + out_col] -- now coalesced (see source)
```
This confirms the global write is now built from `out_row`/`out_col` in
the coalesced order (source-level: `out[out_row * rows + out_col]` with
`out_col` varying with `threadIdx.x`) — the one-variable coalescing
change the hypothesis describes, now confirmed at the instruction level,
not just by reading the source.

**The bank-conflict prediction is also confirmed, precisely, by the same
`LDS` instruction's address computation.** The transposed read's address
is `(threadIdx.x * 32 + threadIdx.y) * 4` bytes. For one warp (32
consecutive `threadIdx.x` values, `threadIdx.y` fixed — `threadIdx.x` is
the fastest-varying axis within a warp for this 32x32 block layout), the
per-lane address stride is `32 * 4 = 128` bytes. This GPU's shared memory
has 32 banks of 4 bytes each, cycling every `32*4 = 128` bytes — so
**every one of the 32 lanes' addresses maps to bank
`(threadIdx.x*32 + threadIdx.y) mod 32 = threadIdx.y`, the SAME bank for
the entire warp: a 32-way bank conflict on this one `LDS`, on every tile,
every launch.** This is exactly what `transpose_tiled.cuh`'s comment
predicted before this SASS evidence existed ("all 32 threads of a warp
land on the same bank, because this variant does not pad the tile") — the
prediction is now backed by the actual compiled instruction's address
arithmetic, not just source-level reasoning. See
`docs/architecture-bottleneck-diagram.md` for a diagram of exactly this
mechanism (and the one-word-padding fix that would resolve it).

## 5. Nsight Systems (host-side only — `profiling/nsight-systems/01-transpose-*.stats.txt`; ADR 0014: device-side GPU trace is unavailable in this environment)

Both captures show 41 `cudaLaunchKernel` calls (10 warmup + 30 measured +
1 correctness-check launch, matching `apps/bench_transpose_main.cpp`'s
documented behavior exactly) and, interestingly, that `cudaMemcpy` (not
kernel launch) is v2_tiled's single largest host-side API cost
(28.9M ns total across 3 calls) versus v1_naive's `cudaMalloc` (100.3M ns
— but that column is dominated by one-time allocator warm-up, not
representative of steady-state cost; see the raw table). This is
consistent evidence of "no host-side launch-pattern difference between
the two variants" — reinforcing that whatever changed is inside the
kernel, matching the SASS/occupancy findings above rather than contradicting them.

## 6. Interpretation

**The 2.60x speedup is fully explained by memory coalescing, not by
occupancy** (which is provably identical, §3) **and not by a difference
in how many kernels are launched or how the host drives them** (identical,
§5). SASS confirms the mechanism directly: V1's global store issues one
128-{byte-ish}-separated transaction per thread instead of one per warp
(the address literally contains `col * rows`, §4); V2 trades that for a
coalesced global store, at the cost of one `BAR.SYNC` and — a new,
previously-undemonstrated finding — a 32-way shared-memory bank conflict
on its transposed `LDS`. **That the bank conflict exists and V2 is still
2.60x faster than V1 is itself informative**: shared-memory bandwidth is
high enough on this architecture that even a fully-conflicted (32-way
serialized) shared load is still cheaper than the global-memory
transaction pattern it replaces — global-memory coalescing dominates
shared-memory bank efficiency at this problem size. Fixing the bank
conflict (padding the tile to `[32][33]`, the spec's stated V4 rung) is
explicit future work, not implemented here — but it is no longer only a
source-level prediction: this case study is the first point at which the
conflict's existence is confirmed by the actual compiled instruction's
address arithmetic, which is exactly the kind of before/after opportunity
a future V4 padded-tile rung could re-measure against.
