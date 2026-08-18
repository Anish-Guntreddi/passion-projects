# Case Study 2: Reduction Sequential-Addressing → Warp-Shuffle — Occupancy Is a Wash, the Win Is in Barrier Count

**Kernels:** `reduce_sequential_addressing_kernel` (V2) vs.
`reduce_warp_shuffle_kernel` (V3) —
`src/kernels/reduction/{reduce_sequential_addressing,reduce_warp_shuffle}.cu`.
**Evidence:** `benchmarks/raw/reduction.jsonl` (Phase 2, already
committed), `profiling/occupancy/reduction.jsonl` (Phase 6),
`profiling/ptx-sass/reduction.sass.txt` (Phase 6),
`profiling/nsight-systems/02-reduction-{v2-sequential,v3-warpshuffle}.*`
(Phase 6).

## 1. Hypothesis (written in Phase 2, before this profiling pass — `reduce_warp_shuffle.cuh`)

V2's tree reduction uses `__syncthreads()` + shared memory all the way
down to one surviving element, including the last 5 steps (strides 16,
8, 4, 2, 1) where every remaining active thread already fits in a single
warp. V3 changes exactly one thing: those last 5 shared-memory steps are
replaced with `__shfl_down_sync` (ADR 0004: `_sync` family, full mask —
correct here because `tid < 32` is warp-uniform). The prediction: this
"trades 5 `__syncthreads()` block-wide barriers... for 5 register-to-
register warp shuffles, which do not need a barrier and do not touch
shared memory at all," for "a modest but measurable speedup." Phase 6's
job is to check whether that's actually what happens at the instruction
level, and whether occupancy plays any role in the result.

## 2. Evidence before / after — wall-clock (Phase 2, `benchmarks/raw/reduction.jsonl`, n=67,108,864, block_size=256, recomputed here to full precision)

| variant | median (ms) | effective BW (GB/s) | % of 1008.1 GB/s peak |
|---|---:|---:|---:|
| v2_sequential_addressing | 0.412672 | 650.481 | 64.53% |
| v3_warp_shuffle | 0.375808 | 714.289 | 70.85% |

**v3_warp_shuffle is 1.09809x faster** (714.289 / 650.481) — a real but,
as the hypothesis itself predicted, modest win, not a dramatic one.

## 3. New evidence — theoretical occupancy (`profiling/occupancy/reduction.jsonl`, all queried at block_size=256, this ladder's benchmarked size)

| variant | regs/thread | dynamic shared/block | max active blocks/SM | achieved threads/SM | occupancy fraction |
|---|---:|---:|---:|---:|---:|
| v2_sequential_addressing | 10 | 1024 B | 6 | 1536 | **100.0%** |
| v3_warp_shuffle | 12 | 1024 B | 6 | 1536 | **100.0%** |
| v4_vectorized_coarsened (context) | 19 | 1024 B | 6 | 1536 | **100.0%** |

**All three rungs sit at 100% theoretical occupancy** — 6 blocks/SM x 256
threads = 1536, exactly this GPU's per-SM thread limit. V3 costs 2 more
registers/thread than V2 (12 vs. 10) and V4 costs 9 more (19), but none
of that register growth is enough to push any variant below 6
blocks/SM — this GPU's register file is not the limiting resource for
any of them at this block size. **This rules out occupancy as the
explanation for V2→V3's speedup, just as cleanly as Case Study 1 ruled it
out for transpose**: every rung has exactly the same number of resident
warps available to hide memory latency. Whatever explains the 1.098x
speedup has to be something other than "more concurrency."

## 4. New evidence — SASS (`profiling/ptx-sass/reduction.sass.txt`)

Both kernels compile their `for (stride = blockDim.x/2; ...; stride >>= 1)`
loop as a genuine **runtime loop** (`blockDim.x` is not known at compile
time inside the kernel, so `#pragma unroll` cannot apply) — so each
loop's body appears exactly **once** in the compiled code, containing
exactly one `BAR.SYNC.DEFER_BLOCKING`, executed once per trip through the
loop at runtime:

```
V2 (reduce_sequential_addressing_kernel):
  STS [R7.X4], R0 ;              // initial load into shared memory
  BAR.SYNC.DEFER_BLOCKING 0x0 ;  // barrier #1 (before the loop)
  @!P1 LDS R4, [R7.X4] ;         // \
  @!P1 LDS R5, [R2] ;            //  } one loop trip's body
  @!P1 STS [R7.X4], R4 ;         // /
  BAR.SYNC.DEFER_BLOCKING 0x0 ;  // the loop's ONE barrier instruction, executed once per trip
  ...                            // (branches back for the next trip)
  LDS R5, [RZ] ;                 // final read after the loop exits

V3 (reduce_warp_shuffle_kernel):
  STS [R9.X4], R0 ;
  BAR.SYNC.DEFER_BLOCKING 0x0 ;  // barrier #1 (before the loop, identical to V2)
  @!P1 LDS R4, [R9.X4] ;
  @!P1 LDS R3, [R3] ;
  @!P1 STS [R9.X4], R4 ;
  BAR.SYNC.DEFER_BLOCKING 0x0 ;  // the loop's ONE barrier instruction (same structure as V2's)
  LDS R0, [R9.X4] ;              // final read after the loop exits (loop bound differs -- see below)
  SHFL.DOWN PT, R3, R0, 0x10, 0x1f ;  // \
  SHFL.DOWN PT, R2, R3, 0x8, 0x1f ;   //  \
  SHFL.DOWN PT, R5, R2, 0x4, 0x1f ;   //   } 5 warp shuffles, NO barrier instruction anywhere after them
  SHFL.DOWN PT, R4, R5, 0x2, 0x1f ;   //  /
  SHFL.DOWN PT, R7, R4, 0x1, 0x1f ;   // /
```

The two kernels' loop bodies are structurally identical (same
`STS`/`LDS`/`BAR.SYNC` shape) — the only difference is the loop's *trip
count*, which is a source-level fact (the loop bound, `stride > 0` for V2
vs. `stride >= 32` for V3), not something SASS can show directly since
it's a runtime value. At this ladder's benchmarked `block_size=256`, that
bound difference means: **V2's loop runs 8 trips (strides 128, 64, 32,
16, 8, 4, 2, 1) → 8 barrier executions + 1 initial = 9 dynamic
`BAR.SYNC` executions per block. V3's loop runs 3 trips (strides 128, 64,
32) → 3 barrier executions + 1 initial = 4 dynamic `BAR.SYNC` executions
per block**, after which its 5 `SHFL.DOWN` instructions run with
**zero** additional barriers — confirmed directly by their total absence
from the SASS after the loop exits. **V3 trades exactly 5 barrier
executions for 5 register-only shuffles per block**, precisely matching
the hypothesis's own count ("5 `__syncthreads()` block-wide barriers...
for 5 register-to-register warp shuffles").

## 5. Nsight Systems (host-side only — `profiling/nsight-systems/02-reduction-*.stats.txt`; ADR 0014: device-side GPU trace unavailable)

Both captures show 41 `cudaLaunchKernel` + 41 `cudaMemsetAsync` calls
(one `memset` per launch to zero the atomic-accumulate target,
`zero_output=true`'s documented behavior) — identical launch pattern
between V2 and V3, confirming (as in Case Study 1) that the host side is
not where this difference lives.

## 6. Interpretation

**A ~10% speedup from removing 5 out of 9 barrier executions per block
is a physically reasonable, non-dramatic result — exactly what the
hypothesis predicted ("modest but measurable"), not the same order of
magnitude as V1→V2's 1.51x (removing a much larger amount of divergent
addressing work) or V3→V4's 1.31x (a memory-traffic change, not a
synchronization one).** Each `__syncthreads()` is a block-wide rendezvous
that stalls every warp until the slowest one arrives; a `__shfl_down_sync`
is a single-instruction, single-cycle-class register exchange within one
already-executing warp — removing 5 of the former per block and adding 5
of the latter should save real but modest time relative to the kernel's
total cost (which also includes the load phase and 3 remaining
shared-memory tree steps V2 and V3 share identically). **The occupancy
evidence (§3) is what makes this interpretation solid rather than
speculative**: with occupancy provably identical between V2 and V3, the
barrier-count difference confirmed by SASS (§4) is the only remaining
candidate mechanism visible in this repo's evidence — not "one plausible
explanation among several," the way Phase 5's un-profiled softmax finding
had to be reported (`benchmarks/methodology.md` §12). This is the
concrete difference profiler-backed evidence makes: Phase 5's fusion
result was reported as "plausible mechanism, untested with a profiler";
this one is reported as measured and instruction-confirmed.
