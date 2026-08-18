# Case Study 3: GEMM Tiled → Register-Tiled — Per-SM Occupancy Is a Wash, Grid-Level Utilization Is the Story

**Kernels:** `gemm_tiled_kernel` (V3) vs. `gemm_register_tiled_kernel`
(V4) — `src/kernels/gemm/{gemm_tiled,gemm_register_tiled}.cu`.
**Evidence:** `benchmarks/raw/gemm.jsonl` (Phase 4, already committed),
`profiling/occupancy/gemm.jsonl` (Phase 6, a 5-size sweep per variant —
this case study's central new evidence),
`profiling/ptx-sass/gemm.sass.txt` (Phase 6),
`profiling/nsight-systems/03-gemm-{v3-tiled,v4-register-tiled}.*`
(Phase 6).

## 1. Hypothesis (written in Phase 4, before this profiling pass — `gemm_register_tiled.cuh`)

V3 gives each thread exactly one output element; for a given K-chunk, the
`b_tile` value it reads is *identical* for every thread sharing that
output column, but V3 does not exploit that — each of those threads
independently re-reads the same shared-memory address. V4 changes
exactly one thing: each thread now owns `kGemmRegTM=8` output rows
sharing the same column, so it reads each `b_tile` value into a register
**once** per K-step and reuses it across 8 accumulators instead of 8
different threads each re-reading it. The hypothesis explicitly predicts
**a further speedup "at every size where V3 was already shared-memory-
bandwidth-limited rather than occupancy-limited"** — i.e., it already
anticipates occupancy/grid-size could be a confounding variable, before
any of this profiling data existed. Phase 4's own benchmark writeup
(`benchmarks/methodology.md` §11) found a regression at small sizes and a
decisive win at large sizes, and attributed it to grid size (block count
vs. SM count) using launch-config arithmetic alone. **This case study is
the first point that mechanism is checked against the CUDA Runtime's own
occupancy numbers, not just a block-count calculation.**

## 2. Evidence before / after — wall-clock (Phase 4, `benchmarks/raw/gemm.jsonl`, recomputed here to full precision)

| M=N=K | V3 GFLOP/s | V4 GFLOP/s | V4/V3 ratio | V4 vs. V3 |
|---:|---:|---:|---:|---:|
| 128  | 459.096  | 275.941  | 0.601053 | **39.89% slower** |
| 256  | 2340.57  | 1310.72  | 0.560000 | **44.00% slower** |
| 512  | 5618.63  | 5269.23  | 0.937814 | 6.22% slower |
| 1024 | 6179.17  | 15196.8  | 2.459359 | **145.94% faster** |
| 2048 | 6224.88  | 16571.1  | 2.662075 | **166.21% faster** |

A genuine, non-monotonic crossover: V4 loses at 128/256, is roughly level
at 512, then wins decisively from 1024 up.

## 3. New evidence — theoretical occupancy per SM (`profiling/occupancy/gemm.jsonl`)

| variant | block size | regs/thread | static shared/block | max active blocks/SM | achieved threads/SM | occupancy fraction |
|---|---:|---:|---:|---:|---:|---:|
| v3_tiled | 1024 | 38 | 8192 B | 1 | 1024 | **66.7%** |
| v4_register_tiled | 512 | 62 | 8192 B | 2 | 1024 | **66.7%** |

**Per-SM theoretical occupancy is exactly identical between V3 and V4**
(66.7%, 1024 threads/SM either way), despite V4 using 63% more
registers/thread (62 vs. 38) and half V3's block size (512 vs. 1024) —
V4's smaller blocks simply fit 2-at-a-time where V3's larger blocks fit
1-at-a-time, netting the same per-SM thread count. **This rules out
per-SM occupancy as the explanation for either the small-size regression
or the large-size win** — a thread on either variant has exactly the
same number of co-resident warps available to hide memory latency behind,
at every problem size (per-SM occupancy here does not depend on the
problem's M/N/K at all, only on the fixed block shape). Whatever explains
the crossover has to be something occupancy-*per-SM* cannot see.

## 4. New evidence — grid-level utilization (`profiling/occupancy/gemm.jsonl`, `--m`/`--n` swept across the same 5 sizes as the benchmark)

`theoretical_max_resident_blocks = device_sm_count (128) x max_active_blocks_per_sm` —
how many blocks of this shape could be co-resident **across the whole
GPU** in a single wave. `grid_utilization_fraction = min(1, grid_blocks / theoretical_max_resident_blocks)`:

| M=N=K | V3 grid (blocks) | V3 max resident (128 SMs × 1 blk/SM) | V3 grid utilization | V4 grid (blocks) | V4 max resident (128 SMs × 2 blk/SM) | V4 grid utilization |
|---:|---:|---:|---:|---:|---:|---:|
| 128  | 16   | 128 | **12.5%**  | 4    | 256 | **1.56%**  |
| 256  | 64   | 128 | **50.0%**  | 16   | 256 | **6.25%**  |
| 512  | 256  | 128 | **100.0%** | 64   | 256 | **25.0%**  |
| 1024 | 1024 | 128 | 100.0%     | 256  | 256 | **100.0%** |
| 2048 | 4096 | 128 | 100.0%     | 1024 | 256 | 100.0%     |

(V3's 32x32 tile gives `ceil(M/32) x ceil(N/32)` blocks; V4's 64x64 tile
gives `ceil(M/64) x ceil(N/64)` — a 4x-fewer-blocks tile for the same
M=N, which is exactly why V4 needs a 4x-larger problem before its grid
catches up to V3's, and matches these numbers precisely: V4 reaches 100%
utilization at M=N=1024, the SAME size the wall-clock table above shows
the crossover to a decisive win.)

## 5. New evidence — PTX/SASS instruction-density corroboration (`profiling/ptx-sass/gemm.sass.txt`)

Counting shared-memory loads (`LDS`) against fused-multiply-adds (`FFMA`)
in each kernel's compiled inner-loop body (a **static** count — how many
of each instruction appear once in the compiled kernel, not a runtime
trace):

| variant | LDS (static count) | FFMA (static count) | FFMA per LDS |
|---|---:|---:|---:|
| v3_tiled | 40 | 32 | 0.80 |
| v4_register_tiled | 48 | 128 | 2.67 |

V4 does **3.3x more arithmetic per shared-memory load** than V3 in its
compiled inner loop — directly consistent with the hypothesis's claim
that V4 reuses each `b_tile` read across `kGemmRegTM=8` accumulators
instead of re-reading it 8 separate times. (The exact instruction counts
reflect `ptxas`'s own scheduling/CSE decisions, not a 1:1 transcription of
the unrolled source loop trip count — reported as measured, not
derived from a hand-count of the source.)

## 6. Interpretation

**The crossover is a grid-level utilization effect, not a per-SM
occupancy effect, and now that distinction is measured rather than
inferred from launch-config arithmetic alone.** At M=N=128, V4's 64x64
tile produces only 4 blocks against this GPU's 128 SMs — even with 2
blocks/SM theoretically fittable, **only 4 of 256 possible resident-block
slots are ever filled (1.56% grid utilization)**, so the other ~124 SMs
sit idle for the kernel's entire duration while V3's 16 blocks (12.5% of
its own, larger, 128-slot capacity) at least spread work over more SMs.
V4's per-thread register-reuse optimization (§5, more FFMA per LDS) is
real and compiled correctly, but **it cannot pay for itself when most of
the chip has no work to do at all** — the register-tiling win requires
enough blocks to saturate the grid before the "more work per shared-
memory read" advantage has anywhere to apply. By M=N=1024, both variants
reach 100% grid utilization simultaneously (§4's table), and from that
point on V4's higher arithmetic-per-shared-load ratio (§5) plus its
2-blocks/SM occupancy (matching V3's 1-block/SM at the SAME per-SM
thread count, §3 — so no occupancy cost is paid for the switch) combine
into the decisive 2.46x-2.66x win the wall-clock data shows. **This
confirms, with occupancy-API and grid-math evidence rather than block-
count arithmetic alone, exactly the caveat `gemm_register_tiled.cuh`'s
hypothesis stated before any of this data (including Phase 4's own
original benchmark) existed**: register tiling helps once a kernel is
shared-memory-bandwidth-limited rather than occupancy-limited — and here,
"occupancy-limited" turns out to mean *grid-level* occupancy (whole-GPU
block coverage), not *per-SM* occupancy (which never differs between the
two variants at any size tested).
