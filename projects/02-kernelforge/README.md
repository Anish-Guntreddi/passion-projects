# KernelForge

CUDA kernel optimization laboratory. Full product spec: `02-kernelforge-spec.md`
(copied into this repo root). This README covers the full roadmap,
**Phase 0 (Harness & correctness infra) through Phase 7 (Portfolio
release)** — all MVP phases per `02-kernelforge-spec.md` Part 3.

Target hardware: **NVIDIA GeForce RTX 4090, sm_89** (ADR 0001) — the only
GPU this repo is built and tested against. Build environment: **WSL2
Ubuntu on Windows 11**, CUDA toolkit 12.6, g++ 13.3, CMake 3.28, Ninja.

## Status

| Phase | Deliverables | Exit criterion | Status |
|---|---|---|---|
| 0 | CMake/CUDA project, capability check, device-info, error macros, deterministic reference/test utilities, benchmark schema | Vector add validates; benchmark output machine-readable | **Met** |
| 1 | Vector/SAXPY, transpose, coalescing/stride microbenchmarks | Repo empirically demonstrates contiguous vs. pathological access | **Met** |
| 2 | Reduction ladder (5 versions), 2 scan strategies, correctness across awkward sizes | ≥4 reduction versions and 2 scan strategies benchmarked | **Met** |
| 3 | Histogram/atomics lab: contention baseline, privatization, coarsening | Low- vs high-contention workloads documented with measured results | **Met** |
| 4 | GEMM ladder (naive → coalesced → tiled → register-tiled), cuBLAS ceiling comparison | Report explains memory reuse and resource tradeoffs | **Met** |
| 5 | Softmax + RMSNorm ladders with references and optimization experiments | Benchmarks across representative tensor shapes | **Met** |
| 6 | 3 profiler-backed case studies: theoretical occupancy, PTX/SASS inspection, Nsight Systems host-side traces | 3 complete optimization case studies | **Met** |
| 7 | Plots (`analysis/generate_plots.py` → `benchmarks/plots/`), architecture bottleneck diagram, reproducibility scripts, fresh-clone verification | Fresh-clone reproduction verified | **Met** |

See `docs/decisions/` for every architectural decision (D1-D8 from the
spec, plus implementation choices — 14 ADRs total). See
`benchmarks/methodology.md` for the full, numbers-backed results writeup
(§8 Phase 1, §9 Phase 2, §10 Phase 3, §11 Phase 4, §12-13 Phase 5, §14
Phase 6), and each kernel family's own `README.md` (`src/kernels/
{reduction,scan,histogram,gemm,softmax,norm}/README.md`) for the ladder
table and every rung's written hypothesis. Phase 6's 3 case studies are
in `profiling/case-studies/`.

## Repository layout

```
02-kernelforge/
  CMakeLists.txt, cmake/            CMake + CUDA build config
  src/common/                       error macros, timers, device query, RNG,
                                     CPU references, comparison, stats,
                                     benchmark schema, test framework
  src/kernels/vector/                vector_add, saxpy
  src/kernels/transpose/             transpose_naive (V1), transpose_tiled (V2)
  src/kernels/microbench/            stride_copy (dedicated coalescing microbenchmark)
  src/kernels/reduction/             5-rung reduction ladder + README (ladder hypotheses)
  src/kernels/scan/                  2 scan strategies + shared multi-block driver + README
  src/kernels/histogram/             3-rung atomics lab (baseline/privatized/coarsened) + README
  src/kernels/gemm/                  4-rung GEMM ladder + cuBLAS ceiling reference + README
  src/kernels/softmax/               3-rung softmax ladder + README
  src/kernels/norm/                  3-rung RMSNorm ladder + README
  apps/                              device-info, bench_vector_add, bench_saxpy,
                                     bench_transpose, bench_stride, bench_reduction,
                                     bench_scan, bench_histogram, bench_gemm,
                                     bench_softmax, bench_rmsnorm, occupancy_report
  tests/                             kf_tests (151 correctness tests, no GPU-less mode)
  benchmarks/                        methodology.md, schema/, configs/, raw/, plots/
  profiling/                         case-studies/ (3 Phase 6 writeups), occupancy/,
                                     ptx-sass/, nsight-systems/, nsight-compute/,
                                     schema/, metric-guide.md
  docs/                              gpu-model.md, optimization-method.md,
                                     architecture-bottleneck-diagram.md, decisions/ (14 ADRs)
  scripts/                           configure/build/test/bench-* .sh + Python drivers,
                                     run_occupancy_report.sh, run_ptx_sass_dump.sh,
                                     run_nsys_case_studies.sh (Phase 6)
  analysis/                          Python (pandas/matplotlib) plot pipeline (Phase 7)
```

## Building and testing

All commands run **inside WSL2 Ubuntu**. From Windows:

```
wsl -d Ubuntu -- bash -lc "/mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge/scripts/build.sh"
wsl -d Ubuntu -- bash -lc "/mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge/scripts/test.sh"
```

Or, from inside a WSL shell:

```bash
cd /mnt/c/Users/guntr/Documents/passion-projects/projects/02-kernelforge
scripts/configure.sh   # cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ...
scripts/build.sh       # cmake --build build
scripts/test.sh         # build (if needed) + run kf_tests + validate benchmarks/raw/*.jsonl against schema
```

`scripts/configure.sh` pins `nvcc` by explicit path
(`/usr/local/cuda/bin/nvcc`) rather than relying on `$PATH`, and defaults
`CMAKE_CUDA_ARCHITECTURES=89` (override with `KF_CUDA_ARCH=<n>` env var).

### Device info

```bash
scripts/run_device_info.sh          # human-readable
scripts/run_device_info.sh --json   # machine-readable
```

### Correctness tests

```bash
build/tests/kf_tests                       # all 151 tests
build/tests/kf_tests --filter=Transpose    # substring filter on "Suite.Name"
build/tests/kf_tests --filter=Reduction    # Phase 2 reduction ladder only
build/tests/kf_tests --filter=Scan         # Phase 2 scan strategies only
build/tests/kf_tests --filter=Histogram    # Phase 3 atomics lab only
build/tests/kf_tests --filter=Gemm         # Phase 4 GEMM ladder only
build/tests/kf_tests --filter=Softmax      # Phase 5 softmax ladder only
build/tests/kf_tests --filter=RmsNorm      # Phase 5 RMSNorm ladder only
build/tests/kf_tests --filter=Occupancy    # Phase 6 occupancy-report sanity checks only
```

Or via CTest: `cd build && ctest --output-on-failure`.

### Compute Sanitizer

```bash
scripts/run_sanitizer.sh memcheck    # or racecheck / initcheck / synccheck
```

Verified clean on this repo's full 151-test suite (2026-08-17, including
Phase 6): `========= ERROR SUMMARY: 0 errors`.

### Reproducing the benchmarks

```bash
scripts/run_all_benchmarks.sh
```

This builds (if needed), runs all ten `bench_*` binaries across every
`(variant, size)` point in `benchmarks/configs/*.json`, appends results to
`benchmarks/raw/*.jsonl`, and validates every record against
`benchmarks/schema/bench_result.schema.json`. A single family can be run
with e.g. `scripts/run_bench_transpose.sh`, `scripts/run_bench_reduction.sh`
(reduction's v0 rung is intentionally excluded from one of its six
configured sizes — see `benchmarks/methodology.md` §9.1 and that script's
own comment for why), `scripts/run_bench_scan.sh`, `scripts/
run_bench_histogram.sh` (runs BOTH the uniform and skewed contention
sweeps — Phase 3's exit criterion), `scripts/run_bench_gemm.sh`, `scripts/
run_bench_softmax.sh`, or `scripts/run_bench_rmsnorm.sh`. A single point
can be run directly, e.g.:

```bash
build/apps/bench_transpose  --variant v2_tiled                  --n 4096    --warmup 10 --reps 30
build/apps/bench_reduction  --variant v4_vectorized_coarsened   --n 1048576 --warmup 10 --reps 30
build/apps/bench_scan       --variant hillis_steele --block-size 1024 --n 65536 --warmup 10 --reps 30
build/apps/bench_histogram  --variant v3_privatized_coarsened --contention skewed --num-bins 256 --n 2097152 --warmup 10 --reps 30
build/apps/bench_gemm       --variant v4_register_tiled --n 2048 --warmup 10 --reps 30
build/apps/bench_softmax    --variant v3_fused_online --rows 4096 --n 2048 --warmup 10 --reps 30
build/apps/bench_rmsnorm    --variant v3_vectorized --rows 4096 --n 2048 --warmup 10 --reps 30
```

Every benchmark binary runs exactly one correctness check against the CPU
reference before any timing loop, and refuses to emit a result if it
fails (FR6). `scripts/bench_driver.py` additionally refuses to *write* any
result with `correctness_passed=false`.

### Reproducing the Phase 6 profiling evidence

```bash
scripts/run_occupancy_report.sh   # profiling/occupancy/*.jsonl -- theoretical occupancy (no GPU lock needed, deterministic)
scripts/run_ptx_sass_dump.sh      # profiling/ptx-sass/*.txt -- PTX/SASS static disassembly (no GPU needed at all)
scripts/run_nsys_case_studies.sh  # profiling/nsight-systems/*.nsys-rep + .stats.txt -- real timed captures, needs the GPU-benchmark lock if numbers will be cited
```

See `profiling/case-studies/` for the 3 writeups these feed, and
`docs/decisions/0014-phase6-profiling-evidence-strategy.md` for why
Nsight Compute (`ncu`) and Nsight Systems' device-side GPU trace are both
unavailable in this environment, and what's used instead.

### Reproducing the plots

```bash
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r analysis/requirements.txt
analysis/.venv/bin/python analysis/generate_plots.py
```

Writes 6 PNGs to `benchmarks/plots/`, every value read directly from the
committed `benchmarks/raw/*.jsonl` / `profiling/occupancy/gemm.jsonl`
records — see `analysis/README.md` for what each plot covers.

## Phase 1 headline result

The stride/coalescing microbenchmark (`src/kernels/microbench/stride_copy.cuh`,
`benchmarks/raw/stride_copy.jsonl`) moves exactly the same number of useful
bytes at every stride — only the address distance between consecutive
threads' accesses changes:

| stride | effective bandwidth | vs. stride=1 |
|---:|---:|---:|
| 1   | 1376.1 GB/s | 1.00x |
| 8   | 327.7 GB/s  | 0.24x |
| 16  | 93.1 GB/s   | 0.068x |
| 128 | 39.8 GB/s   | 0.029x |

The same mechanism, inside a real kernel: `transpose_naive` (coalesced
read, pathological write) vs. `transpose_tiled` (shared-memory-staged,
both coalesced), at 8192x8192 (`benchmarks/raw/transpose.jsonl`):
**272 GB/s vs. 709 GB/s — a 2.6x speedup from changing exactly one
variable**, with `v2_tiled` reaching ~70% of the device's 1008.1 GB/s
theoretical peak bandwidth versus `v1_naive`'s ~27%. Full numbers, every
raw repetition, and the reasoning for each are in
`benchmarks/methodology.md` §8.

## Phase 2 headline result

The 5-rung reduction ladder (`src/kernels/reduction/`, all five variants
now benchmarked in `benchmarks/raw/reduction.jsonl`) is monotonically
faster rung over rung at the largest fully-HBM-bound V1-V4 size
(67,108,864 elements): naive interleaved addressing → sequential
addressing is **1.51x**, → warp-shuffle finalization is a further
**1.10x**, → vectorized/coarsened loads is a further **1.31x**, for a
**2.18x V1→V4 speedup overall (430 → 936 GB/s, ~93% of theoretical
peak)**. V0 (pure global-memory atomics, no shared memory) confirms the
ladder's founding hypothesis directly: its bandwidth is nearly flat
(1.78-2.87 GB/s) across a 4096x range of `n`, so **V1 is 1.8x-149x faster
than V0** depending on size (widening as `n` grows, since V0 is
contention-bound, not bandwidth-bound) — and at n=16,777,216, **V4 is
1197.7x faster than V0** overall. V0 also surfaces a genuinely distinct
finding beyond contention: at the largest benchmarked size it fails its
own correctness gate from fp32 accumulator saturation (summing 67M
elements directly into one `float` via atomics — not a logic bug), so its
benchmark row stops one size short of V1-V4's; see
`benchmarks/methodology.md` §9.1 for the mechanism. Every rung's
prediction matched the direction actually measured; none was falsified.
The two scan strategies (`src/kernels/scan/`,
`benchmarks/raw/scan.jsonl`) show Hillis-Steele beating Blelloch by
1.25x-2.16x across every size up to the 2-level ceiling — the opposite of
what pure work-efficiency would predict, and reported as measured rather
than adjusted to fit the initial guess. Full numbers and interpretation:
`benchmarks/methodology.md` §9.

## Phase 3 headline result

The histogram/atomics lab (`src/kernels/histogram/`,
`benchmarks/raw/histogram_{uniform,skewed}.jsonl`) is Phase 3's exit
criterion made concrete: at 33,554,432 elements / 256 bins, the
global-atomic baseline is **1.93x slower under a high-contention (skewed)
input than a low-contention (uniform) one** (17.5 → 9.1 GB/s) — the
*only* variant whose throughput depends on the input's statistical shape.
Shared-memory privatization (V2) all but erases that dependence (111.8 →
162.4 GB/s, *faster* under skew, the opposite of the naive prediction —
reported as measured, see `benchmarks/methodology.md` §10 for the
plausible-but-unconfirmed mechanism); adding thread coarsening (V3)
removes the contention penalty outright — at the largest size the two
distributions sit within 4% of each other (794.4 vs 762.0 GB/s, uniform
ahead) — while delivering a **45x (uniform) to 84x (skewed) speedup over
the V1 baseline**. Full sweep, every size: `benchmarks/methodology.md` §10.

## Phase 4 headline result

The 4-rung GEMM ladder (`src/kernels/gemm/`, `benchmarks/raw/gemm.jsonl`)
is monotonically faster rung over rung at the largest size swept
(2048x2048x2048): coalescing the naive kernel's memory mapping is **8.26x**,
adding shared-memory tiling a further **1.22x**, and register tiling a
further **2.66x**, for a **26.8x V1→V4 speedup overall (617.8 → 16,571.1
GFLOP/s)** — one variable changed per rung, every variant (V1-V4 plus the
cuBLAS ceiling/reference) numerically agrees with the CPU reference at
every tested shape. V4 reaches **30.8% of cuBLAS's measured throughput**
at this size, reported as exactly that framing per spec FR2 ("cuBLAS is a
ceiling/reference, never a target to beat" —
`docs/decisions/0013-cublas-ceiling-methodology.md`). Register tiling's
win is **not** monotonic: it is a genuine 40-44% *regression* at the two
smallest sizes tested (too few blocks in its 64x64-tile grid to occupy
this GPU's 128 SMs), crossing over to a decisive win from 1024 upward —
confirming a caveat the hypothesis stated before this data existed, not
falsifying it. Full numbers and interpretation: `benchmarks/
methodology.md` §11.

## Phase 5 headline result

The 3-rung softmax and RMSNorm ladders (`src/kernels/{softmax,norm}/`,
`benchmarks/raw/{softmax,rmsnorm}.jsonl`) both show the SAME pattern for
their first rung — warp-shuffle reduction is a real win at small-to-medium
row widths (up to **1.51x** for softmax, **1.38x** for RMSNorm at
cols=128) that fades to parity by cols=4096-8192 — independently confirmed
on two differently-shaped kernel families. Their SECOND rungs diverge
sharply, reported as measured either way: RMSNorm's vectorized (`float4`)
loads deliver a real, if size-dependent, win (up to **16.2% faster** at
cols=768, narrowing to a small loss at the largest size tested), matching
its hypothesis where it wins; softmax's "bounded fusion" (online
max+sum in one pass) does **not** — it is the same speed or measurably
slower than the unfused warp-shuffle rung at six of the seven sizes
tested, contradicting its own hypothesis. The plausible mechanism
(untested with a profiler, disclosed as such): the fused kernel's
online-combine issues two `expf()` calls per element versus the unfused
version's one `expf()` call total across its two passes — trading a
memory-traffic saving for a transcendental-throughput cost that, at these
row sizes, outweighs it. Full numbers and interpretation: `benchmarks/
methodology.md` §12-13.

## Phase 6 headline result

3 case studies (`profiling/case-studies/`) re-examine already-benchmarked
Phase 1/2/4 comparisons with new evidence — theoretical occupancy (CUDA
Runtime occupancy API), PTX/SASS static disassembly, and Nsight Systems
host-side traces (`docs/decisions/0014-phase6-profiling-evidence-
strategy.md` covers why hardware-counter profiling, `ncu` and Nsight
Systems' device-side GPU trace alike, is genuinely unavailable in this
WSL2 environment — `ERR_NVGPUCTRPERM`, reproduced precisely, not
asserted). The throughline across all three: **theoretical occupancy is
proven identical between the "before" and "after" variant in every one of
the 3 case studies** (transpose naive-vs-tiled: 66.7% both; reduction
sequential-addressing-vs-warp-shuffle: 100% both; GEMM tiled-vs-register-
tiled: 66.7% both, at every size tested) — ruling out "more concurrency"
as the explanation every time, and pointing each case study's
interpretation at a different, SASS- or grid-utilization-confirmed
mechanism instead: transpose's win is coalescing (confirmed in the
compiled store address, which SASS also reveals carries a real, previously
undemonstrated 32-way shared-memory bank conflict on the transposed read
that the win *survives*, not one that was hidden); reduction's warp-
shuffle win is a **precisely** 5-barrier-execution-per-block reduction
(matching the hypothesis's own count exactly, confirmed by counting SASS
`BAR.SYNC`/`SHFL.DOWN` instructions against each loop's runtime trip
count); GEMM's register-tiling regression-then-win crossover is a
**grid-level** (whole-GPU block-residency) utilization effect, not a
per-SM one — V4's utilization crosses from 25.0% to 100.0% at exactly
M=N=1024, the same size the Phase 4 wall-clock data crosses from a 6.22%
loss to a 145.94% gain. Full writeups, every number recomputed to full
precision from committed raw artifacts: `profiling/case-studies/
{01-transpose-bank-conflict,02-reduction-warp-shuffle-barriers,
03-gemm-register-tiling-occupancy}.md`; `benchmarks/methodology.md` §14.
`docs/architecture-bottleneck-diagram.md` diagrams the bank-conflict
mechanism directly (warp-lane-to-shared-memory-bank mapping, unpadded vs.
padded); `benchmarks/plots/gemm_occupancy_crossover.png` plots the
grid-utilization crossover alongside the wall-clock speedup it explains.

## Architectural decisions

All spec open decisions (D1-D8) plus implementation choices are recorded
in `docs/decisions/` (14 ADRs):

- 0001 — Target GPU architecture (sm_89, RTX 4090)
- 0002 — Minimum supported compute capability (8.9)
- 0003 — FP32-only MVP
- 0004 — Architecture-safe warp primitives (`_sync` family, full masks) — first exercised in Phase 2's `reduce_warp_shuffle`
- 0005 — Benchmark noise control (clocks NOT locked under WSL2 — disclosed, not hidden)
- 0006 — SASS analysis depth (PTX default, SASS where it explains a result) — exercised in Phase 6 (case studies 1 and 2)
- 0007 — CUB comparisons (deferred/stretch)
- 0008 — Templating by dtype (no, FP32 only)
- 0009 — Lightweight in-repo test framework instead of GoogleTest
- 0010 — Hand-rolled JSON for the benchmark result schema
- 0011 — BenchResult schema extension for Phase 2/3 (`reduction`/`scan`/`histogram` families, `num_bins`/`contention_profile` fields)
- 0012 — BenchResult schema extension for Phase 4/5 (`gemm`/`softmax`/`rmsnorm` families, `k` field; GEMM FLOP-counting convention)
- 0013 — cuBLAS ceiling-comparison methodology (row-major/column-major invocation, handle lifecycle, ceiling-not-rung framing)
- 0014 — Phase 6 profiling evidence strategy (`ncu`/Nsight Systems device-trace are both `ERR_NVGPUCTRPERM`-blocked; theoretical occupancy + PTX/SASS + host-side traces + this repo's own committed timings used instead)

## Known limitations (disclosed, not silently worked around)

- **GPU clocks are not locked.** `nvidia-smi -lgc ...` returns a
  permission error under this WSL2 guest. Every benchmark result records
  `locked_clocks: false` plus an observed (unlocked) clock snapshot — see
  ADR 0005.
- **Nsight Compute (`ncu`) is present but blocked by `ERR_NVGPUCTRPERM`**
  (a GPU-performance-counter permission restriction enforced by the
  Windows host driver under WSL2 — not fixable from inside this guest;
  confirmed 2026-08-17 with a direct reproduction, not asserted from the
  earlier Phase 0-1 "not found" state). **Nsight Systems' device-side GPU
  kernel/memory trace is blocked by the same restriction** — its
  host-side CUDA API trace works and is used instead. Every "plausible
  mechanism" writeup in `benchmarks/methodology.md` §10/§12 (Phases 3/5)
  remains explicitly disclosed as profiler-unconfirmed, since neither
  covers those specific kernels; Phase 6's 3 case studies use theoretical
  occupancy + PTX/SASS + host-side traces instead of hardware counters
  for the kernels they DO cover. See `docs/decisions/
  0014-phase6-profiling-evidence-strategy.md` for the full investigation.
- **Transpose ladder is not complete.** Only V1 (naive) and V2 (tiled,
  coalesced read+write) are implemented. V2's shared-memory tile is
  deliberately *unpadded*, so it carries a shared-memory bank conflict on
  the transposed read — predicted since Phase 1 and, as of Phase 6, now
  **confirmed directly in the compiled SASS's load-address arithmetic**
  (`profiling/case-studies/01-transpose-bank-conflict.md` §4: a 32-way
  conflict on every tile) rather than only a source-level prediction. The
  V4 fix (padding) from the spec's ladder remains explicit future work,
  not silently included or silently omitted (see
  `src/kernels/transpose/transpose_tiled.cuh`).
- **Scan supports n up to `block_size^2` elements only** (1,048,576 at
  `block_size=1024`). A 3rd scan level would be required beyond that —
  explicit future work; `scan_multilevel_launch` throws
  `std::runtime_error` rather than silently truncating or misbehaving (see
  `src/kernels/scan/scan_common.cuh`).
- **Histogram's privatized shared-memory histogram requires
  `num_bins * 4 <= 49,152` bytes** (this GPU's default per-block shared
  memory). `histogram_common.cuh::validate_num_bins_fits_shared_mem` fails
  loudly if exceeded rather than letting the launch fail late.
- **Reduction's V0 rung (`v0_naive_global_atomic`) is not benchmarked at
  the ladder's largest configured size (67,108,864 elements).** It fails
  its own correctness gate there from fp32 accumulator saturation (67M
  direct `atomicAdd`s into one `float`, not a logic bug — see
  `benchmarks/methodology.md` §9.1); the benchmark harness correctly
  refuses to commit a failing result rather than loosen the tolerance to
  hide it. `scripts/run_bench_reduction.sh` reproduces exactly this
  committed shape (v1-v4 at all six sizes, v0 at five of them) from a
  fresh clone.
- **GEMM's benchmark sweep is capped at 2048x2048x2048.** Each
  `bench_gemm` invocation recomputes its own CPU reference (a naive O(N³)
  triple loop) from scratch, which dominates wall-clock time at large N
  far more than the GPU kernels themselves do; 2048 was chosen to keep
  the full 25-point sweep (5 variants x 5 sizes) fast while still
  reaching V4's clear-win regime (`benchmarks/methodology.md` §11).
  Larger sizes are legal (correctness-tested up to 256x256x256 in
  `tests/test_gemm.cpp`) but not part of the committed benchmark sweep.
- **RMSNorm's vectorized rung (V3) requires `cols % 4 == 0`** (every row
  must start 16-byte-aligned for its `float4` loads — see
  `common/launch_validate.hpp::validate_cols_multiple_of_4`). Fails
  loudly for other `cols` rather than risking a silent misaligned access.

## Phase 7 reproducibility verification

Phase 7's exit criterion ("fresh-clone reproduction verified") was
checked by copying this project's full working tree (excluding the
`build/` and `analysis/.venv/` directories — the same exclusions
`.gitignore` applies) to an isolated scratch directory in WSL2 and, from
there, running exactly the commands this README documents, with **no
pre-existing build cache, no pre-existing venv**:

```bash
scripts/build.sh    # -> 77/77 targets built from zero
scripts/test.sh     # -> 151/151 tests passed; 166 + 15 committed records schema-valid
scripts/run_occupancy_report.sh   # -> reproduces the exact 15 committed occupancy records
scripts/run_ptx_sass_dump.sh      # -> reproduces the same compiled instructions as the 6
                                   #    committed PTX/SASS dump files (see caveat below)
python3 -m venv analysis/.venv && analysis/.venv/bin/pip install -r analysis/requirements.txt
analysis/.venv/bin/python analysis/generate_plots.py   # -> all 6 plots regenerate
```

All of the above succeeded, from a directory that had never run
`cmake`/`ninja`/`pip` before. (A literal `git clone` was not used for this
check because this repo is one project directory inside a larger
monorepo whose git history commits work in batches across projects — a
tree-copy of the exact committed file content is the equivalent
verification once this work lands in a commit, and is what was done
here.)

**Caveat on `run_ptx_sass_dump.sh`: not byte-identical, verified
instruction-identical.** Unlike the other five steps, this one's raw
output text is **not** byte-for-byte reproducible from a tree copied to a
different absolute path, and the check above confirmed that directly
rather than assuming it: `nvcc -lineinfo` (used for these dumps'
`--generate-line-info`) embeds the compiling machine's absolute source
path into the fatbin, and each translation unit's anonymous namespace
(every kernel here lives in one) gets an `nvcc`-generated symbol-mangling
hash that is itself derived from that same absolute path — both are
`cuobjdump` output, not something this repo's build controls. Verified by
building `bench_transpose` from a tree copied to `/tmp/kf_pathcheck` (a
different absolute path than this repo's checkout) and diffing its
`cuobjdump --dump-ptx` output against the committed
`profiling/ptx-sass/transpose.ptx.txt`: 58 raw lines differ, all of them
either an `identifier = <path>/....cu` line or a mangled symbol
containing `_GLOBAL__N__<8-hex-hash>_`; after normalizing both the path
string and the hash token, the diff is empty (0 lines) — the compiled
instructions are 100% identical, only these two path-derived,
cosmetic tokens differ. `run_ptx_sass_dump.sh` reproduces the kernels'
actual compiled behavior exactly; it does not reproduce the committed
`.txt` files byte-for-byte, and no case-study conclusion in
`profiling/case-studies/` depends on the literal path or hash text.

## What's next

All MVP phases (0-7) per `02-kernelforge-spec.md` Part 3 are implemented.
Remaining work is exactly the spec's own stretch-goal list (Part 3,
post-MVP only): Tensor Core GEMM, CUTLASS comparison, CUDA Graph
launch-overhead experiment, async copy/double buffering, multi-GPU
bandwidth microbenchmarks, persistent kernels, a custom benchmark-reuse
allocator, and Hopper/Blackwell comparative runs if that hardware becomes
available — plus the disclosed, explicit future work already named
throughout this README's "Known limitations" section (transpose V4
padding, a 3rd scan level, `ncu` if this environment's restriction is
ever lifted).
