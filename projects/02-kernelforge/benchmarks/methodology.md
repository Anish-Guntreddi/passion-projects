# Benchmark Methodology

**This document is written before any benchmark number in this repo is
collected**, per the Phase 0 roadmap requirement ("define the benchmark
result schema and `benchmarks/methodology.md` before collecting any
headline numbers"). It defines exactly how every number under
`benchmarks/raw/` was produced; results are added to this repo only by
running the process described here, never hand-written.

## 1. Schema

Every result is one JSON object, one per line, appended to
`benchmarks/raw/<kernel_family>.jsonl`. The struct that produces it is
`kernelforge::BenchResult` (`src/common/bench_result.hpp`); the equivalent
JSON Schema is `benchmarks/schema/bench_result.schema.json`. Both are
committed before any `.jsonl` file exists (ADR 0010). Fields, briefly:

- `env`: GPU name + compute capability, CUDA driver/runtime version, host
  and CUDA compiler id/version, `CMAKE_CUDA_ARCHITECTURES`, build type,
  a compile-flags note, a locked-clocks flag + explanatory note (see §3),
  an observed (not locked) SM/memory clock snapshot, an OS note, and a UTC
  timestamp.
- Problem shape: `n` (+ `rows`/`cols` for transpose, `stride` for the
  stride microbenchmark), `dtype` (always `"fp32"`, ADR 0003), and the
  actual `launch_config` (block/grid dims) used.
- `warmup_iters`, `measured_reps`, `seed`: exactly what was run and with
  what deterministic input.
- `raw_timings_ms`: every individual repetition's kernel-only elapsed
  time (never just a mean).
- `stats_ms`: min/max/mean/median/stddev/p25/p75 computed from
  `raw_timings_ms` by `kernelforge::compute_stats` (linear-interpolation
  percentiles).
- `bytes_moved` / `effective_bandwidth_gb_s`: useful bytes read+written by
  the kernel (independent of access pattern) divided by the median time.
  This is the metric Phase 1's coalescing story is told with.
- `gflops`: 0.0 for every Phase 0/1 kernel here (they are memory-bound,
  ~0-1 FLOP per element moved; this field exists for later,
  compute-bound families like GEMM).
- `tolerance_atol` / `tolerance_rtol` / `correctness_passed` /
  `correctness_note`: the exact tolerance used and the result of the
  correctness check that ran **before** this result's timing loop.

## 2. How a result is produced (reproducibility)

1. A `bench_<family>` binary (`apps/bench_*_main.cpp`) is built by CMake
   (see repo root `README.md` for the exact configure/build commands).
2. It queries the device and enforces the minimum compute capability
   (ADR 0002) before doing anything else.
3. It generates deterministic input via `kernelforge::make_random_vector`
   seeded from `--seed` (default `kernelforge::kDefaultSeed =
   123456789`).
4. **It runs exactly one correctness check against the CPU reference
   implementation (`src/common/reference_ops.*`) before any timing loop
   starts.** If it fails, the binary exits non-zero and prints why; no
   `BenchResult` is emitted (FR6: "correctness check runs before any
   timing").
5. It allocates device buffers once, runs `--warmup` (default 10)
   untimed launches (each followed by `cudaDeviceSynchronize()`), then
   `--reps` (default 30, always >= 20 per ADR 0005) timed launches, each
   timed individually with `kernelforge::GpuTimer` (`cudaEvent`-based,
   kernel-only — H2D/D2H transfer is outside the timed region, FR4).
6. It prints one `BenchResult` JSON line to stdout.
7. `scripts/bench_driver.py` (stdlib-only: `json` + `subprocess`, no pip
   install required) reads a sweep spec from `benchmarks/configs/*.json`
   (which `variant`s and which sweep values to run, at what
   warmup/reps/seed), invokes the binary once per sweep point, and
   appends each validated JSON line to `benchmarks/raw/<family>.jsonl`.
8. `scripts/validate_results.py` validates every line of every
   `benchmarks/raw/*.jsonl` against `benchmarks/schema/bench_result.schema.json`
   using Python's `jsonschema` package before a run is considered
   complete; `scripts/test.sh` runs this automatically.

Exact commands are in the repo root `README.md` "Reproducing the
benchmarks" section; they are the literal commands used to produce every
number under `benchmarks/raw/`.

## 3. Noise control (ADR 0005)

GPU clocks are **not locked** on this development machine — verified:

```
$ nvidia-smi -lgc 210,2520
The current user does not have permission to change clocks for GPU 00000000:01:00.
Terminating early due to previous errors.
```

This is a real, disclosed WSL2-guest limitation, not a fabricated
constraint. Every `BenchResult.env.locked_clocks` is `false` and
`env.clock_lock_note` repeats this explanation, plus an observed
(unlocked) SM/memory clock snapshot taken at measurement time
(`nvidia-smi --query-gpu=clocks.sm,clocks.mem`), so a reader always knows
the clock state was dynamic, and roughly what it was, without the report
claiming a controlled environment it doesn't have.

Consequently:
- **Within one run session**, comparisons between variants (e.g. naive vs
  tiled transpose, same size, same seed, run back-to-back) are the most
  trustworthy numbers in this repo, since both variants share the same
  thermal/clock trajectory.
- **Absolute GB/s figures** should be read as "typical for this desktop
  under WSL2 boost clocks", not a locked-clock datacenter measurement.
- IQR width is reported, not discarded — a wide IQR is itself informative
  about this platform's noise floor.

## 4. Statistics

- Minimum 20 measured repetitions per (kernel, variant, size); default 30.
- Headline numbers use the **median** (robust to occasional long-tail
  stalls from WSL2/OS scheduling jitter).
- IQR (p25/p75) is always reported alongside the median.
- Full raw distribution is always committed (`raw_timings_ms`), so any
  other statistic can be recomputed later without re-running hardware.

## 5. Effective bandwidth definition

`effective_bandwidth_gb_s = bytes_moved / (stats_ms.median * 1e6)`, where
`bytes_moved` is the number of **useful** bytes the kernel is defined to
move (e.g. vector add: read x + read y + write out = 3 * n * 4 bytes;
transpose: read + write = 2 * n * 4 bytes; stride-copy: read + write per
thread = 2 * num_threads * 4 bytes, **independent of stride**). This last
property is deliberate: it is what makes `effective_bandwidth_gb_s` a
direct measurement of coalescing quality rather than of how much data
happened to move. `docs/gpu-model.md` records the device's theoretical
peak bandwidth (from `device-info`) so any effective figure can be read as
a fraction of peak.

## 6. Correctness gating

No `BenchResult` with `correctness_passed=false` is ever committed to
`benchmarks/raw/` (enforced by the benchmark binaries themselves exiting
before emitting one, and re-checked by `benchmarks/schema/bench_result.schema.json`,
which constrains `correctness_passed` to `true`). Tolerances used:
`atol=1e-5`, `rtol=1e-4` for FP32 (ADR 0003), via
`kernelforge::allclose` (`|actual - expected| <= atol + rtol * |expected|`,
elementwise).

## 7. Configs

`benchmarks/configs/*.json` are the committed, machine-readable sweep
specifications — the single source of truth for "what sizes/variants were
run, with what warmup/reps/seed" for each kernel family. `bench_driver.py`
reads these directly; nothing about a sweep is hand-typed into a script or
README that could drift from what was actually run.

## 8. Phase 1 results — naive vs. coalesced access

This section is filled in only after the process above has actually been
run, and only ever with numbers pulled from the committed
`benchmarks/raw/*.jsonl` files (hard constraint 3: never fabricate
benchmark numbers). See:

- `benchmarks/raw/transpose.jsonl` — `v1_naive` vs `v2_tiled`, same sizes,
  same seed. `v1_naive` has a coalesced read but a pathological
  (stride = rows) write; `v2_tiled` stages through shared memory so both
  the read and the write are coalesced (see
  `src/kernels/transpose/transpose_naive.cuh` /
  `transpose_tiled.cuh` for the mechanism, and
  `docs/optimization-method.md` for the hypothesis/evidence/interpretation
  loop this comparison follows).
- `benchmarks/raw/stride_copy.jsonl` — a stride sweep (1, 2, 4, 8, 16, 32,
  64, 128) at fixed thread count, isolating "distance between consecutive
  threads' memory accesses" as the only variable. `effective_bandwidth_gb_s`
  at each stride is the headline signal.
- `benchmarks/raw/vector_add.jsonl` / `saxpy.jsonl` — unit-stride
  baselines across a size sweep; both are contiguous end-to-end (no
  pathological variant exists for them in this repo), included as the
  "fully healthy" reference point effective-bandwidth figures from the
  other two families can be compared against.

Filled-in narrative with the actual measured numbers lives in this
section once `scripts/run_all_benchmarks.sh` has been run against real
hardware — see the bottom of this file for the as-run summary and links
to the exact commit's raw JSON.

### As-run summary (2026-08-17, RTX 4090 / sm_89 / CUDA 12.6, WSL2)

Produced by `scripts/run_all_benchmarks.sh` → `benchmarks/raw/{vector_add,saxpy,transpose,stride_copy}.jsonl`
(30 committed records total, all `correctness_passed=true`, all schema-valid
per `scripts/validate_results.py`). Device's theoretical peak bandwidth at
measurement time: **1008.1 GB/s** (`device-info`, memory clock 10501 MHz x
384-bit bus x2).

**Transpose, naive vs. tiled, `effective_bandwidth_gb_s` (median of 30 reps):**

| n (n x n) | v1_naive | v2_tiled | speedup |
|---:|---:|---:|---:|
| 256   | 102.4  | 128.0  | 1.25x |
| 512   | 187.3  | 434.0  | 2.32x |
| 1024  | 248.2  | 908.6  | 3.66x |
| 2048  | 275.6  | 1213.6 | 4.40x |
| 4096  | 277.1  | 721.0  | 2.60x |
| 8192  | 272.4  | 708.7  | 2.60x |

At the two largest, fully HBM-bound sizes (4096, 8192 -- working set well
beyond the 72 MiB L2), `v2_tiled` sustains **~70-72% of theoretical peak
bandwidth** versus `v1_naive`'s **~27%**, a consistent **~2.6x** speedup
from changing exactly one thing: staging the transpose through shared
memory so the global write is coalesced instead of strided (see
`src/kernels/transpose/transpose_naive.cuh` vs `transpose_tiled.cuh`).
The smaller sizes (512-2048) show even larger ratios because part or all
of the working set is L2-resident, which *disproportionately* helps
`v2_tiled` (fewer, larger transactions benefit more from cache reuse than
many small ones do) -- itself a real, interesting effect, not noise: it's
consistent across the run and explained by the L2 size relative to the
buffer size at each n.

**Stride/coalescing microbenchmark, fixed at 2^20 threads, `effective_bandwidth_gb_s`:**

| stride | effective BW (GB/s) | relative to stride=1 |
|---:|---:|---:|
| 1   | 1376.1 | 1.00x (baseline) |
| 2   | 978.1  | 0.71x |
| 4   | 630.2  | 0.46x |
| 8   | 327.7  | 0.24x |
| 16  | 93.1   | 0.068x |
| 32  | 67.2   | 0.049x |
| 64  | 43.9   | 0.032x |
| 128 | 39.8   | 0.029x |

Exactly the same number of *useful* bytes moves at every stride (2 x 2^20 x
4 bytes = 8 MiB read + 8 MiB written); only the address distance between
consecutive threads changes. Effective bandwidth collapses **34.6x** from
stride 1 to stride 128 -- the clearest possible isolated demonstration of
the memory-coalescing mechanism, with the transpose comparison above
showing the same mechanism inside a realistic kernel. **This is the
evidence behind the Phase 1 exit criterion**: "repo empirically
demonstrates contiguous vs. pathological access."

**Vector add / SAXPY (unit-stride baseline, no pathological variant):**

| n | vector_add BW (GB/s) | saxpy BW (GB/s) |
|---:|---:|---:|
| 4,096      | 14.0   | 16.0  |
| 65,536     | 195.8  | 205.7 |
| 1,048,576  | 2048.0 | 2048.0 |
| 16,777,216 | 918.7  | 918.8 |
| 67,108,864 | 926.3  | 925.2 |

The smallest size is launch-overhead-dominated (kernel too short to
amortize launch latency); 1,048,576 elements (12 MiB total across 3
buffers) is entirely L2-resident, which is why it exceeds the 1008.1 GB/s
HBM peak (a real GPU phenomenon -- L2 is faster than HBM -- not an error);
the two largest sizes are genuinely HBM-bound and land at **~92% of
theoretical peak**, confirming these unit-stride kernels are close to the
practical ceiling and make a valid "fully healthy" reference point for the
transpose/stride numbers above.

Full per-repetition data, exact launch configs, environment metadata, and
correctness tolerances for every one of these numbers are in the raw
`.jsonl` files; nothing above is restated anywhere without a path back to
one of those committed records.
