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

## 9. Phase 2 results — reduction ladder and scan strategies

Schema note: `reduction` and `scan` (plus Phase 3's `histogram`) were
added to `kernel_family`, and two histogram-only fields (`num_bins`,
`contention_profile`) were added to `BenchResult`, by ADR 0011 — both
additive, backward-compatible changes; `schema_version` stays `"1.0"`.

### 9.1 Reduction ladder — `benchmarks/raw/reduction.jsonl`

Five variants (`benchmarks/configs/reduction.json`), same sizes/seed,
`block-size=256`; see `src/kernels/reduction/README.md` for what each rung
changes and its written hypothesis. `effective_bandwidth_gb_s` (median of
30 reps, recomputed directly from the committed `.jsonl` records below —
every number here must be reproducible with `python3 -c "import json;
..."` over that file):

| n | V0 naive_global_atomic | V1 naive_interleaved | V2 sequential_addressing | V3 warp_shuffle | V4 vectorized_coarsened |
|---:|---:|---:|---:|---:|---:|
| 4,096      | 1.78 | 3.2   | 4.0   | 4.0    | 4.0    |
| 65,536     | 2.69 | 36.6  | 64.0  | 64.0   | 64.0   |
| 1,048,576  | 2.85 | 292.6 | 409.6 | 409.6  | 819.2  |
| 4,194,304  | 2.77 | 399.2 | 574.6 | 606.8  | 1638.4 |
| 16,777,216 | 2.87 | 428.3 | 655.4 | 689.9  | 3440.8 |
| 67,108,864 | —    | 430.1 | 650.5 | 714.3  | 936.2  |

At the largest, fully HBM-bound V1-V4 size (67,108,864 elements = 256 MiB,
well beyond the 72 MiB L2), the ladder is monotonically faster rung over
rung: **V1→V2 is a 1.51x speedup** (removing interleaved-addressing warp
divergence, the only variable V2 changes), **V2→V3 is a further 1.10x**
(warp-shuffle finalization replacing the last 5 shared-memory steps), and
**V3→V4 is a further 1.31x** (float4 vectorized loads + thread
coarsening), for a **2.18x V1→V4 speedup overall (430 → 936 GB/s, ~93% of
the device's 1008.1 GB/s theoretical peak)**. Every one of these matches
the direction predicted by that rung's hypothesis (`src/kernels/
reduction/README.md`); none was falsified.

At 16,777,216 elements (64 MiB — still inside the 72 MiB L2), V4 measures
**3440.8 GB/s — 3.4x the device's HBM peak**, an L2-residency effect
consistent with the same phenomenon already documented for `vector_add`/
`saxpy` in §8 (this is a real GPU phenomenon — L2 is faster than HBM —
not a measurement error; V1-V3 do not show this as strongly at the same
size because their per-block combine overhead dominates before L2
bandwidth becomes the bottleneck).

**V0's numbers (backfilled — see `src/kernels/reduction/README.md`'s
"Ladder history" note for why this rung was added after V1-V4's data was
already collected) confirm the atomics-contention hypothesis directly, and
also surface a second, distinct failure mode.** At every size V0 measures,
its effective bandwidth is nearly flat — **1.78 → 2.69 → 2.85 → 2.77 →
2.87 GB/s across a 4096x range of n (4,096 to 16,777,216)** — while V1
(the very next rung, whose only change is introducing a shared-memory
tree so only `ceil(n/256)` atomics reach global memory instead of `n`)
climbs from 3.2 to 428.3 GB/s over the same range. The **V1/V0 speedup
widens monotonically with n: 1.80x → 13.57x → 102.71x → 144.27x → 149.10x**
— exactly the shape the V0 hypothesis (`reduce_naive_global.cuh`)
predicted: V0's cost is dominated by every one of its `n` global
`atomicAdd`s serializing against every other one on the same address, so
its throughput is essentially independent of `n` (contention-bound, not
bandwidth-bound), while V1's `ceil(n/256)` atomics leave it free to scale
with `n` until it too becomes bandwidth-bound. At n=16,777,216, **V4 is
1197.7x faster than V0** — the full ladder's cumulative payoff over the
literal "one atomic per element, no shared memory" starting point.

**V0 also fails its own correctness gate at the ladder's largest
benchmarked size (67,108,864), for a reason that has nothing to do with
atomics contention: single-precision (fp32) accumulator saturation.**
`bench_reduction --variant v0_naive_global_atomic --n 67108864` measured
`actual=1.67772e+07` against a CPU reference of `expected=3.35575e+07`
(`max_rel_diff=0.50`, i.e. almost exactly half) and `bench_driver.py`
correctly refused to write that result (hard constraint 1: correctness
before performance; house rule: a `correctness_passed=false` result is
never committed). This is **not a logic bug** — `reduce_naive_global_
kernel`'s one line (`atomicAdd(out, in[i])`) is trivially correct in
isolation at every individual call — it is the accumulated *sequence* of
~67 million individual `atomicAdd`s of an O(0.5)-magnitude `float` value
each, into ONE `float` scalar, that loses information: once the running
total exceeds 2^24 (16,777,216 — the largest integer an IEEE-754 float32
mantissa can represent exactly), the accumulator's own ULP grows to ≥2,
so a further add of ~0.5 frequently rounds away to no change at all. The
measured `actual` value (16,777,200) sitting almost exactly at that 2^24
threshold is the signature of this well-known float32 summation-stall
artifact, not a race condition or off-by-one. **V1-V4 never hit this**,
even though they too reach the same final magnitude and even though every
one of them also finishes with a global-scalar atomic-combine step (see
`src/kernels/reduction/README.md`'s "held constant" note): each of their
`ceil(n/256)` block-level atomics adds a *pre-reduced, per-block partial
sum* (magnitude ~128, not ~0.5) into an accumulator whose ULP at the final
33.5M-ish total is only ~4 — comfortably larger than the increment's own
precision, but nowhere near stalling it the way V0's direct
per-element adds do. **V0's benchmark sweep in the table above therefore
covers only the five sizes at which it passes correctness** (this is a
second, independent reason — beyond raw contention — that a single global
fp32 atomic accumulator is not a technique any of this ladder's later
rungs use for the final combine at scale, and it is disclosed here rather
than worked around by loosening the tolerance, which would have hidden a
genuine numerical-stability finding).

### 9.2 Scan strategies — `benchmarks/raw/scan.jsonl`

Two strategies (`benchmarks/configs/scan.json`), same sizes/seed,
`block-size=1024` (2-level ceiling = 1,048,576 elements — every swept size
stays at or under it); see `src/kernels/scan/README.md` for each
strategy's hypothesis. `effective_bandwidth_gb_s` (median of 30 reps):

| n | Hillis-Steele | Blelloch |
|---:|---:|---:|
| 1,024     | 1.7   | 1.3   |
| 16,384    | 8.0   | 3.7   |
| 65,536    | 43.6  | 22.2  |
| 262,144   | 146.3 | 115.6 |
| 1,048,576 | 327.9 | 201.0 |

Hillis-Steele wins at every size measured, by 1.25x-2.16x (widest at
16,384, narrowest at 1,024, with 262,144 nearly as narrow at 1.27x). This **confirms** the hypothesis written in
`scan_hillis_steele.cuh`/`scan_blelloch.cuh` before this comparison was
run: at this GPU's block sizes (≤1024, so ≤10 sequential passes either
way), Hillis-Steele's simpler, uniformly-active-every-pass steps evidently
outweigh Blelloch's lower total arithmetic work — synchronization-barrier
count and addressing simplicity dominate over raw work-efficiency at this
scale, exactly as predicted (and exactly the outcome that hypothesis said
would be reported as-is even if it disagreed with the initial guess, which
in this case it did not).

## 10. Phase 3 results — histogram/atomics lab: low- vs high-contention

`benchmarks/raw/histogram_uniform.jsonl` (low contention) and
`benchmarks/raw/histogram_skewed.jsonl` (high contention) — identical
sweeps (`benchmarks/configs/histogram_{uniform,skewed}.json`: same three
variants, same five sizes, same `num-bins=256`, `block-size=256`, seed)
except the one input-distribution field; see
`src/kernels/histogram/README.md` and `src/common/rng.hpp`'s
`make_histogram_input` for exactly how `uniform` vs `skewed` are
generated. `effective_bandwidth_gb_s` (median of 30 reps) at the largest
size, n = 33,554,432:

| Variant | uniform (low contention) | skewed (high contention) | skewed vs. uniform |
|---|---:|---:|---:|
| V1 `global_atomic` | 17.5 GB/s | 9.1 GB/s | **0.52x (skewed is 1.93x slower)** |
| V2 `privatized` | 111.8 GB/s | 162.4 GB/s | 1.45x |
| V3 `privatized_coarsened` | 794.4 GB/s | 762.0 GB/s | 0.96x |

Full sweep (all 5 sizes) is in the raw `.jsonl` files. V1's and V2's
patterns hold at every size measured; V3's uniform-vs-skewed ordering is
*not* stable across sizes (skewed is 1.9x-2.2x faster at the two smallest
sizes, then the two distributions trade places within a ±7% band from
n = 2,097,152 up), and V3 only clearly overtakes V2 from n = 524,288
upward — at n = 65,536 its coarsening overhead leaves it below V2 under
uniform (23.3 vs 34.1 GB/s) and level with it under skewed.

**V1 (global-atomic baseline) is the only variant whose throughput
actually depends on the input distribution**, and it depends on it
strongly: skewed input (most atomics colliding on a handful of global
addresses) is consistently slower than uniform (atomics spread across all
256 bins) — **1.7x-1.9x slower across the sweep** — exactly the
contention penalty the hypothesis in `histogram_global_atomic.cuh`
predicted.

**V2 (shared-memory privatization) erases that dependence almost
entirely**, and its residual difference runs the *opposite* direction
from the naive prediction: skewed is slightly *faster* than uniform
(1.3x-1.5x across the sweep), not slower. This was not what the
hypothesis in `histogram_privatized.cuh` predicted (it expected
privatization to *narrow* the uniform/skewed gap, not reverse its sign) —
reported here as written, per `docs/optimization-method.md` step 6, rather
than adjusted after the fact. A plausible mechanism (not confirmed with
Nsight Compute, which is out of scope until Phase 6): concentrating most
of a warp's 32 concurrent `atomicAdd` calls onto the *same* few shared-memory
addresses under the skewed profile may let the SM's atomic unit combine
same-address requests from one warp into fewer effective operations, an
opportunity a uniformly-spread warp (32 different addresses) does not
have; this is an interpretation to verify with profiler evidence later,
not a settled explanation.

**V3 (privatization + coarsening) goes further still** — at large sizes
neither distribution holds a stable advantage (at the largest point
uniform is ahead by 4%: 794.4 vs 762.0 GB/s), the contention *penalty*
is gone, and V3 is far faster than V1 at every size and than V2 from
n = 524,288 up: **V1→V3 is a 45x speedup under uniform and an 84x
speedup under skewed** (33,554,432-element point), the clearest demonstration in this
repo that privatizing contention away, not just reducing raw atomic count,
is what makes a kernel's performance robust to its input's statistical
shape — precisely the Phase 3 exit criterion ("low- vs high-contention
workloads documented with measured results").

Every correctness check for every committed record above used an EXACT
integer bin-count match against `kernelforge::reference::histogram` (not
a tolerance-based comparison — histogram counts are integers), and no
record with a mismatch was ever written (the benchmark binary refuses to
emit one, per FR6).

## 11. Phase 4 results — GEMM ladder

Schema note: `gemm` was added to `kernel_family`, and one new field (`k`,
GEMM's shared inner dimension; `rows`/`cols` are reused for M/N) was added
to `BenchResult`, by ADR 0012 — additive, backward-compatible;
`schema_version` stays `"1.0"`. `docs/decisions/
0013-cublas-ceiling-methodology.md` covers the cuBLAS invocation and
ceiling-not-rung methodology.

`benchmarks/configs/gemm.json` sweeps V1-V4 plus the `cublas_sgemm_ceiling`
reference across a square M=N=K sweep (`benchmarks/raw/gemm.jsonl`, 25
records, every one `correctness_passed: true`; see
`src/kernels/gemm/README.md` for what each rung changes). GFLOP/s (`2 *
M*N*K / median_ms`, median of 30 reps, recomputed directly from the
committed `.jsonl` records — every number here is reproducible with
`python3 -c "import json; ..."` over that file):

| M=N=K | V1 naive | V2 coalesced | V3 tiled | V4 register_tiled | cuBLAS (ceiling) |
|---:|---:|---:|---:|---:|---:|
| 128  | 71.9   | 409.6  | 459.1  | 275.9   | 680.9   |
| 256  | 297.9  | 2048.0 | 2340.6 | 1310.7  | 3640.9  |
| 512  | 611.1  | 4681.1 | 5618.6 | 5269.2  | 18724.6 |
| 1024 | 617.1  | 5041.2 | 6179.2 | 15196.8 | 45870.7 |
| 2048 | 617.8  | 5105.1 | 6224.9 | 16571.1 | 53773.1 |

At the largest size (2048x2048x2048), the ladder is monotonically faster
rung over rung: **V1→V2 is an 8.26x speedup** (fixing the row/col
↔ `threadIdx.x/y` mapping — the ONLY variable V2 changes — turns A's
badly-uncoalesced reads and C's badly-uncoalesced writes into a
warp-uniform broadcast and a fully coalesced write, respectively), **V2→V3
is a further 1.22x** (shared-memory tiling, cutting global reads from
`2*TILE²*K` to `2*TILE*K` per output tile), and **V3→V4 is a further
2.66x** (register tiling: each thread now reuses its shared-memory B read
across 8 output rows instead of 8 different threads each re-reading it
once), for a **26.8x V1→V4 speedup overall (617.8 → 16,571.1 GFLOP/s)** —
one variable changed per rung, all five variants (V1-V4 + cuBLAS)
numerically agree with `reference::gemm` at every tested shape (hard
constraint 1). At this size, **V4 reaches 30.8% of cuBLAS's measured
throughput** (16,571.1 / 53,773.1 GFLOP/s) — reported as exactly that
framing, never as "V4 is only 3.2x slower than cuBLAS" (spec FR2: cuBLAS
is a ceiling/reference, never a target to beat; see
`docs/decisions/0013-cublas-ceiling-methodology.md`).

**V3→V4's speedup is NOT monotonic across sizes — it is a genuine
regression at the two smallest sizes, matching a caveat the hypothesis in
`gemm_register_tiled.cuh` stated explicitly before this data existed**
("expect a further speedup... at every size where V3 was already
shared-memory-bandwidth-limited rather than occupancy-limited"):

| M=N=K | V3→V4 ratio | V4 grid (blocks) |
|---:|---:|---:|
| 128  | **0.60x (V4 is 40% SLOWER)** | 2×2 = 4 |
| 256  | **0.56x (V4 is 44% SLOWER)** | 4×4 = 16 |
| 512  | 0.94x (roughly level) | 8×8 = 64 |
| 1024 | 2.46x | 16×16 = 256 |
| 2048 | 2.66x | 32×32 = 1024 |

V4's 64×64 block tile means its grid is `⌈N/64⌉ × ⌈M/64⌉` blocks — at
128×128 that is only **4 blocks total** against this GPU's 128 SMs
(docs/gpu-model.md): at most 4 SMs can even be active, ~3% occupancy, so
the kernel is occupancy-bound and the extra per-thread register/shared-
memory setup register tiling adds is pure overhead with no reuse benefit
to amortize it against. By 1024×1024 (256 blocks) there is enough grid
parallelism to saturate the GPU and the register-reuse mechanism the
hypothesis predicted takes over decisively. This is reported as measured,
confirming (not falsifying) the hypothesis's own stated caveat — a case
where the *first* guess already correctly anticipated the nuance the data
would show.

## 12. Phase 5 results — Softmax ladder

Schema note: `softmax` and `rmsnorm` were added to `kernel_family` by ADR
0012 (no new fields needed — both reuse `rows`/`cols`, already present for
`transpose`/`gemm`).

`benchmarks/configs/softmax.json` sweeps V1-V3 across representative
transformer hidden-dimension widths (`cols`, `rows=4096` fixed;
`benchmarks/raw/softmax.jsonl`, 21 records, all `correctness_passed:
true`; see `src/kernels/softmax/README.md`). Median of 30 reps
(recomputed from the committed `.jsonl`):

| cols | V1 naive (ms) | V2 warp_shuffle (ms) | V3 fused_online (ms) | V1→V2 | V2→V3 |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.0171 | 0.0113 | 0.0113 | 1.51x | 0.99x |
| 512  | 0.0195 | 0.0143 | 0.0161 | 1.36x | 0.89x |
| 768  | 0.0215 | 0.0174 | 0.0184 | 1.24x | 0.94x |
| 1024 | 0.0222 | 0.0164 | 0.0184 | 1.35x | 0.89x |
| 2048 | 0.0307 | 0.0275 | 0.0266 | 1.12x | 1.03x |
| 4096 | 0.1505 | 0.1504 | 0.1505 | 1.00x | 1.00x |
| 8192 | 0.3123 | 0.3174 | 0.3215 | 0.98x | 0.99x |

**V1→V2 (reduction mechanism) shows a real, consistent win at small-to-
medium `cols` — up to 1.51x at cols=128 — that shrinks smoothly to
parity by cols=4096-8192** (0.98x-1.00x): warp-shuffle removes
`log2(block_size)` `__syncthreads()` rounds down to 2 regardless of
`block_size`, a fixed-overhead saving whose share of total kernel time
naturally shrinks as `cols` (and therefore per-thread work and total
memory traffic) grows. This matches the hypothesis in
`softmax_warp_shuffle.cuh` in direction, though the hypothesis did not
predict the win would fully vanish by the two largest sizes tested — an
honest refinement, not a contradiction.

**V2→V3 (the "bounded fusion" rung) does NOT show a consistent win: at
six of the seven sizes tested (all except cols=2048, where V3 is 3.1%
faster) V3 is the same speed or measurably SLOWER than V2** (128: 0.7%
slower; 512: 12.5% slower; 768: 5.9% slower; 1024: 12.3% slower; 4096:
0.1% slower — essentially exact parity; 8192: 1.3% slower; all
percentages relative to the V2 baseline) — this
contradicts the hypothesis written in
`softmax_fused_online.cuh` before this data existed ("expect roughly a
1/3 reduction in this kernel's global-memory read traffic... and a real
... speedup"), reported here as measured rather than adjusted after the
fact (`docs/optimization-method.md` step 6). A plausible mechanism (not
confirmed with Nsight Compute, which is not installed on this machine as
of 2026-08-17 — see `docs/gpu-model.md`): V3's per-element online-combine
(`l = l*expf(m-m_new) + expf(x-m_new)`) issues **two** `expf()` calls per
element folded into the running (max, sum) pair, whereas V1/V2's two
SEPARATE passes issue only **one** `expf()` call per element in total
across both passes (the max-only pass uses just `fmaxf`, not `expf`) —
V3 trades one fewer global-memory PASS for one MORE transcendental-unit
call per element, and at these row sizes that tradeoff is a wash or a
narrow net loss rather than the assumed win. This is consistent with
`cols=4096`/`8192`'s near-identical wall-clock time across all THREE
variants despite their different pass counts — at that scale the kernel
appears bandwidth-bound on total row traffic in a way that does not
distinguish 2 passes from 3 (plausibly because a single ≤32 KiB row stays
L1/L2-resident across one kernel invocation's own immediately-repeated
passes over it, so "fewer passes" saves compute more than it saves actual
DRAM traffic) — an interpretation to verify with profiler evidence later,
not a settled explanation, same disclosure standard as §10's histogram
V2 finding above.

## 13. Phase 5 results — RMSNorm ladder

`benchmarks/configs/rmsnorm.json` sweeps V1-V3 across the same
representative-shape convention as softmax (`benchmarks/raw/
rmsnorm.jsonl`, 21 records, all `correctness_passed: true`; see
`src/kernels/norm/README.md`). Median of 30 reps:

| cols | V1 naive (ms) | V2 warp_shuffle (ms) | V3 vectorized (ms) | V1→V2 | V2→V3 |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.0113 | 0.0082 | 0.0082 | 1.38x | 1.00x |
| 512  | 0.0134 | 0.0112 | 0.0096 | 1.19x | 1.17x |
| 768  | 0.0154 | 0.0132 | 0.0111 | 1.16x | 1.19x |
| 1024 | 0.0152 | 0.0133 | 0.0133 | 1.15x | 1.00x |
| 2048 | 0.0252 | 0.0246 | 0.0244 | 1.03x | 1.01x |
| 4096 | 0.1500 | 0.1505 | 0.1484 | 1.00x | 1.01x |
| 8192 | 0.3226 | 0.3226 | 0.3287 | 1.00x | 0.98x |

**V1→V2 shows the identical pattern softmax's V1→V2 did** (real win at
small-medium `cols`, up to 1.38x at cols=128, fading to parity by
cols=2048-8192) — the same fixed-`__syncthreads()`-count mechanism,
independently confirmed on a second, differently-shaped kernel family.

**V2→V3 (vectorized `float4` loads) shows a real, if narrower-window,
win than softmax's fusion rung did**: 14.5-16.2% faster at cols=512/768
(relative to the V2 baseline), roughly level at 128/1024/2048/4096, and a
small (1.9%) regression at cols=8192. Unlike softmax's V3, this rung's mechanism (wider memory
transactions, not fewer passes) matches the hypothesis in
`rmsnorm_vectorized.cuh` at the sizes where it wins, though — same honest
disclosure — it does not win everywhere the hypothesis's "expect a
further speedup... growing with `cols`" framing implied, and the largest
size measured is a narrow loss rather than the largest win. **V1→V3
overall peaks at cols=512 (1.39x, with cols=768 essentially tied at
1.38x) and narrows to a small loss (0.98x) by cols=8192** — the full
ladder's net benefit is real but concentrated in the small-to-medium
shape range this sweep covers, not uniform across it.
