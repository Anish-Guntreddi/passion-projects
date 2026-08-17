# Benchmark Methodology

**This document is written before any benchmark number in this repo is
collected**, mirroring KernelForge's methodology doc and the FlashLite
roadmap's own requirement (Phase 0 deliverable: "benchmark result schema";
Phase 1's per-task attachment: "a benchmark or profiler checkpoint"). It
defines exactly how every number under `benchmarks/raw/` was produced;
results are added to this repo only by running the process described here,
never hand-written.

## 1. Schema

Every result is one JSON object, one per line, appended to
`benchmarks/raw/attention.jsonl` (a single file for the `"attention"`
kernel family -- unlike KernelForge's several distinct kernel families,
this project has one, with multiple `variant` values). The struct that
produces it is `flashlite.bench_schema.BenchResult`
(`src/flashlite/bench_schema.py`); the equivalent JSON Schema is
`benchmarks/schema/bench_result.schema.json`. Both are committed before any
`.jsonl` file exists. Fields, briefly:

- `env`: GPU name + compute capability, CUDA driver/runtime version,
  Python/torch version, a build-route note (ADR 0005), a locked-clocks flag
  + explanatory note (see SS3), an observed (not locked) SM/memory clock
  snapshot, an OS note, and a UTC timestamp
  (`flashlite.env_capture.EnvironmentInfo.capture()`).
- Problem shape: `batch`, `heads`, `seq_len`, `head_dim`, `causal`, `dtype`
  (always `"fp32"`, ADR 0004) -- the spec's SS1.7 test-matrix dimensions,
  recorded per result rather than only implied by a config file.
- `warmup_iters`, `measured_reps`, `seed`: exactly what was run and with
  what deterministic input (`flashlite.reference.tensors.make_qkv`).
- `raw_timings_ms`: every individual repetition's kernel-only elapsed time
  (never just a mean), measured by `flashlite.timing.GpuTimer`
  (`torch.cuda.Event`-based).
- `stats_ms`: min/max/mean/median/stddev/p25/p75 computed from
  `raw_timings_ms` by `flashlite.stats.compute_stats` (linear-interpolation
  percentiles, matching KernelForge's `compute_stats` exactly).
- `bytes_moved` / `effective_bandwidth_gb_s`: see SS5.
- `gflops`: `4 * batch * heads * seq_len^2 * head_dim` FLOPs (`Q K^T` +
  `P V`, `docs/attention-math.md` SS4) divided by the median kernel time --
  meaningful for attention (unlike KernelForge's Phase 0/1 memory-bound
  kernels, where this field is 0), since attention does real compute work
  per byte moved.
- `tolerance_atol` / `tolerance_rtol` / `correctness_passed` /
  `correctness_note`: the exact tolerance used and the result of the
  correctness check that ran **before** this result's timing loop.

## 2. How a result is produced (reproducibility)

1. `scripts/build_ext.sh` installs the package (`pip install -e .`),
   building `flashlite._cuda_naive` via `torch.utils.cpp_extension`
   (ADR 0005).
2. `scripts/run_benchmarks.py` reads a sweep spec from
   `benchmarks/configs/attention.json` (which `variant`s and which
   `(batch, heads, seq_len, head_dim, causal)` points to run, at what
   `warmup_iters`/`measured_reps`/`seed`).
3. For each point, it generates deterministic `(q, k, v)` via
   `flashlite.reference.tensors.make_qkv` (seeded from the config's
   `seed`), directly on the CUDA device.
4. **It runs exactly one correctness check
   (`flashlite.compare.allclose_compare`) against `flashlite.reference.attention`
   (V0) before any timing loop starts.** If it fails, that point is skipped
   (printed to stderr) and no `BenchResult` is emitted for it (spec hard
   constraint: "never suppress a numerical mismatch to improve benchmark
   appearance").
5. It runs `warmup_iters` (config default: 5) untimed forward calls,
   `torch.cuda.synchronize()`s, then `measured_reps` (config default: 20 --
   the schema's stated minimum, matching KernelForge's ADR 0005 threshold)
   timed calls, each timed individually with `GpuTimer` (kernel-only;
   input generation and the correctness check are both outside the timed
   region).
6. It appends one `BenchResult` JSON line to `benchmarks/raw/attention.jsonl`.
7. `scripts/validate_results.py` validates every line of every
   `benchmarks/raw/*.jsonl` against
   `benchmarks/schema/bench_result.schema.json` (using the `jsonschema`
   package) plus one cross-field check the schema can't express
   (`len(raw_timings_ms) == measured_reps`); `scripts/test.sh` runs this
   automatically (with `--allow-empty` so a fresh clone's test suite does
   not require a GPU benchmark run to have already happened).

Exact commands are in the repo root `README.md` "Reproducing the
benchmarks" section.

## 3. Noise control

GPU clocks are **not locked** on this development machine -- the same
WSL2-guest limitation KernelForge already documented and verified
(`../02-kernelforge/docs/decisions/0005-benchmark-noise-control.md`):
`nvidia-smi -lgc ...` returns "The current user does not have permission to
change clocks" under this WSL2 guest. Every `BenchResult.env.locked_clocks`
is `false` and `env.clock_lock_note` repeats this explanation, plus an
observed (unlocked) SM/memory clock snapshot taken at measurement time, so
a reader always knows the clock state was dynamic, and roughly what it was.

Consequently, exactly the same caveats KernelForge's methodology doc
records apply here: within-run comparisons (same seed, same shape, `v0_reference`
vs `v1_naive` run back-to-back) are the most trustworthy numbers in this
repo; absolute GB/s and GFLOP/s figures should be read as "typical for this
desktop under WSL2 boost clocks," not a locked-clock datacenter
measurement; IQR width is reported, not discarded.

## 4. Statistics

- Minimum 20 measured repetitions per (variant, shape) point (schema-
  enforced via `measured_reps >= 20`); the committed Phase 0/1 sweep uses
  exactly 20 (`benchmarks/configs/attention.json`).
- Headline numbers use the **median** (robust to occasional long-tail
  stalls from WSL2/OS scheduling jitter).
- IQR (p25/p75) is always reported alongside the median.
- Full raw distribution is always committed (`raw_timings_ms`), so any
  other statistic can be recomputed later without re-running hardware.

## 5. Effective bandwidth definition

`effective_bandwidth_gb_s = bytes_moved / (stats_ms.median_ms * 1e6)`,
where `bytes_moved = 4 * (4 * batch * heads * seq_len * head_dim)` --  the
**useful, irreducible** byte count: one read each of `Q`, `K`, `V` and one
write of `O`, in float32 (4 bytes/element). This is deliberately NOT the
actual measured HBM traffic of the naive (V1) kernel, which additionally
writes and re-reads the full `[B, H, S, S]` score/probability matrix
(`docs/attention-math.md` SS4). That gap is the point: comparing this
formula's implied bandwidth (what a kernel that never materialized `S`
would need to move) against the *measured* effective bandwidth of `v1_naive`
is exactly the evidence Phase 2 (memory accounting) will use to justify
tiling/fusion in Phases 3 and 5 -- this field is defined now, in Phase 0,
so that comparison is available the moment Phase 2 needs it, without
redefining the metric retroactively.

## 6. Correctness gating

No `BenchResult` with `correctness_passed=false` is ever committed to
`benchmarks/raw/` (enforced by `scripts/run_benchmarks.py` itself skipping
the point before emitting one, and re-checked by
`benchmarks/schema/bench_result.schema.json`, which constrains
`correctness_passed` to `true`). Tolerances used: `atol=1e-5`, `rtol=1e-4`
for FP32 (ADR 0004), via `flashlite.compare.allclose_compare`
(`|actual - expected| <= atol + rtol * |expected|`, elementwise, the
`numpy.allclose` convention).

## 7. Configs

`benchmarks/configs/attention.json` is the committed, machine-readable
sweep specification -- the single source of truth for "what shapes/variants
were run, with what warmup/reps/seed." `scripts/run_benchmarks.py` reads it
directly; nothing about a sweep is hand-typed into a script or README that
could drift from what was actually run.

## 8. Phase 0/1 results -- v0_reference vs v1_naive

This section is filled in only after the process above has actually been
run, and only ever with numbers pulled from the committed
`benchmarks/raw/attention.jsonl` file (spec hard constraint: "never claim a
speedup without raw reproducible results"). The Phase 0/1 sweep
(`benchmarks/configs/attention.json`) is intentionally short -- 3 shapes x 2
variants -- since Phases 0-1's exit criteria are about *correctness*
("trusted PyTorch outputs exist", "custom path matches reference within
documented tolerance"), not about a benchmark headline; the naive kernel is
not expected to be fast (it fully materializes `S`/`P` and does zero
memory-access optimization, `docs/attention-math.md` SS5), and this section
does not claim otherwise. The full sequence-length/head-dim sweep the
benchmark plan (spec SS1.8) calls for is built out starting Phase 2, once
there is a tiled variant worth comparing the naive one against.

### As-run summary (2026-08-17, RTX 4090 / sm_89 / CUDA 12.6, WSL2)

Produced by `python scripts/run_benchmarks.py` -> `benchmarks/raw/attention.jsonl`
(6 committed records, all `correctness_passed=true`, all schema-valid per
`scripts/validate_results.py`). Environment: torch `2.13.0+cu126`, CUDA
driver `591.86`, `measured_reps=20` (schema minimum), `seed=123456789`.
`observed_sm_clock_mhz=210` / `observed_mem_clock_mhz=405` at capture time
-- these are idle/base clocks (the GPU was not under sustained load when
`nvidia-smi` was queried between short benchmark points), not a boosted
in-flight clock; per SS3, clocks are unlocked on this platform regardless,
so absolute figures below are read as "typical for this desktop under
WSL2", not a locked-clock measurement.

**`v0_reference` vs `v1_naive`, median kernel time (ms) and effective
bandwidth (GB/s), 20 reps:**

| shape (B, H, S, D), causal | v0_reference median ms | v0 GB/s | v1_naive median ms | v1 GB/s |
|---|---:|---:|---:|---:|
| (1, 4, 128, 64), non-causal | 0.0722 | 7.26 | 0.0635 | 8.26 |
| (1, 4, 128, 64), causal     | 0.1248 | 4.20 | 0.0304 | 17.26 |
| (2, 8, 512, 64), causal     | 0.1524 | 55.04 | 0.6052 | 13.86 |

Every `correctness_note` above 0 max-abs-diff for `v0_reference` (it
matches itself, so `max_abs_diff=0.0` exactly, as expected) and a small,
tolerance-passing FP32 discrepancy for `v1_naive` (max abs diff
`5.96e-08`-`8.94e-08`, max rel diff up to `5.0e-03`, all comfortably inside
`atol=1e-5, rtol=1e-4` since these are near-zero-probability tail entries
where the relative-error term dominates the absolute one) -- see the raw
`.jsonl` for the exact per-point `correctness_note` strings.

Reading these numbers honestly, per the spec's ban on claiming benchmark
superiority without evidence: `v1_naive` is **faster** than `v0_reference`
at the two smaller shapes (Python/kernel-launch overhead dominates
`v0_reference`'s multiple `torch.matmul`/`torch.softmax` dispatches at
these sizes, while `v1_naive` is three raw kernel launches with no
autograd/dispatcher overhead) but **6x slower** at the larger
`(2, 8, 512, 64)` causal shape, where `v0_reference`'s calls into
cuBLAS/cuDNN-backed, tuned `matmul`/`softmax` kernels start to win over
`v1_naive`'s un-tiled, one-thread-per-output-element kernels doing a full
`D`- or `S`-length loop per thread with no shared-memory reuse
(`docs/attention-math.md` SS4-SS5). This crossover -- naive wins small,
loses bigger -- is exactly the expected shape of the story Phases 2-5
exist to fix (tiling and fusion reduce the redundant global-memory traffic
`v1_naive` pays for every element, independent of kernel-launch overhead);
it is not a claim that `v1_naive` is a good kernel, only evidence of
*why* it is not, ahead of the phases that address it.

Full per-repetition timings, exact launch shapes, environment metadata,
and correctness tolerances for every point above are in
`benchmarks/raw/attention.jsonl`; nothing above is restated without a path
back to that committed record.
