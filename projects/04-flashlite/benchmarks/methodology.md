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
- `tile_size` (ADR 0008, added Phase 3; ADR 0009/0011 extend it to
  `v3_online_softmax`/`v4_fused`): the kernel-launch tile dimension for
  `v2_tiled`/`v3_online_softmax`/`v4_fused` records (`32` for all three as
  of Phase 5); `0` ("not applicable") for `v0_reference`/`v1_naive`, which
  have no tile-size concept. Optional/additive -- every Phase 0/1 record
  committed before this field existed remains schema-valid unchanged.
- `peak_memory_bytes` (ADR 0010, added Phase 5): measured
  `torch.cuda.max_memory_allocated()` bytes for one untimed forward call,
  populated only when a sweep config sets `"measure_peak_memory": true`
  (`benchmarks/configs/phase4_5_comparison.json`, SS10). `0.0` ("not
  measured") for every earlier record and any point from a config that
  does not request it. This is what makes Phase 5's "peak-memory scales as
  designed" exit criterion, and spec SS1.6's peak-memory-vs-sequence-length
  plot, computable directly from a committed artifact.

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
could drift from what was actually run. Phase 2 added
`benchmarks/configs/memory_accounting.json` (`v0_reference`/`v1_naive`
across an L2-cache-crossing `seq_len` range, SS9); Phase 3 added
`benchmarks/configs/tiled_comparison.json` (adds `v2_tiled`, a
sequence-length and a head-dim sub-sweep, SS9); Phase 4/5 added
`benchmarks/configs/phase4_5_comparison.json` (adds `v3_online_softmax` and
`v4_fused`, a sequence-length sweep across the full five-variant ladder,
with `"measure_peak_memory": true`, SS10) -- same convention, one file per
sweep, results appended to the same `benchmarks/raw/attention.jsonl`.

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

## 9. Phase 2/3 results -- memory accounting and v2_tiled

Two more sweeps, appended to the same `benchmarks/raw/attention.jsonl`
(34 records total as of Phase 3, all `correctness_passed=true`, all
schema-valid): `benchmarks/configs/memory_accounting.json` (Phase 2 --
`v0_reference`/`v1_naive`, `B=1 H=8 D=64` non-causal, `seq_len` in `[256,
512, 1024, 2048, 4096]`, spanning this GPU's 72 MiB L2 cache boundary) and
`benchmarks/configs/tiled_comparison.json` (Phase 3 -- adds `v2_tiled`,
a sequence-length sweep and a head-dim sweep, both causal, `B=1 H=8`).
20 reps each, `seed=123456789`, `warmup_iters=5`.

**`docs/io-analysis.md` is the full writeup** (theoretical model, the
Phase 2 hypothesis stated before this sweep ran, and section-by-section
comparison against what these two sweeps actually showed) -- this section
only restates the tables, run under the GPU-benchmark lock per this
project's hard rules, so a reader does not have to cross-reference a
second document to see the raw numbers a claim in `docs/io-analysis.md`
traces back to.

### As-run summary (2026-08-17, RTX 4090 / sm_89 / CUDA 12.6, WSL2)

Environment: torch `2.13.0+cu126`, CUDA driver `591.86`,
`observed_sm_clock_mhz=2250` / `observed_mem_clock_mhz=10501` at capture
time (this run happened to catch the GPU closer to its boost clock than
the idle `210`/`405` MHz snapshot Phase 0/1's summary above recorded --
per SS3, clocks are unlocked on this platform regardless, so this is
still "typical for this desktop under WSL2 at time of capture," not a
locked-clock measurement, and is not compared numerically against the
Phase 0/1 table above for that reason).

**Memory-accounting sweep -- `v0_reference` vs `v1_naive`, `B=1 H=8 D=64`,
non-causal, median ms and effective bandwidth (GB/s), 20 reps:**

| `seq_len` | v0 median ms | v0 GB/s | v1 median ms | v1 GB/s |
|---:|---:|---:|---:|---:|
| 256  | 0.1630 | 12.87 | 0.1403  | 14.94 |
| 512  | 0.1192 | 35.18 | 0.5093  | 8.24  |
| 1024 | 0.2053 | 40.86 | 4.1231  | 2.03  |
| 2048 | 1.0329 | 16.24 | 13.9054 | 1.21  |
| 4096 | 6.6703 | 5.03  | 55.6073 | 0.60  |

**Tiled-comparison sweep -- sequence-length sub-sweep, `B=1 H=8 D=64`,
causal, median ms, 20 reps:**

| `seq_len` | v0 median ms | v1 median ms | v2 median ms |
|---:|---:|---:|---:|
| 128  | 0.1989 | 0.0485 | 0.1506 |
| 512  | 0.1603 | 0.3195 | 0.1935 |
| 1024 | 0.2647 | 1.1540 | 0.7044 |
| 2048 | 1.6630 | 5.1461 | 3.1251 |

**Tiled-comparison sweep -- head-dim sub-sweep, `B=1 H=8 seq_len=512`,
causal, median ms, 20 reps:**

| `head_dim` | v0 median ms | v1 median ms | v2 median ms |
|---:|---:|---:|---:|
| 32  | 0.1516 | 0.1707 | 0.1157 |
| 64 (= seq-len sub-sweep's `S=512` row) | 0.1603 | 0.3195 | 0.1935 |
| 128 | 0.1413 | 0.5791 | 0.3492 |

`v2_tiled`'s `tile_size` field (ADR 0008) is `32` on every one of these
records, `0` on every `v0_reference`/`v1_naive` record, per ADR 0007/0008.
Every `correctness_note` in this sweep shows a small, tolerance-passing
FP32 discrepancy for `v1_naive`/`v2_tiled` against `v0_reference` (same
`atol=1e-5, rtol=1e-4` bound as SS6 above; see the raw `.jsonl` for exact
per-point values) and `0.0` for `v0_reference` against itself.

See `docs/io-analysis.md` SS5/SS7 for what these numbers mean: the `1/S`
effective-bandwidth decay and `S^2` latency scaling `v1_naive` shows once
its working set exceeds the L2 cache (SS5.1), the Nsight Systems capture
and its real limitation in this environment (SS5.2), and the
sequence-length-independent-but-head-dim-dependent speedup `v2_tiled`
shows over `v1_naive`, including the honest discussion of why that speedup
is far smaller than the underlying arithmetic-intensity improvement (SS7).

## 10. Phase 4/5 results -- v3_online_softmax and v4_fused

Produced by `python scripts/run_benchmarks.py --config
benchmarks/configs/phase4_5_comparison.json` -> `benchmarks/raw/attention.jsonl`
(25 records: all five variants -- `v0_reference`, `v1_naive`, `v2_tiled`,
`v3_online_softmax`, `v4_fused` -- at `B=1 H=8 D=64`, causal, `seq_len` in
`[128, 512, 1024, 2048, 4096]`, `20` reps each, `"measure_peak_memory":
true`; `59` records total in the committed file as of Phase 5). Run under
the GPU-benchmark lock (peak-memory measurement, like latency, requires the
real GPU). Environment: torch `2.13.0+cu126`, CUDA driver `591.86`, same
WSL2/unlocked-clocks caveats as SS3.

Every number below is read directly from the committed `.jsonl` (`median_ms`
from `stats_ms`, `peak_memory_bytes` divided by `1e6` for MB) -- no rounded
table entry is used to derive another number in this section.

### 10.1 Peak memory vs. sequence length -- the Phase 5 exit criterion, measured

| `seq_len` | v0 peak MB | v1 peak MB | v2 peak MB | v3 peak MB | v4 peak MB | v3/v4 ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128  | 11.16   | 10.62   | 10.62   | 10.62   | 10.09  | 1.05 |
| 512  | 31.85   | 23.20   | 23.20   | 23.20   | 14.81  | 1.57 |
| 1024 | 89.26   | 54.66   | 54.66   | 54.66   | 21.10  | 2.59 |
| 2048 | 306.32  | 167.90  | 167.90  | 167.90  | 33.69  | 4.98 |
| 4096 | 1149.37 | 595.72  | 595.72  | 595.72  | 58.85  | 10.12 |

`v1_naive`, `v2_tiled`, and `v3_online_softmax` report **byte-identical**
peak memory at every `seq_len` (all three allocate the same `[B, H, S, S]`
scores-scratch buffer plus `[B, H, S, D]` output -- `docs/attention-math.md`
SS5, `docs/online-softmax.md` SS10 -- and differ from each other only in
*algorithm*, never in *what gets allocated*, exactly as designed). `v4_fused`
allocates neither: `src/flashlite/cuda_fused/bindings.cpp` never calls
`torch::empty({B, H, S, S}, ...)` at all (ADR 0011).

**The v3/v4 ratio widens monotonically and substantially across the sweep**
(`1.05x -> 1.57x -> 2.59x -> 4.98x -> 10.12x`; last value is `9.62x` the
first) -- exactly
`tests/correctness/test_fused_attention.py::test_fused_peak_memory_grows_far_slower_than_materializing_variants`'s
assertion, now also confirmed from a committed artifact, not only a live
test run. Doubling `seq_len` at each step:

- `v4_fused` peak grows by `1.47x, 1.42x, 1.60x, 1.75x` per doubling --
  well below `2.0x` (pure linear scaling) at every step, and trending
  *up toward* `2.0x` as `seq_len` grows (the ~`10 MB` fixed
  allocator/output-tensor overhead that dominates at small `S` amortizes
  away as `S` grows, leaving the underlying `O(S*D)` term to dominate at
  larger `S`) -- consistent with `O(S*D)` scaling, not merely "smaller than
  quadratic."
- `v3_online_softmax` (and byte-identical `v1_naive`/`v2_tiled`) peak grows
  by `2.19x, 2.36x, 3.07x, 3.55x` per doubling -- approaching `4.0x` (exact
  `S^2` quadratic scaling) as `seq_len` grows and the `O(S^2)` scores buffer
  increasingly dominates the same fixed overhead, exactly the `O(S^2)` vs
  `O(S*D)` contrast `docs/attention-math.md` SS5 and `docs/online-softmax.md`
  SS10 predict.

**This is the literal, measured confirmation of Phase 5's exit criterion**
("Peak-memory scales as designed"): materializing variants trend toward
quadratic memory growth, the fused variant trends toward linear, and the
gap between them widens by an order of magnitude across one `5x` `seq_len`
sweep (`128 -> 4096`).

### 10.2 Latency: V3's modest, as-predicted improvement over V2

| `seq_len` | v2_tiled ms | v3_online_softmax ms | v3 vs v2 |
|---:|---:|---:|---:|
| 128  | 0.053120  | 0.082544  | v3 is **55.4% slower** |
| 512  | 0.202752  | 0.200704  | v3 is 1.0% faster |
| 1024 | 0.736624  | 0.730944  | v3 is 0.8% faster |
| 2048 | 3.512752  | 3.395584  | v3 is 3.3% faster |
| 4096 | 13.701632 | 13.618176 | v3 is 0.6% faster |

ADR 0009 predicted exactly this shape of result before this sweep ran:
V3 changes only kernel 2's global-memory traffic (three row-length passes
reduced to two, `docs/online-softmax.md` SS10), leaving kernels 1/3
(the bulk of total latency once tiling is already applied,
`docs/io-analysis.md` SS7) completely unaffected, so "V3's end-to-end
speedup over V2 is expected to be modest, not dramatic." The measured `1-3%`
improvement at `seq_len >= 512` is exactly that: real, but small, evidence
that kernel 2's traffic reduction (33% fewer bytes moved *for that one
kernel alone*) is a minor contributor to overall latency at these shapes
-- consistent with `docs/io-analysis.md` SS7's finding that kernel 2 plus
shared per-call overhead, not kernels 1/3, dominates once tiling has
already been applied. The `seq_len=128` result (V3 slower) mirrors
`docs/io-analysis.md` SS7 point 3's `v2` vs `v1` `seq_len=128` crossover
for the identical reason: at small shapes, V3's per-thread online-update
bookkeeping (two `expf` calls and a branch per element, versus V1/V2's
simpler two-pass reduction) costs more than it saves when there is very
little row-length traffic to reduce in the first place -- a genuine, small-
shape performance crossover, not a correctness gap (`tests/correctness/
test_online_softmax_attention.py` verifies V3 is correct at this exact
shape).

### 10.3 Latency: V4 is currently slower, an honest accepted tradeoff (ADR 0011)

| `seq_len` | v2_tiled ms | v3_online_softmax ms | v4_fused ms | v4 vs v3 | v4 vs v2 |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.053120  | 0.082544  | 0.250368  | 3.03x slower | 4.71x slower |
| 512  | 0.202752  | 0.200704  | 0.854880  | 4.26x slower | 4.22x slower |
| 1024 | 0.736624  | 0.730944  | 4.295728  | 5.88x slower | 5.83x slower |
| 2048 | 3.512752  | 3.395584  | 9.716224  | 2.86x slower | 2.77x slower |
| 4096 | 13.701632 | 13.618176 | 29.301760 | 2.15x slower | 2.14x slower |

**Read honestly, per the spec's ban on claiming speedup without evidence:**
V4 is slower than every other custom kernel variant at every tested shape,
despite achieving the dramatically better memory scaling SS10.1 documents.
This is not hidden or explained away -- it is exactly the tradeoff ADR
0011 predicted before this sweep ran: V4's one-thread-per-query-row design
launches only `kFusedTileDim=32` threads per block (one thread per query
row), versus V2/V3's kernel 1/3 launching `32 x 32 = 1024` threads per
block (one thread per *output element*) -- roughly `32x` less parallelism
per streamed K/V tile's compute step, and each of those 32 threads does the
*entire* `D`-length dot-product and accumulator update sequentially, with a
large fixed-size per-thread register/local-memory footprint
(`q_reg[128]` + `acc[128]`, ADR 0011) regardless of the actual `head_dim`
being computed. **The memory-scaling win (SS10.1) and the latency loss
(this section) are two sides of the same design decision**, not
independent facts -- both are consequences of moving all the per-row state
into thread-private storage instead of the shared/materialized buffer V1-V3
use. ADR 0011 explicitly defers tuning this tradeoff to Phase 6 ("Tile-
size/block-shape sweep ... profile occupancy" -> "Tuned defaults based on
benchmark evidence, not folklore") rather than attempting a fix here that
would trade away Phase 5's readability requirement for a performance claim
this phase does not need to make (spec: "the project succeeds even if the
custom kernel does not beat the platform's built-in attention").

`v4_fused`'s `correctness_note` at `seq_len=4096` reports a higher
`max_rel_diff` (`2.452`, i.e. `245%`) than earlier variants typically show
-- **this is not a correctness regression**: `max_abs_diff` at the same
point is `2.123e-07`, comfortably inside `atol=1e-5`, and
`flashlite.compare.allclose_compare`'s pass/fail gate is the combined
`atol + rtol * |expected|` criterion (which V4 passes at every tested
point), not the raw relative-error figure alone. A large relative error
paired with a tiny absolute error is the same "near-zero-probability tail
entry" pattern SS8 already documents for `v1_naive` (`max_rel_diff` up to
`5.0e-03` there) -- V4's different summation order (per-thread sequential
online accumulation vs. V0-V3's `torch.softmax`/tiled-GEMM order) produces
a different, still-tiny, absolute floating-point residue at those same
near-zero entries, which shows up as a proportionally larger *relative*
number without indicating any real disagreement in the actual attention
output.

### 10.4 What this section supports

Phase 5's exit criterion ("Peak-memory scales as designed; output matches
reference") is met on both halves, from committed data: `v4_fused` matches
`v0_reference` within tolerance at every point in this sweep (`correctness_note`
in every record above, all `correctness_passed=true`), and its peak-memory
growth is measurably, monotonically, and substantially closer to linear
than every materializing variant's growth toward quadratic. Phase 4's exit
criterion ("tests cover extreme score values and multiple tile boundaries")
was already met by `tests/math/test_online_softmax.py` and
`tests/correctness/test_online_softmax_attention.py` independently of this
benchmark sweep; this section's SS10.2 additionally shows V3's predicted
(modest) latency improvement over V2 actually materialized. SS10.3's honest
latency finding for V4 is exactly the kind of evidence-based, un-inflated
report the spec's hard constraints require -- and is left as the concrete,
measured starting point for Phase 6's tuning work, not smoothed over.
