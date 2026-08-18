# IO Analysis: Memory Accounting Before Optimizing (Phase 2)

This is the roadmap Phase 2 deliverable ("Theoretical bytes/FLOPs
calculation + measured profiler baseline **before optimizing**" -> exit
criterion "Baseline bottleneck hypothesis documented"). Sections 1-4 are
the theoretical model plus the already-committed Phase 0/1 numbers,
written to end in one explicit, falsifiable hypothesis (SS4) -- section 5
is what `benchmarks/configs/memory_accounting.json` and an Nsight Systems
capture actually showed, checked against it. Section 6 repeats the
exercise for V2 (tiled), and section 7 is what
`benchmarks/configs/tiled_comparison.json` actually showed. This keeps the
"hypothesis before measurement" discipline `docs/decisions/` and
`benchmarks/methodology.md` already establish for this repo. **One
correction note up front, in the spirit of not hiding a mistake:** an
earlier draft of SS4/SS6 below stated the predicted direction of the
effective-bandwidth trend backwards, and claimed V2's improvement over V1
should grow with sequence length. Re-deriving both algebraically before
running any benchmark caught both errors; what is written below is the
corrected version, and SS5/SS7 check the corrected claims, not the
original wrong ones.

## 1. Restating the compute cost (from docs/attention-math.md SS4)

Per (batch, head) slice, `S = QK^T` and `O = PV` are each `2 * S^2 * D`
FLOPs (a `D`-term dot product per output entry, multiply + add). Total
compute per slice: `4 * S^2 * D` FLOPs, independent of which variant
computes it -- V0, V1, V2, V3, V4 all compute the same mathematical
function (docs/attention-math.md SS1), so this number is fixed for the
whole ladder. This repo's kernels mask causally with `-inf` post-hoc, not
by skipping work, so causal does not change this count.

## 2. Irreducible bytes and the "ideal" arithmetic intensity

The irreducible traffic for one slice -- one read each of `Q`, `K`, `V`,
one write of `O`, all `[S, D]` float32 -- is `4 * S * D * 4 bytes = 16 * S
* D` bytes (this is exactly `bench_schema.BenchResult.bytes_moved`'s
formula, per `benchmarks/methodology.md` SS5). An attention implementation
that achieved this exactly (never materializing anything else) would have
arithmetic intensity:

```
AI_ideal(S, D) = FLOPs / bytes = (4 * S^2 * D) / (16 * S * D) = S / 4  FLOP/byte
```

Notably **independent of `D`** -- it grows linearly with sequence length
only. This is the standard argument for why long-context attention gets
*more* compute-bound (not less) as sequences get longer, provided nothing
forces extra memory traffic along the way.

## 3. This GPU's roofline ridge point

Captured from this repo's own target hardware (RTX 4090, sm_89, ADR 0001;
same physical card as KernelForge's ADR 0001/`docs/gpu-model.md`, whose
already-verified `nvidia-smi`-derived numbers are cited below rather than
re-derived independently):

- **Peak FP32 (non-tensor-core) compute:** `128 SMs * 128 FP32 cores/SM *
  2 FLOPs/FMA * 2.520 GHz = 82.58 TFLOP/s` (128 SMs and the 2520 MHz boost
  clock per KernelForge's `docs/gpu-model.md`, sourced there from
  `scripts/run_device_info.sh` on this same machine; 128 FP32 cores/SM is
  the published Ada Lovelace SM configuration; confirmed independently
  here via `torch.cuda.get_device_properties(0).multi_processor_count ==
  128` in this repo's own venv).
- **Peak HBM bandwidth:** `1008.1 GB/s` (`2 * 10501 MHz * 384-bit / 8 /
  1e9`, KernelForge `docs/gpu-model.md`).
- **Ridge point** (compute peak / bandwidth peak): `82.58e12 / 1008.1e9 =
  81.9 FLOP/byte`. Below this line an implementation is memory-bound on
  this GPU; above it, compute-bound.

From SS2: `AI_ideal(S, D) = S / 4` crosses `81.9` at `S = 4 * 81.9 ~= 328`.
An ideal (never-over-materializing) implementation of this project's
problem would cross from memory-bound to compute-bound on this GPU around
`S ~= 328`, independent of `D` and batch/head count.

## 4. V1's theoretical no-reuse HW-traffic model, and the Phase 2 hypothesis

V1's three kernels (`src/flashlite/cuda_naive/attention_naive.cu`) are
"one thread per output element, no shared-memory reuse" by design. Counting
global-memory transactions at the granularity of individual elements the
kernel code actually touches (no L1/L2 cache reuse credited -- a
pessimistic upper bound on real traffic, not a claim that real hardware
moves exactly this many bytes), per (batch, head) slice:

| Kernel | Reads | Writes | Source |
|---|---:|---:|---|
| 1. `compute_scores_kernel` | `2*S^2*D` (Q+K, D-length loop/thread, no reuse) | `S^2` (scores) | every one of the `S^2` threads independently loops `d=0..D-1` reading `q[q_row+d]`, `k[k_row+d]` |
| 2. `softmax_rows_kernel` | `3*S^2` (row read once for max, once for sum, once for normalize) | `S^2` (in place) | three separate full passes over each row -- read directly from the code, not assumed |
| 3. `weighted_sum_kernel` | `2*S^2*D` (P+V, S-length loop/thread, no reuse) | `S*D` (out) | every one of the `S*D` threads independently loops `j=0..S-1` reading `probs[...]`, `v[...]` |

Total per slice: `4*S^2*D + 5*S^2 + S*D` elements; in float32 bytes,
`bytes_v1(S,D) = 4*(4*S^2*D + 5*S^2 + S*D)`.

```
AI_v1(S, D) = (4*S^2*D) / bytes_v1(S, D) = S*D / (4*S^2*D + 5*S^2 + S*D)
```

For `S >> D`, this approaches `AI_v1(S,D) ~= D / (4*D + 5)` --
**independent of `S`**:

| `D` | `AI_v1` (large-`S` limit) |
|---:|---:|
| 32  | 32/133  ~= 0.241 FLOP/byte |
| 64  | 64/261  ~= 0.245 FLOP/byte |
| 128 | 128/517 ~= 0.248 FLOP/byte |

**Compare to SS2's `AI_ideal(S,D) = S/4`, which grows with `S`.** V1's
modeled arithmetic intensity does not grow with `S` at all -- pinned near
`0.24-0.25 FLOP/byte`, about 330x below this GPU's `81.9` ridge point,
regardless of sequence length.

### What this predicts about the *reported* `effective_bandwidth_gb_s` field

`BenchResult.effective_bandwidth_gb_s = bytes_moved_irreducible / time`
(`benchmarks/methodology.md` SS5) -- deliberately the *irreducible* count
divided by real measured time, not real traffic divided by time. If V1
becomes genuinely HBM-bandwidth-bound once its actual (much larger) working
set exceeds this GPU's 72 MiB L2 cache (`docs/gpu-model.md`), then, writing
`f` for whatever fraction of peak HBM bandwidth it achieves on its real
traffic:

```
real_bytes(S,D)      = irreducible_bytes(S,D) / AI_v1(S,D)     [both are 4*S^2*D-ish scale at large S -> real_bytes is QUADRATIC in S]
time(S,D)             = real_bytes(S,D) / (f * BW_peak)         [quadratic in S too]
effective_bandwidth(S) = irreducible_bytes(S,D) / time(S,D)
                       = AI_v1(D) * f * BW_peak / S              [because irreducible_bytes is LINEAR in S, time is QUADRATIC]
```

**The reported `effective_bandwidth_gb_s` should fall roughly as `1/S`
once L2 is exceeded -- i.e., `effective_bandwidth_gb_s * S` should
approach a roughly constant value at large `S`.** (An earlier draft of
this document stated the opposite direction; re-deriving it here, term by
term, is what caught that error before any benchmark ran.)

### HYPOTHESIS (written 2026-08-17, before benchmarks/configs/memory_accounting.json was run)

1. V1 is memory-bandwidth-bound at every sequence length this repo tests
   (arithmetic intensity `~0.24-0.25 FLOP/byte`, far below the `81.9`
   ridge point), unlike an ideal implementation whose intensity grows with
   `S`.
2. Once `S` is large enough that V1's `O(S^2)` scores buffer (plus
   Q/K/V/O) no longer fits the 72 MiB L2 cache -- crossing at `S=2048` for
   the `(B=1, H=8, D=64)` shape this sweep uses (`151.0 MB` total working
   set, per `benchmarks/configs/memory_accounting.json`'s header comment)
   -- `effective_bandwidth_gb_s * seq_len` should approach a roughly
   constant value (the `1/S` decay derived above), and should look visibly
   different from the smaller, L2-resident shapes.
3. V1's median latency should grow *faster* than the irreducible `O(S*D)`
   traffic alone would predict once HBM-bound -- consistent with (though
   an `S`-only sweep at fixed `D` cannot, by itself, distinguish between)
   either the `O(S^2)` softmax-materialization term or the `O(S^2*D)`
   QK^T/PV term dominating, since both are `O(S^2)` at fixed `D`. SS7's
   head-dim sweep (varying `D` at fixed `S`) is what actually separates
   these two hypotheses.
4. V1 should not be competitive with V0 (which dispatches to already
   well-optimized ATen/cuBLAS-backed `matmul`/`softmax` kernels) at these
   larger shapes -- Phase 0/1 only found V1 winning at the smallest tested
   shapes on kernel-launch-overhead grounds, and nothing about tiling the
   *un-tiled* V1 kernel between here and there should change that.

## 5. Phase 2 measured baseline

Produced by `python scripts/run_benchmarks.py --config
benchmarks/configs/memory_accounting.json` -> `benchmarks/raw/attention.jsonl`
(10 records, `v0_reference`/`v1_naive`, `B=1 H=8 D=64` non-causal, `S` in
`[256, 512, 1024, 2048, 4096]`, 20 reps each), plus two `nsys profile`
captures under `profiling/nsight-systems/`. Both required the GPU
(real hardware measurement), run under the repo's GPU-benchmark lock.

### 5.1 Effective-bandwidth sweep (v0_reference vs v1_naive, B=1 H=8 D=64, non-causal, 20 reps)

All figures below are read directly from the committed
`benchmarks/raw/attention.jsonl` (median of 20 reps each; `bw*S` is
`effective_bandwidth_gb_s * seq_len`, computed here to check SS4's
predicted `1/S` decay):

| `S` | working set vs 72 MiB L2 | v0 median ms | v0 GB/s | v1 median ms | v1 GB/s | v1 `bw*S` |
|---:|---|---:|---:|---:|---:|---:|
| 256  | 4.2 MB, fits   | 0.1630 | 12.87 | 0.1403  | 14.94 | 3825.6 |
| 512  | 12.6 MB, fits  | 0.1192 | 35.18 | 0.5093  | 8.24  | 4216.8 |
| 1024 | 41.9 MB, fits  | 0.2053 | 40.86 | 4.1231  | 2.03  | 2083.3 |
| 2048 | 151.0 MB, **exceeds** | 1.0329 | 16.24 | 13.9054 | 1.21  | **2471.0** |
| 4096 | 570.4 MB, **exceeds** | 6.6703 | 5.03  | 55.6073 | 0.60  | **2471.6** |

**The `1/S` prediction holds tightly in exactly the regime it was derived
for.** Once the working set exceeds the L2 cache (`S=2048` and `S=4096`),
`v1_naive`'s `bw*S` product is `2471.0` and `2471.6` -- a `0.03%`
difference across a 2x change in `S`. Below that crossing (`S<=1024`,
working set still L2-resident), the product is neither close to that
constant nor monotonic (`4216.8` at `S=512`, `2083.3` at `S=1024`) --
consistent with cache-assisted traffic and small-shape effects dominating
there instead of the HBM-bound law SS4 derived, exactly as SS4 predicted
should be visible as a qualitative difference between the two regimes.

**Latency itself grows faster than `O(S)` once HBM-bound:** going from
`S=2048` to `S=4096` (a 2x increase in `S`) increases `v1_naive`'s median
latency by `55.6073/13.9054 = 4.00x` -- an exact `S^2` signature, matching
SS4 point 3's prediction that latency should scale beyond what the
irreducible `O(S*D)` traffic alone implies once memory-bound. This
sequence-length-only sweep (fixed `D=64`) cannot by itself say whether the
`O(S^2*D)` (QK^T/PV) or `O(S^2)` (softmax materialization) term is more
responsible -- SS7's head-dim sweep addresses that.

**V1 vs V0:** `v1_naive` is *slower* than `v0_reference` at every point
`S>=512` in this sweep (`4.3x` at `S=512`, up to `20.1x` at `S=1024`,
`13.5x` at `S=2048`, `8.3x` at `S=4096` -- not a monotonic ratio, but
never close), and only modestly faster at the smallest shape (`S=256`:
`0.1403` vs `0.1630` ms). This matches hypothesis point 4: V1 is not
competitive with V0 at any of these larger shapes. (`v0_reference`'s own
`bw*S` does not follow a simple law across this range either --
`3294`/`18013`/`41838`/`33265`/`20605` -- consistent with cuBLAS/ATen's
matmul and softmax kernels doing their own internal tiling/blocking this
document does not attempt to model; V0 is out of scope for this repo's
optimization ladder, so this is noted, not investigated further.)

### 5.2 Nsight Systems capture

`profiling/nsight-systems/v1_naive_S4096_D64.nsys-rep` and
`v2_tiled_S4096_D64.nsys-rep` (B=1, H=8, S=4096, D=64, non-causal, 5 warmup
+ 20 timed iterations via `scripts/profile_kernels.py`), with a
`cuda_api_sum` extract committed alongside each as CSV.

**What this confirms:** `Num Calls` for `cudaLaunchKernel` is exactly
`75` for both captures (`25` forward calls x `3` kernels/call) --
independent, profiler-level confirmation that both V1 and V2 launch
precisely the documented 3-kernel pipeline per forward call, matching
`attention_naive.cuh`/`attention_tiled.cuh`'s documented design.

**What this could NOT capture, and why:** Nsight Systems' GPU-side kernel
execution timeline (`cuda_gpu_kern_sum`, which would give a genuine
measured per-kernel-name time breakdown -- exactly what SS5.1's
`S`-only-sweep ambiguity above needs to resolve without waiting for SS7)
returned **no data** for either capture ("SKIPPED: ... does not contain
CUDA kernel data"), despite `cudaLaunchKernel` calls being visible at the
CUDA-API level. This was verified to be an environment limitation, not
specific to this repo's kernels: a control capture of a bare
`torch.randn(2048,2048,device="cuda"); x@x` loop (no flashlite code at
all) shows the identical "SKIPPED" result. This is the same class of
constraint KernelForge already documented for `ncu`
(`../02-kernelforge/profiling/metric-guide.md`, `docs/decisions/0006`) --
GPU-side CUPTI activity-record collection is restricted under this
machine's WSL2/driver configuration (typically fixed by a Windows-host
registry change requiring administrator access and a reboot, out of scope
for this session) -- documented here rather than worked around or silently
dropped. `ncu` (Nsight Compute) is also still not installed on this
machine, per KernelForge's existing finding. Consequently, **this
document's per-kernel time attribution (SS5.1's `S^2` vs `S^2*D` question,
SS7's discussion of why V2's measured speedup is much smaller than its
predicted arithmetic-intensity improvement) is argued from the GpuTimer
end-to-end latency numbers and the theoretical byte-count model, not from
a profiler-measured per-kernel split** -- a real limitation of what could
be measured in this environment, stated plainly rather than papered over.

### 5.3 Does the data support the hypothesis?

Yes, on every point that could be checked with the tools available:
memory-bound arithmetic intensity (point 1, by construction/model, `~330x`
below the ridge point at every tested shape), the `1/S` effective-bandwidth
decay once L2 is exceeded (point 2, confirmed to `0.03%` between two
independent shapes), the `S^2` latency scaling once HBM-bound (point 3,
confirmed exactly, `4.00x` for a `2x` change in `S`), and V1's lack of
competitiveness with V0 at these shapes (point 4, confirmed at every point
`S>=512`). The one question this sweep could not resolve on its own --
whether the `O(S^2)` softmax term or the `O(S^2*D)` QK^T/PV term dominates
V1's real traffic -- is addressed using the head-dim sweep in SS7, since
GPU-side profiler data to answer it directly was not available (SS5.2).

## 6. V2 (tiled) theoretical prediction (written before Phase 3's benchmark ran)

Tiling (ADR 0007, `kAttnTileDim = 32`, written `T` below) changes ONLY
kernels 1 and 3's Q/K and P/V global-memory read terms -- each `2*S^2*D`
no-reuse read term becomes `2*S^2*D/T` (KernelForge `gemm_tiled.cuh`-style
"TILE-fold reduction" argument). Kernel 1/3's *write* terms are unchanged.
Kernel 2 (softmax) is byte-for-byte unchanged (`3*S^2` reads + `S^2`
writes, same as V1 -- see `attention_tiled.cu`'s header comment on why).

```
bytes_v2(S, D) = 4 * ( 2*S^2*D/T + S^2   [kernel 1]
                      + 3*S^2 + S^2       [kernel 2, unchanged]
                      + 2*S^2*D/T + S*D ) [kernel 3]
               = 4 * ( 4*S^2*D/T + 5*S^2 + S*D )
```

For `S >> D`, `AI_v2(S, D) ~= D / (D/8 + 5) = 8*D / (D + 40)`:

| `D` | `AI_v1` (large-`S` limit) | `AI_v2` (large-`S` limit, `T=32`) | ratio |
|---:|---:|---:|---:|
| 32  | 0.241 | 8*32/72   = 3.56 | ~14.8x |
| 64  | 0.245 | 8*64/104  = 4.92 | ~20.1x |
| 128 | 0.248 | 8*128/168 = 6.10 | ~24.6x |

**Correction versus an earlier draft:** this ratio does **not** grow with
`S`. Both `bytes_v1` and `bytes_v2` are dominated by their own `S^2`-scale
terms once `S` is large (the `S*D` output-write term becomes negligible in
both), so `bytes_v1(S,D)/bytes_v2(S,D)` approaches a limit that depends on
`D` alone, not `S` -- the table above already IS the large-`S` limit, not
a lower bound that keeps climbing. The corrected prediction is: **V2's
arithmetic-intensity improvement over V1 should be roughly constant across
`S` (once `S` is large enough for both to be `S^2`-term-dominated) and
should grow with `D`** (more reduction opportunity per tile relative to
the fixed `T=32` and the `D`-independent softmax term).

Two further predictions follow from kernel 2 being completely unimproved:

- **V2 should remain well below the `81.9 FLOP/byte` ridge point** (still
  memory-bound, just less catastrophically than V1) -- the untouched
  softmax term puts a ceiling on how much a tiling-only change to kernels
  1/3 can improve overall intensity.
- **The measured end-to-end latency speedup should be considerably smaller
  than the `~15-25x` arithmetic-intensity improvement above.** Kernel 2's
  traffic, and any shared per-call overhead (memory allocation, kernel
  launch dispatch), is common to both V1 and V2 and completely unaffected
  by tiling; an Amdahl's-law-style ceiling applies regardless of how much
  kernels 1/3 individually improve. The *qualitative* pattern that should
  survive is the *ordering* by `D` (bigger `D` -> bigger measured speedup),
  even though the *magnitude* should not match the AI-ratio table.

## 7. Phase 3 measured results

Produced by `python scripts/run_benchmarks.py --config
benchmarks/configs/tiled_comparison.json` -> `benchmarks/raw/attention.jsonl`
(18 records: a sequence-length sweep and a head-dim sweep, both causal,
`B=1 H=8`, 20 reps each). Full table in `benchmarks/methodology.md` SS9;
this section states what the numbers mean for SS6's predictions.

**Sequence-length sweep (`B=1 H=8 D=64`, causal, 20 reps):**

| `S` | v1_naive ms | v2_tiled ms | v2 vs v1 |
|---:|---:|---:|---:|
| 128  | 0.0485 | 0.1506 | v2 is **3.1x slower** |
| 512  | 0.3195 | 0.1935 | v2 is 1.65x faster |
| 1024 | 1.1540 | 0.7044 | v2 is 1.64x faster |
| 2048 | 5.1461 | 3.1251 | v2 is 1.65x faster |

**Head-dim sweep (`B=1 H=8 S=512`, causal, 20 reps):**

| `D` | v1_naive ms | v2_tiled ms | v2 vs v1 |
|---:|---:|---:|---:|
| 32  | 0.1707 | 0.1157 | 1.48x faster |
| 64  | 0.3195 | 0.1935 | 1.65x faster |
| 128 | 0.5791 | 0.3492 | 1.66x faster |

**One SS6 prediction holds cleanly; the other holds only partially, and the
partial version is more interesting:**

1. **Speedup is roughly constant across `S` once `S>=512`** (`1.64-1.65x`,
   not growing with `S`) -- matching the corrected "the AI ratio is
   `S`-independent at large `S`" claim, not the original (wrong) "grows
   with `S`" draft.
2. **Speedup grows with `D` from 32 to 64 (`1.48x -> 1.65x`), then
   essentially flattens from 64 to 128 (`1.65x -> 1.66x`)** -- SS6's
   AI-ratio table predicted continued, substantial growth across this same
   range (`14.8x -> 20.1x -> 24.6x`, still climbing steadily from `D=64` to
   `D=128`). The measured speedup does not track that continued climb at
   all. This is a *stronger* version of the Amdahl's-law point SS6 made,
   not a weaker one: it says kernel 2's (and shared overhead's) unimproved
   traffic is *already* the binding constraint on end-to-end latency by
   `D=64`, so the substantial additional arithmetic-intensity headroom
   tiling buys between `D=64` and `D=128` in kernels 1/3 alone goes almost
   entirely unrealized end-to-end. **This also answers SS5.1's open
   question** (whether V1's `O(S^2)` softmax term or `O(S^2*D)` QK^T/PV
   term dominates its real traffic): the fact that tiling produces *any*
   measurable speedup at all (`1.48-1.66x`, not `~1.0x`) means the QK^T/PV
   term is a real, non-negligible contributor to V1's actual latency, not
   a pessimistic upper bound that vanishes in practice -- while the
   speedup plateauing well below the `14.8-24.6x` AI-ratio ceiling, and
   specifically *not* continuing to grow past `D=64`, means kernel 2 (plus
   shared per-call overhead) is the larger and increasingly dominant
   remaining contributor. Both terms matter; neither alone explains the
   measured numbers, which is the most honest conclusion this repo's
   available tools (no working GPU-side kernel profiler, SS5.2) can
   support.
3. **The `S=128` crossover (V2 slower than V1) is real and expected, not a
   bug:** `tests/correctness/test_tiled_attention.py` verifies V2 is
   *correct* at `S=1` through `S=257` (including this exact shape's
   neighborhood), so this is a genuine performance crossover, not a
   correctness gap. `kAttnTileDim=32` launches a full `32x32`-thread block
   per `32x32` output tile regardless of how small `S` is (ADR 0007); at
   `S=128` there are only `4x4=16` such blocks total for the whole `(B=1,
   H=8)` score-tensor grid, each doing proportionally more
   shared-memory-staging bookkeeping relative to actual useful work than
   V1's simpler one-thread-per-output-element launch. This is exactly the
   kind of launch-configuration tradeoff the spec defers to Phase 6's
   empirical tile-size sweep (D5) -- not a Phase 3 defect.
4. **V2 is still not competitive with V0** at these shapes (e.g. `S=2048,
   D=64`: `v0_reference=1.6630ms` vs `v2_tiled=3.1251ms`, `v2` `1.88x`
   slower) -- consistent with the spec's explicit allowance that this
   project succeeds even if the custom kernel never beats the platform's
   built-in attention; V2 improves meaningfully over V1 without claiming
   to have closed that gap.

## 8. What this motivates next

The qualitative conclusion this document exists to support: **the naive
materialized-attention pipeline pays `O(S^2)`-family global-memory traffic
for a mathematical function whose irreducible traffic is `O(S*D)`, and
Phase 3's tiling improves but does not remove this** -- it reduces the
QK^T/PV portion of that traffic by a `D`-dependent, `S`-independent factor
(measured `1.5-2x` end-to-end, `~15-25x` in the underlying arithmetic
intensity of just the two tiled kernels), while leaving the softmax
kernel's materialize/re-read traffic completely untouched, and SS7 point 2
found direct evidence that BOTH terms are real, non-negligible
contributors to V1's measured latency. Phase 4 (online softmax) exists
specifically to remove the softmax term's own materialize/re-read cost by
computing the row-normalization running statistic incrementally,
tile-by-tile, instead of requiring the full row up front
(`docs/online-softmax.md`); Phase 5 (fused IO-aware attention) then
combines both to avoid materializing the `[S, S]` matrix at all, which is
the only way this project's arithmetic intensity can approach SS2's
`AI_ideal = S/4` line and this GPU's `81.9 FLOP/byte` ridge point at
realistic sequence lengths.

## 9. Addendum: what Phases 4/5 actually found (written after both landed)

`benchmarks/methodology.md` SS10 has the full writeup; this is the
one-paragraph connection back to this document's own predictions, so a
reader finishing this file does not have to guess whether SS8's roadmap
panned out. **The memory-traffic story above is confirmed exactly as
predicted**: `v3_online_softmax` (Phase 4) reduces kernel 2's traffic (two
row passes instead of three) but, exactly as SS7 point 2's Amdahl's-law
argument already anticipated for V2, this produces only a small (`~1-3%`
at `seq_len >= 512`) end-to-end latency improvement over V2, since kernel 2
was never the *only* non-negligible term. `v4_fused` (Phase 5) is where
this section's central claim -- "the only way this project's arithmetic
intensity can approach `AI_ideal`... is to avoid materializing `[S, S]` at
all" -- gets tested directly: it measurably achieves close-to-linear
peak-memory scaling (`~1.4-1.7x` per `seq_len` doubling, versus the
materializing variants' `~2.2-3.5x` trending toward the `4x` quadratic
signature), the literal removal this document's whole argument was
building toward. **What SS8 did not predict, and this repo does not hide:**
V4 is currently *slower* than V2/V3 at every tested shape (`benchmarks/methodology.md`
SS10.3) -- a real, measured cost of the register-pressure/parallelism
tradeoff its one-thread-per-query-row design makes (ADR 0011), not a
correctness problem. Memory-traffic reduction and wall-clock speed are
related but distinct axes; this project's premise (SS1: "attention
performance is dominated by memory movement") is about *why* naive
attention is slow at scale, not a guarantee that every memory-reducing
rewrite is automatically also the fastest implementation on a given GPU
without further tuning -- which is exactly what Phase 6 exists to pursue.
