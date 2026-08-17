# Histogram / Atomics Lab (Phase 3)

Bins an `n`-element FP32 array (values conceptually in `[0, 1)`) into
`num_bins` integer counters: `bin = clamp(floor(x * num_bins), 0, num_bins
- 1)`. Three GPU variants, each changing exactly one variable from the
previous rung (hard constraint 4), all validated against
`kernelforge::reference::histogram` (`src/common/reference_ops.cpp`)
before any timing is trusted.

| Rung | File | What changes vs. previous rung | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| V1 `v1_global_atomic` | `histogram_global_atomic.cuh/.cu` | — (baseline) | Every element's `atomicAdd` goes straight to global memory; no privatization. Contention scales directly with how many elements collide on the same bin. |
| V2 `v2_privatized` | `histogram_privatized.cuh/.cu` | Where atomics land: block-private shared-memory histogram, merged once per block | Moves per-element atomics to shared memory (far higher throughput, lower latency); global atomic traffic drops from `n` to `num_blocks * num_bins`. |
| V3 `v3_privatized_coarsened` | `histogram_privatized_coarsened.cuh/.cu` | Elements processed per thread (`--coarsen`, default 8) | Fewer, busier blocks -> proportionally less global-merge atomic traffic than V2, for the same privatization strategy. |

## Low- vs high-contention input distributions (Phase 3 exit criterion)

The actual "lab" in this lab is running all three variants above against
**two different input distributions** that are identical in every other
respect (same `n`, `num_bins`, block size, seed) — see
`kernelforge::make_histogram_input` (`src/common/rng.hpp`):

- **`uniform` (low contention).** Values ~ Uniform(0, 1) directly. After
  binning, load spreads roughly evenly across all `num_bins` bins — a
  given bin's counter is touched by only ~`n/num_bins` elements over the
  whole launch, so collisions on any one memory address within the same
  instant are comparatively rare.
- **`skewed` (high contention).** Values = `uniform^24`, which
  concentrates the resulting distribution's mass near 0 (a standard
  power-law-style skew, similar in shape to real skewed categorical data).
  After binning, the large majority of elements land in the first handful
  of bins out of `num_bins` — most threads' atomics target the SAME few
  addresses.

`benchmarks/configs/histogram_uniform.json` and
`benchmarks/configs/histogram_skewed.json` run the identical
(variant x size) sweep against each distribution; results land in
`benchmarks/raw/histogram_uniform.jsonl` / `histogram_skewed.jsonl`. Every
committed record's `contention_profile` field (ADR 0011) states which
distribution produced it, so a reader never has to infer this from the
filename alone. See `benchmarks/methodology.md` §10 for the as-run
before/after table (V1 vs V2 vs V3, uniform vs skewed) and its
interpretation — filled in only after `scripts/run_all_benchmarks.sh` has
actually produced the numbers (hard constraint 3).

## Correctness

`tests/test_histogram.cpp` checks all three variants, against both
contention profiles, on: n = 0 (empty; every bin 0), n = 1, small
non-block-multiple n, num_bins = 1 (degenerate, every element in one
bin), num_bins not a power of two, a large multi-block n, and (V3 only)
a sweep of `elems_per_thread` including 1 (degenerates to one element per
thread, should match V2's counts exactly) and a value that does not evenly
divide a block's chunk.

## Shared memory limit

Privatized variants (V2, V3) need `num_bins * sizeof(int)` bytes of
dynamic shared memory per block. `histogram_common.cuh::
validate_num_bins_fits_shared_mem` fails loudly (hard constraint 8) if
that exceeds this GPU's 49,152-byte default per-block shared memory
(`docs/gpu-model.md`) rather than letting the CUDA runtime reject the
launch after the fact.
