# Architecture (Phases 0-5)

## Data flow

```
tests / scripts/run_benchmarks.py / scripts/profile_kernels.py
        |
        v
flashlite.reference.tensors.make_qkv   (deterministic seeded Q, K, V)
        |
        +--------------+----------------+----------------------+----------------------+
        |               |                |                       |                       |
        v               v                v                       v                       v
flashlite.reference  ops.attention(     ops.attention(         ops.attention(         ops.attention(
  .attention          variant="naive")   variant="tiled")       variant=               variant="fused")
  (V0: torch.matmul       |                  |                  "online_softmax")          |
   + torch.softmax,       v                  v                       |                       v
   CPU or CUDA, the  _cuda_naive       _cuda_tiled                   v                  _cuda_fused
   trusted ground      .attention_       .attention_          _cuda_online_softmax    .attention_fused_
   truth every          naive_forward     tiled_forward          .attention_online_     forward
   variant is           (V1)              (V2)                    softmax_forward       (V4)
   checked against)         |                  |                    (V3)                    |
        |          flashlite::validate_attention_inputs (cuda_common/shape_validate.hpp,
        |            shared by V1-V4 -- see "Shared vs. duplicated code" below; V4 additionally
        |            checks head_dim <= kMaxHeadDimFused in its own bindings.cpp, ADR 0011)
        |                   |                  |                      |                      |
        |         launch_attention_naive  launch_attention_tiled  launch_attention_    launch_attention_
        |          (attention_naive.cu)    (attention_tiled.cu)    online_softmax        fused
        |           1. compute_scores_     1. compute_scores_      (attention_online_    (attention_fused.cu)
        |              kernel                 tiled_kernel          softmax.cu)           ONE kernel:
        |              -> scores[B,H,S,S]     (shared-mem Q/K)     1. compute_scores_     streams K/V tiles,
        |           2. softmax_rows_        2. softmax_rows_         tiled_kernel          online-softmax
        |              kernel                  kernel                 (= V2's)             updates a running
        |              -> scores in place      (byte-for-byte        2. online_softmax_    per-thread output
        |           3. weighted_sum_            = V1's)                rows_kernel          accumulator --
        |              kernel                3. weighted_sum_          (combined max+sum    NO scores buffer
        |              -> out[B,H,S,D]          tiled_kernel            pass, ADR 0009)      anywhere (ADR 0011)
        |                   |                     (shared-mem P/V)   3. weighted_sum_            |
        |                   |                     |                     tiled_kernel               |
        |                   |                     |                     (= V2's)                    |
        +-------------------+---------------------+---------------------+----------------------------+
                          |
                          v
              flashlite.compare.allclose_compare
           (elementwise |actual-expected| <= atol + rtol*|expected|;
            max/mean abs error, max rel error, first-mismatch detail)
                          |
                          v
              flashlite.timing.GpuTimer (cudaEvent-based, kernel-only)
           + torch.cuda.max_memory_allocated() (peak-memory measurement,
             ADR 0010, opt-in per benchmark config, SS-below)
                          |
                          v
              flashlite.bench_schema.BenchResult + flashlite.env_capture
                          |
                          v
            benchmarks/raw/attention.jsonl  (one JSON object per line,
            validated against benchmarks/schema/bench_result.schema.json
            by scripts/validate_results.py)
```

`scripts/profile_kernels.py` (Phase 2) is a second, non-timing entry point
into the same `flashlite.ops.attention` dispatcher, meant to be wrapped by
`nsys profile` rather than measured by `GpuTimer` -- see
`docs/io-analysis.md` SS5.2 for what it captured and could not capture in
this environment.

`src/flashlite/online_softmax.py` (Phase 4) is a SEPARATE, CUDA-independent
entry point, not part of the diagram above at all: it is the standalone
"unit" spec SS1.4 requires be tested independently of any kernel
(`tests/math/test_online_softmax.py`, no `flashlite.ops` or CUDA extension
involved) -- `docs/online-softmax.md` derives the math it transcribes, and
V3's kernel 2 / V4's single kernel each realize the identical combine
identity at the CUDA level, documented in their own kernel comments rather
than by importing this Python module (there is no Python-to-CUDA code
sharing in this repo; see "Why a Python/pybind11 harness" below for the
general reason CUDA and Python stay separate implementations here).

This mirrors the spec's Part 2 architecture line ("Python benchmark/test
harness -> reference attention -> custom extension/kernel dispatcher ->
correctness comparator -> timing/memory profiler -> raw results/plots")
and KernelForge's separation of concerns (error-checked kernel launchers,
a CPU/host-side reference, a tolerance-based comparator, a benchmark
schema captured with environment metadata) -- ported from KernelForge's
CMake+CUDA C++ harness to a PyTorch/pybind11 harness because D6's resolved
default is the `torch.utils.cpp_extension` integration route (ADR 0005),
not a from-scratch CMake+libtorch build.

## Shared vs. duplicated code across variants (added Phase 3, extended Phase 4/5)

Two different kinds of "shared" code appear once a second kernel variant
(V2) exists, and this repo treats them differently on purpose:

- **Infrastructure that is not part of the optimization-ladder story is
  shared.** `src/flashlite/cuda_common/error_check.cuh` (CUDA
  error-check macros) and `cuda_common/shape_validate.hpp` (the
  TORCH_CHECK shape/dtype/layout contract, identical for every variant per
  ADR 0002/0003/0004) moved out of `cuda_naive/` in Phase 3, when
  `cuda_tiled/` needed the identical logic; V3 and V4 reuse the same shared
  header without further changes (V4 adds one variant-specific check --
  `head_dim <= kMaxHeadDimFused` -- directly in its own `bindings.cpp`
  rather than growing the shared validator with a check only one variant
  needs, ADR 0011). N copies of the same validation block silently drifting
  apart (different wording, a forgotten check) is a real risk sharing
  eliminates, and sharing it does not blur what any phase is actually
  demonstrating.
- **The kernels themselves are deliberately NOT shared.** V1's three
  kernels (`attention_naive.cu`), V2's three kernels (`attention_tiled.cu`),
  and V3's three kernels (`attention_online_softmax.cu`) are three separate,
  complete files, even though V2 and V3 share kernel 1
  (`compute_scores_tiled_kernel`) and kernel 3 (`weighted_sum_tiled_kernel`)
  byte-for-byte, and V1/V2 share kernel 2 (`softmax_rows_kernel`)
  byte-for-byte (see `attention_tiled.cuh`'s and `attention_online_softmax.cuh`'s
  headers for why each is intentionally duplicated, not extracted, even
  though either easily could be). V4 (`attention_fused.cu`) is a single new
  kernel, not a duplicate of anything -- it realizes the same online-softmax
  combine identity V3's kernel 2 uses, but applied per-key-column directly
  to freshly-computed scores instead of to an already-materialized row
  (ADR 0011). This is the spec's own instruction (Part 2: "Variants live
  side-by-side so Git history and the benchmark report show the
  optimization ladder") applied literally -- a reader should be able to
  open any one variant's `.cu` file alone and see that variant's complete
  pipeline, including whichever kernels did not change from its
  predecessor, without following an import into a shared module to find it.

## Package layout

```
src/flashlite/
  reference/attention.py     V0: trusted PyTorch reference (matmul + softmax)
  reference/tensors.py       deterministic seeded Q/K/V generation
  compare.py                 tolerance-based correctness comparator
  stats.py                   timing distribution statistics (median, IQR, ...)
  timing.py                  GpuTimer (torch.cuda.Event-based, kernel-only)
  env_capture.py             EnvironmentInfo (GPU, CUDA/torch versions, clocks)
  bench_schema.py            BenchResult dataclass + JSON (de)serialization
  online_softmax.py          Phase 4: standalone online-softmax "unit" (docs/online-softmax.md),
                               no CUDA dependency, tested independently before any kernel
  ops.py                     variant dispatcher ("reference"|"naive"|"tiled"|"online_softmax"|"fused")
  cuda_common/               shared CUDA/C++ infrastructure (Phase 3), NOT part of the
                              "variants live side-by-side" ladder story -- see above
    error_check.cuh            CUDA error-check macros (throw, don't abort)
    shape_validate.hpp          TORCH_CHECK shape/dtype/layout contract, shared by V1-V4
  cuda_naive/                V1: naive CUDA kernel + pybind11 bindings
    attention_naive.cuh        kernel/launcher declarations + shape contract
    attention_naive.cu         3-kernel naive pipeline (score/softmax/weighted-sum)
    bindings.cpp                calls cuda_common/shape_validate.hpp + pybind11 module
  cuda_tiled/                V2 (Phase 3): shared-memory-tiled CUDA kernel + bindings
    attention_tiled.cuh        kernel/launcher declarations + tiling hypothesis (ADR 0007)
    attention_tiled.cu         3-kernel tiled pipeline (tiled score/softmax/tiled weighted-sum)
    bindings.cpp                calls cuda_common/shape_validate.hpp + pybind11 module
  cuda_online_softmax/       V3 (Phase 4): kernel 1/3 = V2's, kernel 2 = combined-pass online softmax
    attention_online_softmax.cuh  kernel/launcher declarations (ADR 0009's design rationale)
    attention_online_softmax.cu   3-kernel pipeline (tiled score / ONLINE softmax / tiled weighted-sum)
    bindings.cpp                   calls cuda_common/shape_validate.hpp + pybind11 module
  cuda_fused/                V4 (Phase 5): single fused kernel, no [B,H,S,S] buffer at all
    attention_fused.cuh        kernel/launcher declarations + design rationale (ADR 0011)
    attention_fused.cu          ONE kernel: streams K/V tiles, per-thread online-softmax
                                  output accumulator, no scores buffer
    bindings.cpp                 calls cuda_common/shape_validate.hpp + its own head_dim<=128 check
tests/
  math/          Phase 0: reference correctness, RNG determinism, bench schema;
                  Phase 4: tests/math/test_online_softmax.py (CPU-only, no CUDA)
  correctness/   Phase 1: V1 vs V0; Phase 3: V2 vs V0/V1; Phase 4: V3 vs V0/V2;
                  Phase 5: V4 vs V0/V3 + measured peak-memory scaling
  edge_cases/    unsupported-shape rejection paths (V0-V4)
benchmarks/
  schema/         bench_result.schema.json (JSON Schema, mirrors bench_schema.py)
  configs/        sweep specs (attention.json; Phase 2: memory_accounting.json;
                                Phase 3: tiled_comparison.json;
                                Phase 4/5: phase4_5_comparison.json)
  raw/            committed .jsonl results (one file per kernel family)
  methodology.md  how every number under raw/ was produced
profiling/
  nsight-systems/ Phase 2: nsys captures + CUDA-API-level CSV extracts
                   (docs/io-analysis.md SS5.2 for what they could/couldn't show)
docs/
  attention-math.md    V0/V1 math derivation + materialized-form cost analysis
  online-softmax.md    Phase 4: online-softmax derivation (docs/online-softmax.md, spec SS1.4)
  io-analysis.md       Phase 2/3: theoretical memory accounting, hypothesis, measured results;
                        SS9: Phase 4/5 addendum
  architecture.md      this file
  decisions/           ADRs for the spec's open decisions D1-D4, D6, D7, plus
                        Phase 3 implementation ADRs 0007/0008, Phase 4 ADR 0009,
                        Phase 5 ADRs 0010/0011
                        (D5 itself is still deferred to Phase 6, per the spec)
```

## Why a Python/pybind11 harness instead of a from-scratch CMake+CUDA build

KernelForge (`../02-kernelforge`) is a pure CUDA C++ project: its kernels
are called from C++ test/benchmark binaries, so a hand-rolled CMake build
with no Python dependency is the natural, simplest choice there (its ADR
0009 makes the analogous "don't add unnecessary tooling" call for its test
framework). FlashLite's kernels are called from Python tensors (D6's
resolved default, ADR 0005) -- `torch.utils.cpp_extension.CUDAExtension`
already **is** the officially-supported, minimal-dependency way to compile
and link a CUDA kernel against a specific installed PyTorch/pybind11 ABI;
reimplementing that linkage by hand in a parallel CMake build would add
complexity (matching libtorch's ABI flags, finding `Python.h`/pybind11,
tracking `TORCH_CUDA_ARCH_LIST`) that buys nothing over the supported path,
so this project uses it directly rather than mirroring KernelForge's build
system literally. What IS carried over deliberately: the error-check macro
pattern (`cuda_common/error_check.cuh`, throw-not-abort), CUDA-event timing
(`timing.py` wraps `torch.cuda.Event` exactly the way KernelForge's
`GpuTimer` wraps `cudaEvent_t`), deterministic seeded RNG
(`reference/tensors.py`, same default seed value), a CPU/host-side
reference checked before any kernel is trusted, and the benchmark JSON
schema + methodology-doc convention.

## Test framework: pytest, not a ported KF_TEST

The spec's tech-stack table specifies "pytest + kernel-level correctness
harness" for FlashLite (as opposed to KernelForge's from-scratch, no-
GoogleTest `KF_TEST` macro framework, justified by KernelForge's own ADR
0009). The two are the same underlying philosophy -- avoid an unnecessarily
heavy test framework, since the actual test bodies here are simple
(build deterministic input -> compute variant -> compare against reference)
-- applied to the language each project actually uses to call its kernels.
FlashLite's V1 kernel is reachable ONLY through a Python tensor binding
(`flashlite._cuda_naive.attention_naive_forward`), unlike KernelForge's
kernels, which are also callable from a raw C++ host program; there is no
standalone C++ entry point here for a `KF_TEST`-style binary to exercise
that pytest cannot already exercise through the compiled extension, so no
separate C++ test binary is introduced. `tests/{math,correctness,edge_cases}/`
plays exactly the role KernelForge's `tests/` directory plays: one process,
self-discovering test functions (pytest's collection is pytest's own
version of "self-registering"), a `--filter`-equivalent (`pytest -k`), and
zero new external test-framework dependencies beyond pytest itself (already
in every Python project in this portfolio, e.g. `../01-forgelm`).
