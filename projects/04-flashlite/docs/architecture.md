# Architecture (Phases 0-1)

## Data flow

```
tests / scripts/run_benchmarks.py
        |
        v
flashlite.reference.tensors.make_qkv   (deterministic seeded Q, K, V)
        |
        +---------------------------------------------+
        |                                              |
        v                                              v
flashlite.reference.attention          flashlite.ops.attention(variant="naive")
  (V0: torch.matmul + torch.softmax,        |
   CPU or CUDA, the trusted ground           v
   truth every variant below is       flashlite._cuda_naive.attention_naive_forward
   checked against)                     (V1: pybind11 binding, src/flashlite/cuda_naive/)
        |                                    |
        |    TORCH_CHECK shape/dtype/layout validation (bindings.cpp)
        |                                    |
        |                          launch_attention_naive (attention_naive.cu)
        |                            1. compute_scores_kernel   -> scores[B,H,S,S]
        |                            2. softmax_rows_kernel     -> scores in place
        |                            3. weighted_sum_kernel     -> out[B,H,S,D]
        |                                    |
        +-----------------+-------------------+
                          |
                          v
              flashlite.compare.allclose_compare
           (elementwise |actual-expected| <= atol + rtol*|expected|;
            max/mean abs error, max rel error, first-mismatch detail)
                          |
                          v
              flashlite.timing.GpuTimer (cudaEvent-based, kernel-only)
                          |
                          v
              flashlite.bench_schema.BenchResult + flashlite.env_capture
                          |
                          v
            benchmarks/raw/attention.jsonl  (one JSON object per line,
            validated against benchmarks/schema/bench_result.schema.json
            by scripts/validate_results.py)
```

This mirrors the spec's Part 2 architecture line ("Python benchmark/test
harness -> reference attention -> custom extension/kernel dispatcher ->
correctness comparator -> timing/memory profiler -> raw results/plots")
and KernelForge's separation of concerns (error-checked kernel launchers,
a CPU/host-side reference, a tolerance-based comparator, a benchmark
schema captured with environment metadata) -- ported from KernelForge's
CMake+CUDA C++ harness to a PyTorch/pybind11 harness because D6's resolved
default is the `torch.utils.cpp_extension` integration route (ADR 0005),
not a from-scratch CMake+libtorch build.

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
  ops.py                     variant dispatcher ("reference" | "naive" | ...)
  cuda_naive/                V1: naive CUDA kernel + pybind11 bindings
    error_check.cuh            CUDA error-check macros (throw, don't abort)
    attention_naive.cuh        kernel/launcher declarations + shape contract
    attention_naive.cu         3-kernel naive pipeline (score/softmax/weighted-sum)
    bindings.cpp                TORCH_CHECK validation + pybind11 module
tests/
  math/          Phase 0: reference correctness, RNG determinism, bench schema
  correctness/   Phase 1: V1 vs V0 across the shape/dtype/causal matrix
  edge_cases/    unsupported-shape rejection paths (both V0 and V1)
benchmarks/
  schema/         bench_result.schema.json (JSON Schema, mirrors bench_schema.py)
  configs/        sweep specs (benchmarks/configs/attention.json)
  raw/            committed .jsonl results (one file per kernel family)
  methodology.md  how every number under raw/ was produced
docs/
  attention-math.md    V0/V1 math derivation + materialized-form cost analysis
  architecture.md      this file
  decisions/           ADRs for the spec's open decisions D1-D4, D6, D7
                        (D5 is deferred to Phase 6, per the spec)
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
pattern (`error_check.cuh`, throw-not-abort), CUDA-event timing
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
