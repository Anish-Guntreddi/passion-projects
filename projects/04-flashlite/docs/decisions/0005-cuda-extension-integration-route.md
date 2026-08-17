# ADR 0005: pybind11 + torch.utils.cpp_extension Integration Route (D6)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D6 -- default: "pybind11/torch cpp_extension; ADR."

## Context
FlashLite's CUDA kernels must be callable from Python `torch.Tensor`
objects (the reference harness, correctness comparator, and benchmark
driver are all Python/PyTorch, per the spec's tech-stack table). Two
routes were available: (a) `torch.utils.cpp_extension` (setuptools-based;
either JIT `load()` or a proper installable `CUDAExtension` in `setup.py`),
which links against pybind11 and the exact installed libtorch ABI
automatically, or (b) a hand-rolled CMake build using `find_package(Torch)`
and manually wiring up pybind11/`Python.h` include paths and libtorch
linkage -- closer to what the spec's generic repository-structure sketch's
top-level `CMakeLists.txt # CUDA extension` comment suggests literally, and
closer to KernelForge's own build system.

## Decision
Use **`torch.utils.cpp_extension.CUDAExtension`**, driven from `setup.py`
(package metadata lives in `pyproject.toml`; `ext_modules` needs `setup.py`
since torch's build helper is not yet expressible in `pyproject.toml`
alone). `src/flashlite/cuda_naive/bindings.cpp` defines a
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)` exposing
`attention_naive_forward(q, k, v, causal) -> Tensor`, built into
`flashlite._cuda_naive` and imported lazily by `flashlite.ops.attention`.
This **supersedes** the spec's generic repo-structure sketch's top-level
`CMakeLists.txt` for the actual build mechanism -- the directory-layout
conventions it calls out (`src/reference/`, `src/cuda/<variant>/`,
`src/bindings/`) are preserved (as `src/flashlite/reference/`,
`src/flashlite/cuda_naive/`, and `bindings.cpp` living alongside its kernel
rather than in a separate top-level `bindings/` directory, since each
variant's binding module is one small file tightly coupled to that
variant's kernel).

## Consequences
- `pip install -e .` (via `scripts/build_ext.sh`) is the single build step
  for the whole project -- no separate CMake configure/build/install cycle,
  no manual libtorch path wiring, and the extension automatically links
  against whichever installed torch build (`torch==2.6.0+cu124` in this
  repo's WSL2 venv, verified) provides.
- ABI compatibility between the compiled extension and the Python
  interpreter/torch build that imports it is handled by torch's build
  helper, not by this project -- eliminates an entire class of build bugs
  a hand-rolled CMake+libtorch setup would need to solve itself (matching
  `-D_GLIBCXX_USE_CXX11_ABI`, locating the right `Python.h`, etc.).
- Error-check discipline is still explicit and thrown, not silent:
  `error_check.cuh`'s `FL_CUDA_CHECK`/`FL_CUDA_CHECK_LAST_ERROR` macros
  (mirroring KernelForge's `error_check.cuh` almost verbatim) wrap every
  CUDA runtime call inside the kernel launcher; `bindings.cpp`'s
  `TORCH_CHECK` calls validate shape/dtype/layout before any kernel
  launches. Both paths raise Python exceptions pybind11 translates
  automatically (`std::runtime_error` -> `RuntimeError`, `c10::Error` ->
  `RuntimeError`), satisfying the spec's "fail clearly" requirement without
  reimplementing exception translation by hand.
- Target architecture: `TORCH_CUDA_ARCH_LIST=8.9` is set in `setup.py`
  (sm_89 / RTX 4090, matching KernelForge's ADR 0001 and this project's own
  ADR 0001), so a fresh build does not depend on torch's build-time GPU
  auto-detection succeeding in every environment.
