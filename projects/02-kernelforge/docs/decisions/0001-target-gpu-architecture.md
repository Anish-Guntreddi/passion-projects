# ADR 0001: Target GPU Architecture (D1)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D1 — "Target GPU architecture available to developer → human decision; record in device-info output."

## Context
KernelForge needs one concrete, fixed target architecture to compile
`CMAKE_CUDA_ARCHITECTURES` against and to reason about (occupancy, shared
memory size, warp size, memory bus width) when writing hypotheses. The
development machine is a Windows 11 host running WSL2 Ubuntu with a single
discrete NVIDIA GPU passed through via WSL's GPU paravirtualization.

Detected hardware (captured by `apps/device_info_main`, see
`docs/gpu-model.md` for the full dump):

- GPU: NVIDIA GeForce RTX 4090
- Compute capability: 8.9 (Ada Lovelace)
- VRAM: 24564 MiB
- Driver: 591.86 (host Windows driver, exposed into WSL2)
- CUDA toolkit (nvcc): 12.6, release V12.6.85

## Decision
Target architecture is **Ada Lovelace, `sm_89`**, matching the RTX 4090
that is the only GPU available to the developer. `CMAKE_CUDA_ARCHITECTURES`
defaults to `89` in the top-level `CMakeLists.txt` and can be overridden
with `-DKF_CUDA_ARCH=...` for portability, but no other architecture is
tested against in this repo.

## Consequences
- All kernels are written and tuned against Ada's parameters: 128 KB max
  shared memory/SM (99 KB opt-in per block), 32-thread warps, 24 GB GDDR6X
  at a 384-bit bus.
- Architecture-specific choices (e.g. shared memory carveout, warp-level
  primitive availability) are guarded by capability checks
  (`kernelforge::require_capability_or_throw`) rather than assumed, so the
  failure mode on a different GPU is a loud, explicit error instead of
  silent miscompilation or wrong numbers (hard constraint 8).
- Benchmark results are only valid for sm_89 and are labeled with the GPU
  name/compute-capability in every committed result (FR4); no results are
  extrapolated to other architectures.
