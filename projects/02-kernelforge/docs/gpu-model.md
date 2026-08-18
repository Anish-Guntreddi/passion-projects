# GPU Model — NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89)

This is the one and only GPU this repo targets (ADR 0001). Every number
below is the literal output of `scripts/run_device_info.sh` on the
development machine (WSL2 Ubuntu, 2026-08-17) — not a spec-sheet value
copied from NVIDIA's marketing page.

```
$ scripts/run_device_info.sh
KernelForge device-info
========================
Device index         : 0
Name                 : NVIDIA GeForce RTX 4090
Compute capability   : 8.9
Total global memory  : 23.9878 GiB
SM count             : 128
Warp size            : 32
Max threads/block    : 1024
Max threads/SM       : 1536
Shared mem/block     : 49152 bytes
Shared mem/SM        : 102400 bytes
Registers/block      : 65536
Core clock           : 2520 MHz
Memory clock         : 10501 MHz
Memory bus width     : 384 bits
L2 cache size        : 73728 KiB
ECC enabled          : no
Concurrent kernels   : yes
Async engine count   : 1
Unified addressing   : yes
CUDA driver version  : 13.1
CUDA runtime version : 12.6
Theoretical peak BW  : 1008.1 GB/s

Capability check: OK (>= 8.9 required, see ADR 0002)
```

`--json` prints the same fields as a single machine-readable object; this
is what every `EnvironmentInfo` embedded in a `BenchResult` is derived
from (`src/common/bench_result.cpp::EnvironmentInfo::capture`).

## What these numbers mean for the kernels in this repo

- **128 SMs, 1536 threads/SM max → 196,608 threads resident at full
  occupancy.** Every kernel here launches far more threads than that for
  any interesting size, so grid-level parallelism is never the
  bottleneck in Phases 0-1; memory access pattern is.
- **72 MiB L2 cache.** This is large enough to fully cache the vector
  kernels' working set up to roughly n ≈ 2^21-2^22 elements (3 buffers x
  4 bytes x n), which is exactly why the `n=1,048,576` `vector_add`/`saxpy`
  benchmark point measures *above* the 1008.1 GB/s HBM peak
  (`benchmarks/methodology.md` §8) — it's measuring L2 bandwidth, not HBM
  bandwidth. Any benchmark comparison in this repo that wants to isolate
  HBM behavior uses sizes large enough to exceed L2 (the transpose sweep's
  8192x8192 point is 256 MiB per buffer; the stride sweep's buffers are
  similarly sized at the larger strides).
- **1008.1 GB/s theoretical peak bandwidth** (`2 x 10501 MHz x 384 bits /
  8 / 1e9`, the standard GDDR6X double-data-rate formula) is the
  denominator every "% of peak" figure in this repo's benchmark writeups
  is computed against.
- **49,152 bytes (48 KiB) default shared memory per block.** The
  transpose tile (`32 x 32 x 4 bytes = 4096 bytes`) uses well under 10% of
  this, so shared-memory capacity is not a limiting factor for
  `transpose_tiled` at Phase 1 — occupancy there is governed by the
  1024-thread block size (one full block per tile), not by shared memory.
- **Warp size 32.** Every coalescing argument in this repo (naive vs.
  tiled transpose, the stride microbenchmark) is stated in terms of what
  a single 32-thread warp's memory requests look like, because that's the
  real unit the memory subsystem coalesces at.

## Toolchain captured alongside every benchmark result

| Component | Version (this repo's build) |
|---|---|
| Host compiler | GNU (g++) 13.3.0 |
| CUDA compiler | NVIDIA nvcc 12.6.85 (toolkit 12.6) |
| CUDA driver (host) | 13.1 (Windows driver 591.86, exposed into WSL2) |
| `CMAKE_CUDA_ARCHITECTURES` | 89 (sm_89 only, ADR 0001/0002) |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| OS | Ubuntu (WSL2 guest) on Windows 11 Pro host |

Every `BenchResult.env` records this same information at the moment each
benchmark ran (not just once, here) — this table is a human-readable
summary of what is otherwise machine-verifiable per-result.

## Known WSL2 limitations (documented, not worked around silently)

- **GPU clocks cannot be locked** from this WSL2 guest
  (`nvidia-smi -lgc ...` → permission denied). See ADR 0005. Every
  benchmark result records `locked_clocks: false` plus an observed
  (unlocked) clock snapshot.
- **Nsight Compute (`ncu`) is present but blocked by `ERR_NVGPUCTRPERM`**
  (a GPU-performance-counter permission restriction enforced by the
  Windows host driver under WSL2, not fixable from inside this guest) —
  confirmed at Phase 6 (2026-08-17) with a direct reproduction; the
  earlier "not installed" note above described Phase 0-1's less precise
  `which ncu` check, before the actual package location/behavior was
  investigated. **Nsight Systems' device-side GPU kernel trace is
  affected by the same restriction** (its host-side CUDA API trace is
  unaffected and works). See `docs/decisions/
  0014-phase6-profiling-evidence-strategy.md` for the full investigation
  and what Phase 6 uses instead (theoretical occupancy via the CUDA
  Runtime API, PTX/SASS static disassembly, `nsys` host-side traces, and
  this repo's own committed CUDA-event kernel timings).
