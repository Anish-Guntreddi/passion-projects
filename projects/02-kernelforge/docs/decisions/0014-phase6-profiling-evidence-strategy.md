# ADR 0014: Phase 6 Profiling Evidence Strategy

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
Phase 6 ("Profiling & low-level analysis") requires FR5/FR6: "≥3 kernels
with complete profiler-backed case studies... Nsight Systems for
timeline/system questions, Nsight Compute for kernel questions." ADR 0006
already flagged (Phase 0-1) that `ncu` was not found on `$PATH` on this
machine, and `docs/gpu-model.md`/README carried that as a known
limitation blocking Phase 6.

Revisiting this at the start of Phase 6 (2026-08-17) found a more precise
picture than "not installed": `ncu` **is** present on this machine, just
not on the login shell's default `$PATH` (`cuda-nsight-compute-12-6` is an
installed apt package):

```
$ dpkg -l | grep cuda-nsight-compute
ii  cuda-nsight-compute-12-6   12.6.3-1   amd64   NVIDIA Nsight Compute
$ /usr/local/cuda/bin/ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2024 NVIDIA Corporation
Version 2024.3.2.0 (build 34861637) (public-release)
```

But actually invoking it against a real kernel launch fails, not with a
"not found" error, but with a GPU-performance-counter **permission**
error:

```
$ /usr/local/cuda/bin/ncu --set basic --launch-count 1 \
    -o /tmp/ncu_test --force-overwrite \
    build/apps/bench_transpose --variant v2_tiled --n 512 --warmup 5 --reps 20
==PROF== Connected to process 400028 (.../build/apps/bench_transpose)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0. For instructions
on enabling permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

`ERR_NVGPUCTRPERM` is NVIDIA's documented error for the driver-level GPU
performance-counter access-control feature (`NVreg_RestrictProfiling
ToAdminUsers`), which restricts hardware performance-counter access to
admin/root users by default since driver 418.43. Under native Linux this
is fixed by a kernel-module parameter change + reboot (needs root).
**Under WSL2, GPU access is virtualized through the Windows host's
NVIDIA driver** (`dxcore`/`libcuda.so` talk to the host driver, not a
guest kernel module), so the equivalent fix is a **Windows-host**
administrator action (an NVIDIA Control Panel "Developer" setting or a
Windows registry key under `HKLM`), not anything reachable from inside
this WSL2 guest. `sudo -n true` inside WSL2 confirms no passwordless
sudo is available here either (`sudo: a password is required`), and even
root inside the WSL2 guest would not help, since the restriction is
enforced by the host driver the guest has no access to. **This is
therefore a genuine, environment-level block, not a missing-package
problem** — reproducible exactly as shown above, not asserted from a
prior "not found" state.

Everything else Phase 6 needs, by contrast, verified working:

```
$ nsys --version
NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0
$ /usr/local/cuda/bin/cuobjdump --version | head -2
Cuobjdump: NVIDIA (R) Cuda Object Dump Utility, Version 12.6.85
$ /usr/local/cuda/bin/nvdisasm --version | head -2
nvdisasm: NVIDIA (R) CUDA disassembler, Release 12.6, V12.6.85
```

`nsys` (Nsight Systems) uses CUPTI's **activity/tracing** API, which is a
different CUPTI subsystem from the **counter**-based Profiling API `ncu`
uses. Confirmed working for host-side data (see Decision §2 below) — but
actually capturing a real kernel and inspecting the result shows this
subsystem is **only partially usable** under this WSL2 guest, not fully
independent of the same restriction:

```
$ nsys profile --trace=cuda -o /tmp/t build/apps/bench_transpose --variant v1_naive --n 512 --warmup 2 --reps 20
... (capture succeeds, .nsys-rep generated, no error/warning printed)
$ nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum /tmp/t.nsys-rep
SKIPPED: ... does not contain CUDA kernel data.
SKIPPED: ... does not contain GPU memory data.
 ** CUDA API Summary (cuda_api_sum):
 [... a normal, populated host-side table of cudaMalloc/cudaMemcpy/
      cudaLaunchKernel/cudaEventSynchronize call counts and durations ...]
$ nsys status -e | grep -A2 'GPU Metrics'
GPU Metrics: None of the installed GPUs are supported:
	NVIDIA GeForce RTX 4090 ... - Insufficient privilege, see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

So: `nsys` captures **host-side CUDA Runtime API call data** (which
function was called, how many times, how long the call itself took, e.g.
`cudaLaunchKernel`/`cudaMemcpy`/`cudaEventSynchronize` counts and
durations) successfully and without error — this is real, useful evidence
for launch-count/launch-overhead/host-side-timing questions. But
**device-side GPU kernel-execution and GPU memory-operation trace data is
silently absent** from every capture (no error at capture time; `nsys
stats` reports `SKIPPED: ... does not contain CUDA kernel data` at
analysis time) — the same `ERR_NVGPUCTRPERM` restriction that blocks
`ncu` blocks this half of `nsys` too. This was cross-checked against
FlashLite's already-committed `.nsys-rep` files
(`projects/04-flashlite/profiling/nsight-systems/*.nsys-rep`) — the same
`SKIPPED: ... does not contain CUDA kernel data` result — confirming this
is an environment-wide restriction, not specific to this project's
invocation.

`cuobjdump`/`nvdisasm` need no GPU access at all: they are static
disassemblers reading a compiled `.cubin`/executable on disk.

## Decision
Phase 6's 3 profiler-backed case studies substitute the following
evidence sources for `ncu`'s hardware-counter metrics, chosen so every
number is still measured (never asserted) and each source answers a
specific class of question `ncu` would otherwise answer:

1. **Theoretical occupancy, from the CUDA Runtime's own occupancy API**
   (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
   `cudaOccupancyMaxPotentialBlockSize`, `cudaFuncGetAttributes` — see
   `src/common/occupancy.cuh`, one `*_query_occupancy()` wrapper per
   case-study kernel, and the `occupancy_report` CLI,
   `apps/occupancy_report_main.cpp`). This is the CUDA driver computing
   occupancy from the actual compiled kernel's real register count and
   shared-memory footprint (not a guess) plus this device's real limits
   — the same *theoretical*-occupancy numbers `ncu`'s "Launch Statistics"
   section would report, just without the *achieved* (runtime,
   counter-based) occupancy number `ncu` additionally provides and this
   environment cannot. Committed as raw JSON records under
   `profiling/occupancy/*.jsonl` (schema:
   `profiling/schema/occupancy_report.schema.json`), the same
   "raw-file-is-the-source-of-truth" pattern as `benchmarks/raw/` (ADR
   0010).
2. **Nsight Systems host-side CUDA API traces** for any question about
   launch *count* or host-observed call *overhead/latency* (NOT
   device-side kernel-execution duration -- that half of `nsys`'s data is
   blocked too, see Context above) — captured as `.nsys-rep` files (+ a
   `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum`
   text summary alongside each, which documents the GPU-side `SKIPPED`
   result explicitly rather than omitting those report sections) under
   `profiling/nsight-systems/`. Device-side kernel *duration* claims in
   this repo's case studies instead cite the already-committed
   `benchmarks/raw/*.jsonl` CUDA-event timings (source #4 below) — the
   evidence this repo has used for kernel timing since Phase 0, unaffected
   by this restriction since it uses `cudaEvent*` timestamps recorded by
   the application itself, not a CUPTI trace.
3. **PTX and SASS inspection** (`cuobjdump --dump-ptx` /
   `--dump-sass` on the built benchmark binaries) for any question about
   what instructions a kernel actually compiles to — e.g. counting
   `BAR.SYNC` vs `SHFL.DOWN` instructions, or confirming a shared-memory
   load's addressing pattern — per ADR 0006's existing PTX-default/
   SASS-where-it-explains-a-result policy. Committed under
   `profiling/ptx-sass/`.
4. **`benchmarks/raw/*.jsonl` wall-clock evidence already committed in
   Phases 1-5**, recomputed with full precision for any case study that
   cites it (never re-typed from a rounded README figure) — the "measured
   outcome" step of FR5's loop for case studies that revisit an
   already-shipped ladder rung rather than introducing a new one.
5. **`ncu` is retried, and its exact failure documented, in
   `profiling/nsight-compute/README.md`, rather than that directory being
   silently left empty with no explanation.** This is not presented as
   equivalent to actual hardware-counter evidence anywhere — every
   occupancy number in this repo is explicitly labeled "theoretical" (the
   `OccupancyReport` struct's own field comments, every case study's
   text), and no case study claims an *achieved* occupancy, warp-execution
   efficiency, or memory-throughput-efficiency percentage that only a
   real hardware counter could produce.

## Consequences
- Phase 6's case studies are honest about what evidence backs each claim:
  a mechanism explained by PTX/SASS instruction counts is a **static**
  fact about the compiled kernel (true on every run, not a per-run
  measurement); a launch-count/host-overhead claim backed by `nsys`'s
  `cuda_api_sum` is a real **host-side timeline** measurement; a kernel
  *duration* claim is the application's own committed CUDA-event timing
  (`benchmarks/raw/*.jsonl`), never a `nsys` GPU-side trace (which this
  environment cannot produce); a resource-occupancy claim backed by
  `occupancy_report` is a real **theoretical ceiling** computed by the
  CUDA driver from the actual compiled kernel, not an *achieved* number.
  None of these is silently presented as "what `ncu` (or a fully working
  `nsys` GPU trace) would show."
- If `ncu` becomes usable in a future environment (e.g. this repo built
  and profiled on native Linux, or a Windows-host admin enables the
  registry setting `ERR_NVGPUCTRPERM`'s own error message links to),
  `profiling/nsight-compute/` and `profiling/metric-guide.md` are exactly
  where that evidence would be added — nothing in this ADR's approach
  needs to be reverted, only extended.
- This is the same "disclose the limitation, satisfy the phase's intent
  with what's actually available" pattern already used for locked clocks
  (ADR 0005) — the environment constraint is stated once, precisely, with
  a reproduction command, and every downstream document links back here
  instead of re-litigating it.
