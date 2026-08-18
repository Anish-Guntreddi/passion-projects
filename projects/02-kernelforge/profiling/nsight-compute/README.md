# Nsight Compute (`ncu`) — Why This Directory Has No `.ncu-rep` Files

**This is not an oversight or an unattempted TODO.** `ncu` was reinstalled/
retried at the start of Phase 6 (2026-08-17) specifically to check whether
the Phase 0-1 "not found" state (ADR 0006) still held, found to have
changed, and retried against this repo's own binaries. Full writeup:
`docs/decisions/0014-phase6-profiling-evidence-strategy.md`. Summary:

```
$ /usr/local/cuda/bin/ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Version 2024.3.2.0 (build 34861637) (public-release)
```

`ncu` **is present** on this machine (`cuda-nsight-compute-12-6` apt
package). Running it against a real kernel from this repo:

```
$ /usr/local/cuda/bin/ncu --set basic --launch-count 1 -o /tmp/ncu_test --force-overwrite \
    build/apps/bench_transpose --variant v2_tiled --n 512 --warmup 5 --reps 20
==PROF== Connected to process 400028 (.../build/apps/bench_transpose)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0. For instructions
on enabling permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

`ERR_NVGPUCTRPERM` is a **driver-level access-control restriction**
(`NVreg_RestrictProfilingToAdminUsers`, default-on since driver 418.43),
not a missing package or a wrong invocation. Under WSL2, GPU access is
virtualized through the **Windows host's** NVIDIA driver — the fix (an
NVIDIA Control Panel "Developer" setting, or the Windows registry key the
error message links to) requires **Windows-host administrator access**,
which is out of scope for a WSL2-guest coding task and not something
`sudo` inside this guest can reach (`sudo -n true` here also confirms no
passwordless sudo is available regardless). This is a genuine environment
constraint, reproduced exactly as shown above, not asserted from stale
information.

Nsight Systems' GPU-side kernel trace is affected by the same
restriction (see the ADR) — so this directory being empty is not worked
around by substituting `nsys` GPU-trace data either. What Phase 6 uses
instead, and why each is a legitimate (if narrower) substitute for what
`ncu` would show, is in ADR 0014:

- **Theoretical occupancy** (`profiling/occupancy/*.jsonl`,
  `src/common/occupancy.cuh`) in place of `ncu`'s "Launch Statistics" /
  "Occupancy" sections' theoretical-occupancy fields (their *achieved*-
  occupancy fields, which need hardware counters, are the one thing this
  substitute genuinely cannot provide).
- **PTX/SASS inspection** (`profiling/ptx-sass/`) in place of `ncu`'s
  disassembly view, for instruction-level questions (e.g. counting
  `BAR.SYNC`/`SHFL.DOWN`/`LDS` instructions).
- **Nsight Systems host-side CUDA API traces**
  (`profiling/nsight-systems/*.stats.txt`) for launch-count/host-overhead
  questions.
- **This repo's own committed CUDA-event kernel timings**
  (`benchmarks/raw/*.jsonl`) for kernel-duration questions — unaffected by
  this restriction, since they use `cudaEvent*` timestamps the
  application records itself, not a CUPTI trace.

If this environment's restriction is ever lifted (Windows-host admin
setting changed, or this repo built on native Linux with root access to
set the kernel module parameter), this directory is where `.ncu-rep`
captures for the same 3 case-study kernels would go — nothing else in
Phase 6's approach needs to change to add that evidence in.
