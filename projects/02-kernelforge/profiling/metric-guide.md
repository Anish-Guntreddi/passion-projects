# Profiling Metric Guide

**Status: not yet populated.** This file is part of the repo structure
from Phase 0 (per the spec's tech-stack plan) but its content is a Phase
6 ("Profiling & low-level analysis") deliverable — no profiler evidence is
claimed anywhere in Phases 0-1, per hard constraint 3 (never claim
speedup/evidence that isn't backed by committed artifacts).

## What's here today

- `nsight-systems/` — empty, reserved for `.nsys-rep` timeline captures.
- `nsight-compute/` — empty, reserved for `.ncu-rep` kernel-metric captures.

## Verified tool availability on the development machine (2026-08-17, WSL2)

```
$ which nsys && nsys --version
/usr/local/bin/nsys
NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0

$ which ncu
(not found)

$ /usr/local/cuda/bin/compute-sanitizer --version
NVIDIA (R) Compute Sanitizer
Version 2024.3.0.0 (build 34841621) (public-release)
```

**Nsight Compute (`ncu`) is not installed on this machine.** Nsight
Systems and Compute Sanitizer are present and confirmed working
(`scripts/run_sanitizer.sh` runs clean over this repo's test suite as of
Phase 1). See ADR 0006 for how this affects the Phase 6 plan: `ncu`
(or an equivalent kernel-metric source) needs to be installed before any
SASS/occupancy/bank-conflict metric can be *measured* rather than
predicted from source-level reasoning.

## Plan for Phase 6

Once `ncu` is available, this file will be filled in with, at minimum:
- Which metrics answer which kind of question (e.g.
  `gld_efficiency`/`gst_efficiency` or their Ada-generation equivalents
  for coalescing questions; `shared_ld_bank_conflict` for the
  known-but-unmeasured `transpose_tiled` bank conflict noted in
  `src/kernels/transpose/transpose_tiled.cuh`; occupancy/achieved-occupancy
  for launch-configuration questions).
- Exact `ncu` invocation used for each of the 3 profiler-backed case
  studies (FR6), with the resulting reports committed under
  `profiling/nsight-compute/`.
- Nsight Systems timeline captures for any multi-kernel/host-device
  overlap question, under `profiling/nsight-systems/`.

Nothing above is a claim of what a profiler *would* show — only a plan
for what will be measured, filled in only once actually measured.
