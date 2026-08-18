# ADR 0006: SASS Analysis Depth (D6)

- **Status:** Accepted; exercised in Phase 6 (2026-08-17) — see
  `profiling/case-studies/01-transpose-bank-conflict.md` (SASS used to
  confirm the shared-memory bank-conflict addressing pattern) and
  `profiling/case-studies/02-reduction-warp-shuffle-barriers.md` (SASS
  used to confirm barrier/shuffle instruction counts); Case Study 3
  (`03-gemm-register-tiling-occupancy.md`) stays at the PTX/occupancy
  level per this ADR's default, since occupancy + register-count evidence
  fully explains that result without needing SASS.
- **Date:** 2026-08-17 (decision); 2026-08-17 (exercised, Phase 6)
- **Decision driver:** Spec open decision D6 — default: PTX for 3 case
  studies; SASS only where it explains a result.

## Context
Phase 6 ("Profiling & low-level analysis") is where PTX/SASS inspection
happens (FR6: ≥3 profiler-backed case studies with PTX notes, SASS
optional). Recorded now so the convention is fixed in advance.

## Decision
- The **default evidence level is PTX** (`nvcc -ptx`, or
  `cuobjdump -ptx` on the built binary), examined for the 3 kernels
  chosen as the profiler-backed case studies.
- **SASS** (`cuobjdump -sass` / Nsight Compute's SASS view) is pulled in
  **only when PTX is insufficient to explain an observed result** —
  e.g. when a register-allocation or instruction-scheduling detail that
  only exists post-ptxas is the actual explanation (bank-conflict
  padding effects, register pressure/occupancy cliffs).
- Every PTX/SASS excerpt included in a writeup is generated from the
  actual committed source at the commit it documents (regenerable via a
  script in `scripts/`), never hand-transcribed or reconstructed from
  memory.

## Consequences
- No action required in Phases 0–1 (no profiler-backed case study is
  claimed yet). Nsight Compute (`ncu`) was **not found** on this WSL2
  install (`which ncu` → not found) while Nsight Systems (`nsys`,
  v2024.5.1.113) and `compute-sanitizer` (v2024.3.0.0) are present —
  this is noted here so Phase 6 planning knows `ncu` needs to be
  installed (or its absence worked around) before SASS/kernel-metric
  work can start; it is not required for Phases 0–1.
- **Phase 6 update (2026-08-17):** `ncu` turned out to be present after
  all (a `$PATH` issue, not a missing package) but is blocked by a
  genuine driver-level permission restriction (`ERR_NVGPUCTRPERM`) on
  actual use — see `docs/decisions/0014-phase6-profiling-evidence-
  strategy.md` for the full investigation. `cuobjdump`/`nvdisasm` (used
  for this ADR's PTX/SASS extraction) are both present and working,
  needing no GPU access at all. This ADR's PTX-default/SASS-where-it-
  explains-a-result policy is applied unchanged by ADR 0014's evidence
  strategy — only the source of the "kernel metric" evidence (theoretical
  occupancy + PTX/SASS + this repo's own committed timings, rather than
  `ncu`) is different from what was anticipated here.
