# Optimization Method

Every performance change in this repo — starting with the naive-vs-tiled
transpose comparison in Phase 1 — follows the same six-step loop (FR5).
This document defines the loop once so individual kernel writeups can
stay short and just fill in the steps.

## The loop

1. **Hypothesis.** Written *before* the code that tests it. States: which
   single variable is changing, what GPU-architecture mechanism predicts
   a change, and what direction/rough magnitude is expected and why. A
   hypothesis that just says "this should be faster" is not acceptable —
   it must name the mechanism (e.g. "global memory coalescing: a warp's
   32 threads issuing a strided store with stride = row-count each land
   in a different 32-/128-byte transaction, versus one transaction for a
   unit-stride store").
2. **Evidence before.** Run the *current* (pre-change) variant through
   the standard benchmark harness (warmup + ≥20 reps, median + IQR, ADR
   0005) and, where the hypothesis is about memory traffic, note the
   theoretical peak (device memory bandwidth from `device-info`) so the
   "before" number can be expressed as a fraction of peak.
3. **Change.** Exactly one variable changes between the "before" variant
   and the "after" variant (hard constraint 4). Both variants are kept
   side by side in source (hard constraint 5) — the "after" variant is a
   new file/function, never an in-place edit that deletes the "before"
   code.
4. **Evidence after.** Same benchmark harness, same input sizes, same
   warmup/rep counts, same machine session (so clock/thermal state is as
   comparable as unlocked clocks allow — ADR 0005).
5. **Measured outcome.** Report the before/after median (and IQR) and the
   derived metric relevant to the hypothesis (effective GB/s for memory-
   bound kernels, GFLOP/s for compute-bound ones). No speedup number is
   ever written from anything other than two committed `BenchResult`
   records (hard constraint 3).
6. **Interpretation.** Explain the result using the architecture
   mechanism named in step 1. If the result contradicts the hypothesis,
   that is reported too — a falsified hypothesis with an honest
   explanation is more valuable, and more interview-defensible, than a
   cherry-picked confirming number.

## Where this is first applied

The Phase 1 memory-access lab is the first full instance of this loop:
`docs/../benchmarks` (methodology) defines *how* every measurement is
taken, and the transpose naive→tiled comparison (see
`benchmarks/raw/transpose.jsonl` once generated, and the postmortem in
`benchmarks/methodology.md` §"Phase 1 results") is the first *filled-in*
instance of the loop above, plus the dedicated stride microbenchmark
which isolates the same mechanism (coalescing) in the smallest possible
kernel, independent of any transpose-specific complexity.

## Non-goals of this document

This file explains the *process*. It intentionally contains no
benchmark numbers itself — those live only in `benchmarks/raw/*.jsonl`
(machine-readable, regenerable) and are summarized in
`benchmarks/methodology.md` and per-kernel READMEs, never duplicated by
hand here where they could drift out of sync with the committed source
of truth.
