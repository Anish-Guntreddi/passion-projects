# ADR 0005: Benchmark Noise Control (D5)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Spec open decision D5 — default: locked clocks
  documented if possible; ≥20 reps, median + IQR.

## Context
The spec's recommended default is to lock GPU clocks where possible and
otherwise document the caveat, and to always report ≥20 repetitions with
median + IQR rather than a single timing (FR4, hard constraint 3).

Verified on this machine (2026-08-17, inside WSL2 Ubuntu):

```
$ nvidia-smi -lgc 210,2520
The current user does not have permission to change clocks for GPU 00000000:01:00.0.
Terminating early due to previous errors.
```

Locking application/GPU clocks requires elevated privileges that are not
available to the WSL2 GPU-passthrough user on this machine (this is a
known WSL2 limitation — clock/power management ioctls are largely
reserved for the Windows host driver, not the Linux guest). `nvidia-smi
-q -d CLOCK` does work read-only and is used to *record* observed clocks
alongside every run instead.

## Decision
1. **Clocks are not locked** on this machine/environment. Every benchmark
   run's `EnvironmentInfo.locked_clocks` field is `false`, and
   `EnvironmentInfo.clock_lock_note` records the exact `nvidia-smi -lgc`
   permission-denied output above, so no report can imply a controlled
   clock state that was never achieved. This is a real, disclosed
   limitation, not a fabricated one (hard constraint 3).
2. **Warmup:** every benchmark runs a fixed number of untimed warmup
   iterations (default 10) before any timed repetition, to let clocks
   ramp to their steady-state boost state and let caches/allocators
   settle.
3. **Repetitions:** every benchmark runs **≥ 20 timed repetitions**
   (default 30) per (kernel, variant, size) triple. Each repetition is
   timed individually with CUDA events (`kernelforge::GpuTimer`), and
   **all raw per-repetition timings are recorded** in
   `BenchResult::raw_timings_ms`, not just a mean — matching FR4's "median
   + distribution (never a single timing)".
4. **Reported statistics:** median, IQR (25th/75th percentile), min, max,
   mean, and stddev are all computed and stored
   (`kernelforge::compute_stats`). Headline numbers in any writeup use the
   **median**, since it is robust to the occasional long-tail stall
   (driver/OS scheduling jitter) that is common in a WSL2 guest.
5. **Between-run noise is disclosed, not hidden:** because clocks are
   unlocked, IQR width itself is treated as a data point — a wide IQR is
   reported as-is rather than discarded, since throwing away high
   variance samples would understate real-world noise on this platform.

## Consequences
- Benchmark numbers from this repo reflect a **desktop/WSL2 environment
  with dynamic boost clocks**, not a locked-clock datacenter measurement.
  Every committed result's `env.locked_clocks=false` makes this
  unambiguous to a reader, and `benchmarks/methodology.md` explains it
  once so individual reports do not need to re-justify it.
- Relative comparisons (variant A vs variant B, same size, same run
  session) remain meaningful because both variants experience the same
  clock/thermal environment; absolute GB/s or GFLOP/s numbers should be
  read as "typical for this desktop under WSL2", not as a spec-sheet
  peak.
