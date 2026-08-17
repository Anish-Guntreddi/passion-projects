# ADR 0004: CPU-only dev path vs. CUDA expectation (D3)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D3 (`01-forgelm-spec.md` §1.9) — *"CPU-only dev path vs
CUDA expectation → default: CPU path must work for tests; GPU for real
runs."*

## Decision

Every test in `tests/` must pass on a CPU-only machine. GPU-specific tests
(`tests/integration/test_smoke_gpu.py`, tagged `@pytest.mark.gpu`) are
written so they automatically skip via
`@pytest.mark.skipif(not torch.cuda.is_available(), ...)` rather than fail,
and none of the CPU-path tests (unit or integration) require a CUDA device.
Real training/benchmark runs (Phases 3+) are expected to use the GPU where
available, but no test in the suite is allowed to assume one exists.

Environment install: `torch` is installed from the **cu124** wheel index
(`pip install torch --index-url https://download.pytorch.org/whl/cu124`,
see `scripts/setup_env.sh`) rather than the CPU-only wheel index, because
the cu124 wheel is a superset — it runs correctly on a CPU-only machine
*and* uses CUDA automatically when a compatible GPU is present. This means
one dependency spec covers both the "CPU path must work" requirement and
the "GPU for real runs" requirement without maintaining two install paths.

## Consequences

- `forgelm.seed.set_seed` and `forgelm.cli.smoke.run_smoke` both branch on
  `torch.cuda.is_available()` rather than assuming a device; `SmokeConfig`
  accepts `device: "cpu" | "cuda"` explicitly rather than an "auto" mode,
  keeping determinism per-run explicit rather than environment-dependent.
- CI (`.github/workflows/ci.yml`) installs the **CPU-only** wheel index
  (`.../whl/cpu`) since GitHub-hosted runners have no GPU — CI is the
  concrete enforcement of "CPU path must work."
- This environment happens to have an RTX 4090 visible in WSL2, so GPU
  smoke tests do run here (not skipped) — see the final phase report for
  actual observed output. The CUDA *toolkit* (`nvcc`) is not required for
  any of this: precompiled PyTorch CUDA kernels run without it, and no code
  in Phases 0–1 compiles a custom CUDA kernel.
