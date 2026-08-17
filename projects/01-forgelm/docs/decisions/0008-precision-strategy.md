# ADR 0008: Mixed-precision strategy (D6)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D6 (`01-forgelm-spec.md` §1.9) — *"Precision strategy →
default: bf16 autocast if GPU supports it, else fp32."*

## Decision

`forgelm.training.precision.resolve_precision(device, requested)`
implements exactly the spec's default as its `"auto"` mode (the default
`TrainingConfig.precision` value):

- If `device` is CUDA and `torch.cuda.is_bf16_supported()` is `True`:
  wrap the forward pass in `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)`.
- Otherwise (CPU, or a CUDA device without bf16 support): plain fp32, no
  autocast.

`"fp32"` (always plain fp32) and `"bf16"` (force bf16 autocast, error if
not on CUDA) are also accepted as explicit overrides of `"auto"`.

Model parameters themselves are **not** cast to bf16/fp16 by the
precision policy — they stay in whatever dtype `ModelConfig.dtype`
constructed them with (default `float32`). Autocast casts *activations*
to the lower-precision dtype for the ops it wraps, while gradients and
the optimizer's own state (Adam's first/second moment buffers) remain
fp32 — this is the standard "fp32 master weights, autocast for compute"
pattern.

## Rationale

- **bf16 over fp16**: bf16 has the same 8-bit exponent range as fp32
  (only reduced mantissa precision), so it does not need loss-scaling
  (`torch.cuda.amp.GradScaler`) to avoid gradient underflow the way fp16
  does. Skipping loss scaling keeps `Trainer.train_step` simpler — one
  fewer moving part to get right and explain — which matches this
  project's priority on correctness and interview-explainability over
  squeezing out maximum throughput. fp16 is not implemented as an option
  at all in this MVP for that reason.
- **No CPU autocast policy**: torch does support CPU autocast in newer
  versions, but this project does not exercise it — D3's "CPU path must
  work for tests" is satisfied by plain fp32 on CPU, which is also what
  every CI run (no GPU) actually exercises.
- **"auto" as the config default** rather than requiring every config to
  pick a precision explicitly: this mirrors D6's own phrasing ("bf16 ...
  if GPU supports it, else fp32") as a single policy, so a config file
  written once behaves correctly whether it later runs on this project's
  CPU-only CI runner or on the RTX 4090 development machine, without
  needing two config variants.

## Consequences

- `Trainer.__init__` resolves the policy once from `(self.device,
  training_config.precision)` and reuses it for every `train_step` /
  `evaluate` call via `Trainer._autocast()`.
- `tests/unit/test_training_precision.py` exercises all three branches on
  CPU only (using `monkeypatch` to simulate a CUDA device with/without
  bf16 support for the branches that would otherwise require real GPU
  hardware), so the *policy logic* is fully covered without a GPU in CI;
  `tests/integration/test_training_gpu.py` (marked `@pytest.mark.gpu`,
  auto-skipped without a CUDA device) separately verifies the resolved
  policy against this machine's real RTX 4090.
- No numeric test in this project treats bf16-autocast results as exactly
  equal to fp32 results — only as "trains and produces finite, reasonable
  losses" — per the project-wide "never exact float equality" rule and
  because autocast intentionally trades precision for throughput.
