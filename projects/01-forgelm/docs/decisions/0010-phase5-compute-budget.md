# ADR 0010: Phase 5 scaling-experiment compute budget (D8)

**Status:** Accepted (human decision recorded per spec §1.9)
**Date:** 2026-08-17
**Open decision:** D8 (`01-forgelm-spec.md` §1.9) — *"Compute budget for
the three scaling runs → human decision."*

## Decision

Phase 5's scaling experiment stays at **toy-to-small** model scale and is
budgeted to **under one hour of GPU wall-clock time in total**, including
every sweep (model-size scaling, context-length sweep, batch-size sweep,
and the optional `torch.compile` A/B) — not one hour per run. The measured
wall time actually spent is recorded in
`benchmarks/results/phase5_manifest.json` and quoted in
`benchmarks/methodology.md`, so this budget is verifiable after the fact,
not just declared up front.

## The experiment matrix

Three model sizes, everything else (dataset, context length for the
scaling+torch.compile runs, optimizer, precision) held fixed so parameter
count is the only thing varying across them:

| Config | d_model | n_layers | n_heads | d_ff | Params (tied embedding) |
|---|---|---|---|---|---|
| `small` | 64 | 2 | 4 | 256 | ≈164K |
| `medium` | 128 | 4 | 4 | 512 | ≈1.12M |
| `larger` | 256 | 6 | 8 | 1024 | ≈6.43M |

(Exact counts are recorded per-run from `forgelm.model.count_parameters`,
not retyped by hand — the table above is the analytic estimate used to
plan the sweep before running it.)

Plus two single-model sensitivity sweeps (spec Part 3 benchmark plan) and
one optional A/B, all using the `medium` config as the fixed point:

- **Context-length sweep**: `context_length ∈ {32, 64, 128, 256}`, batch
  size fixed.
- **Batch-size sweep**: `micro_batch_size ∈ {4, 8, 16, 32, 64}`, context
  length fixed at 128.
- **Optional `torch.compile` A/B**: the `small` config, compiled vs.
  uncompiled, same step count.

Dataset: the already-committed `examples/alice_in_wonderland.txt` corpus
built with the existing `configs/tokenizer_train.yaml` /
`configs/dataset_build.yaml` artifacts (vocab_size 512, context_length
128, ~84.6K tokens total) — reused rather than sourcing a new corpus, per
D3 ("prefer deterministic local development") and to avoid Phase 5 scope
creep into dataset acquisition. `examples/README.md`'s note that Phase 5
"might need more tokens" turned out unnecessary at this compute budget: a
few hundred steps per config against ~76K training tokens is enough to
produce a real, non-degenerate loss-vs-tokens curve for a toy-scale
model, without needing a larger download.

## Why this budget, not a larger one

- **Portfolio purpose, not SOTA research.** The deliverable (spec §1.6)
  is loss curves, throughput numbers, and memory-vs-config plots across
  *qualitatively distinct* model sizes — the shape of the scaling
  relationship matters, not squeezing out the last percent of a
  production-grade training run. A ~164K-to-6.4M parameter spread on a
  ~85K-token corpus is enough to show that shape (params vs. tokens/sec,
  params vs. memory, tokens trained vs. val loss) without training
  anything resembling a production model.
- **The RTX 4090 makes this budget generous, not tight.** Phase 0–3's
  `configs/train_toy.yaml` (d_model=64, 50 steps) already trains in ~40s
  on this GPU. Even the `larger` config here (6.4M params, ~5.7x the toy
  config's `larger`-relative size) at a few hundred steps is expected to
  finish in well under a minute — the ~1 hour ceiling is a safety margin
  against the *sum* of every sweep point, not a target to spend.
- **Consistent with the project's existing "toy-to-small" posture.**
  Phases 0–3 already established that CPU-testable correctness and a
  GPU-accelerated but small real run are the project's operating mode
  (ADR 0004). Phase 5 does not change that posture — it just runs it at
  three sizes instead of one, plus two sensitivity sweeps.
- **Mitigates the spec's own risk register** (§1.10: "scaling runs
  exceeding compute budget") by fixing the budget *before* running
  anything, with the GPU-benchmark-lock protocol (repo-wide, orchestrator
  rule) ensuring the measured window is held only while actually
  measuring and released immediately after.

## Consequences

- Every `BenchmarkResult` (`forgelm.benchmarks.harness.run_benchmark`)
  embeds hardware/software/config/seed, so a reviewer can verify the
  reported numbers came from the described run without re-running it
  (though re-running is also fully supported: `scripts/run_phase5_benchmarks.py`
  is the single source of truth for the whole sweep, taking configs as
  plain Python objects rather than ten hand-maintained YAML files, which
  keeps the experiment matrix in one reviewable place).
- `benchmarks/methodology.md` records the actual measured total wall-clock
  time spent under the GPU lock, confirming (not just claiming) the
  budget was respected. Measured result: **all 14 runs (3 scaling + 4
  context-length + 5 batch-size + 2 compile A/B) completed in 92.3 s
  (1.54 min) total** — the "well under a minute per config" prediction
  above held, and the realized total is ~2.6% of the 1-hour ceiling.
- If a future phase wants a larger scaling study (more configs, more
  steps, a bigger corpus), that is an explicit new decision — this ADR's
  budget is for Phase 5 of *this* project's MVP scope only, not a
  permanent ceiling on the repo.
