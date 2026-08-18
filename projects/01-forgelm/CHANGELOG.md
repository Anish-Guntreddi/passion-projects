# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/) — `0.y.z` while
this project is pre-1.0 MVP scope (spec §1.4's FR1–FR12), per the
portfolio-wide "never claim more stability than is proven" posture.

All dates below are the date each phase's changes were committed
(`git log --date=short`), not a claimed multi-day project timeline — this
project's phases 0–6 were built and reviewed against
`01-forgelm-spec.md`'s roadmap in sequence within the same session.

## [0.1.0] — 2026-08-17

The full MVP: all six roadmap phases (`01-forgelm-spec.md` Part 3)
implemented, tested, and documented. This is the version this project's
release tag (prepared, not created, by this repo — see
`docs/decisions/0011-phase6-documentation-and-release-tooling.md`) points
at.

### Phase 6 — Portfolio hardening
- Filled the previously-flagged documentation gap in `docs/model-math.md`:
  full derivations for RMSNorm, RoPE, causal self-attention, and SwiGLU,
  transcribed from (and cross-referenced to) the actual implementation.
- Added Mermaid architecture diagrams (forward pass; training loop) to
  `docs/architecture.md` — the spec §1.6 "architecture diagram of the
  forward/training path" deliverable.
- Committed a real sample-generation report
  (`benchmarks/sample_generations/toy_sample_report.{json,md}`) —
  previously only reproducible locally into the gitignored `artifacts/`
  path, never committed as portfolio evidence. See ADR 0011.
- Rewrote `README.md`: full model-config table (all `ModelConfig` fields
  for the three Phase 5 scaling configs), a "What I implemented myself"
  section, embedded benchmark plots and the sample-generation excerpt,
  and a from-scratch fresh-clone verification transcript.
- Added `docs/decisions/0011-phase6-documentation-and-release-tooling.md`
  (diagram tooling, committed-sample-generations location, versioning
  scheme).
- Added `CHANGELOG.md` (this file) and
  `tests/unit/test_release_metadata.py` (version/changelog/ADR-numbering
  consistency checks).
- Fresh-clone verification performed and recorded in
  `docs/reproducibility.md`: a real `git clone` into a scratch directory,
  `scripts/setup_env.sh`, `scripts/run_checks.sh`, and a small
  tokenizer→dataset→train→generate run, all followed verbatim from the
  polished `README.md` — install (1m18s), lint/type checks (clean), full
  test suite (269 passed in 64.29s), and the toy train→generate→evaluate
  output matched the committed sample-generation report byte-for-byte on
  every numeric field. This run caught and corrected an inaccurate
  inherited claim ("~40s on an RTX 4090" for the toy demo, which actually
  runs `device: cpu` per its own config in ~2.4s) in `README.md` and
  ADR 0011.
- Post-review correction: `README.md` and `benchmarks/methodology.md`
  claimed the context-length/batch-size peak-memory sweeps stayed
  "well under 1%" of the RTX 4090's 24 GB at every point measured.
  Recomputed `peak_memory_bytes / hardware.gpu_total_memory_bytes` from
  every committed `benchmarks/results/*.json` at full precision: the
  `batch_size_64.json` run is actually **1.7654%** (three of the fourteen
  runs exceed 1%; none exceed 2%). Both files now state the correct
  peak (1.77%, `batch_size_64.json`) instead of the false "under 1%"
  bound.

### Phase 5 — Scaling experiment
- `forgelm.benchmarks`: throughput/step-time/peak-memory/val-loss-vs-tokens
  measurement harness (`run_benchmark`, reusing `Trainer` for real steps).
- Three model-size configs (small/medium/larger: 164,160 / 1,115,264 /
  6,425,856 parameters), a context-length sweep, a batch-size sweep, and
  an optional `torch.compile` A/B — 14 runs, measured on an RTX 4090.
- `scripts/run_phase5_benchmarks.py` (experiment driver),
  `scripts/plot_benchmarks.py` (renders `benchmarks/plots/*.png` from the
  committed `benchmarks/results/*.json`), `benchmarks/methodology.md`.
- ADR 0010 (D8: Phase 5 compute budget).

### Phase 4 — Generation & evaluation
- `forgelm.generation`: checkpoint loading independent of `Trainer`, and
  greedy / temperature / top-k / top-p (nucleus) sampling, seeded and
  reproducible from `(prompt, GenerationSettings)` alone.
- `forgelm.evaluation`: token-weighted loss/perplexity
  (`evaluate_loss`) over a configurable token budget, and a combined
  JSON + Markdown "sample report" (`forgelm sample-report`).
- `forgelm generate` / `forgelm evaluate` / `forgelm sample-report` CLI
  commands.

### Phase 2–3 — Transformer architecture & training system
- `forgelm.model`: RMSNorm, RoPE (rotate-half formulation), causal
  multi-head self-attention, SwiGLU-gated MLP, pre-norm
  `TransformerBlock`, full `TransformerDecoder` with tied
  embedding/output weights (D4) — built from plain `torch.nn`/tensor ops,
  no `transformers`-library model class.
- `forgelm.training`: AdamW with decay/no-decay parameter groups, linear
  -warmup + cosine-decay LR schedule, gradient clipping, gradient
  accumulation, bf16 autocast on CUDA with fp32 fallback (D6), checkpoint
  save/resume with full RNG-state capture and a `config_hash` guard (D7).
- `forgelm train` CLI command; overfit-a-tiny-batch and
  train→save→load→generate integration tests (spec §1.8).
- ADR 0006 (D4: weight tying), ADR 0007 (D5: GQA deferred to stretch),
  ADR 0008 (D6: precision strategy), ADR 0009 (D7: checkpoint format).

### Phase 0–1 — Repo foundation, tokenization & data
- Repo scaffold, `pyproject.toml`, `scripts/setup_env.sh` /
  `scripts/run_checks.sh`, typed-dataclass config layer
  (`forgelm.config`), global RNG seeding (`forgelm.seed`), the `forgelm
  smoke` deterministic placeholder command, GitHub Actions CI
  (lint + type check + tests).
- `forgelm.tokenizer`: trainable byte-level BPE, deterministic,
  lossless round trip, special-token handling, save/load to JSON
  (~280 LOC, under D1's 500-LOC ceiling).
- `forgelm.dataset`: text → token array → deterministic contiguous
  train/val split → fixed-length next-token windows → seeded
  `DataLoader`.
- Committed example corpus (`examples/alice_in_wonderland.txt`, public
  domain — see `examples/README.md`), keeping tokenizer/dataset tests
  fully offline.
- ADR 0001 (config layer), ADR 0002 (D1: tokenizer scope), ADR 0003 (D2:
  dataset choice), ADR 0004 (D3: CPU-only dev path), ADR 0005 (dataset
  module naming).

[0.1.0]: https://github.com/Anish-Guntreddi/passion-projects/tree/main/projects/01-forgelm
