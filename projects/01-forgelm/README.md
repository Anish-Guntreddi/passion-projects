# ForgeLM

A from-scratch decoder-only Transformer training stack — tokenizer,
architecture, training loop, checkpointing, evaluation, and generation,
built without high-level pretrained-model wrappers. Portfolio project 01 of
9 (Track A: ML systems); prerequisite for MiniPaged.

Full spec: `01-forgelm-spec.md` (portfolio repo root).

## Status

Roadmap phases implemented so far: **0 (repo foundation)** and
**1 (tokenization & data)**. See `01-forgelm-spec.md` Part 3 for the full
6-phase roadmap; phases 2–6 (Transformer architecture, training system,
generation/evaluation, scaling experiments, portfolio hardening) are not
yet implemented — `forgelm.model`, `forgelm.training`, `forgelm.evaluation`,
`forgelm.generation`, and `forgelm.benchmarks` are boundary placeholders
only (see their module docstrings).

## What's implemented

- **Config layer** (`forgelm.config`): typed, validated dataclasses loaded
  from YAML, with unknown-field and type-mismatch errors caught at load
  time. See `docs/decisions/0001-config-layer-dataclasses.md`.
- **Seeding** (`forgelm.seed`): one call seeds Python `random`, NumPy, and
  torch (CPU + CUDA) and requests deterministic algorithms.
- **Smoke test** (`forgelm smoke`): a deterministic placeholder
  end-to-end command — Phase 0's exit criterion.
- **Byte-level BPE tokenizer** (`forgelm.tokenizer`): trainable,
  deterministic, lossless (`decode(encode(x)) == x`), special-token aware,
  save/load to JSON. ~280 LOC. See
  `docs/decisions/0002-tokenizer-scope.md`.
- **Dataset pipeline** (`forgelm.dataset`): text → tokens → deterministic
  contiguous train/val split → fixed-length next-token windows → seeded
  DataLoader. See `docs/decisions/0003-dataset-choice.md`.

## Quickstart (WSL2 Ubuntu / any Linux)

```bash
bash scripts/setup_env.sh          # creates .venv, installs torch (cu124) + deps
source .venv/bin/activate
bash scripts/run_checks.sh         # ruff + pyright + pytest (mirrors CI)
```

### Try the CLI

```bash
# Deterministic placeholder pipeline (Phase 0 exit criterion)
forgelm smoke --seed 1337 --size 64 --device cpu
forgelm smoke --config configs/smoke.yaml

# Train a tokenizer on the committed example corpus
forgelm tokenizer train --input examples/alice_in_wonderland.txt \
    --output artifacts/tokenizer.json --vocab-size 1024

forgelm tokenizer encode --model artifacts/tokenizer.json --text "Curiouser and curiouser!"
forgelm tokenizer decode --model artifacts/tokenizer.json --ids "72,101,108,108,111"

# Build train/val token arrays + dataset statistics
forgelm dataset build --input examples/alice_in_wonderland.txt \
    --tokenizer artifacts/tokenizer.json --output artifacts/dataset \
    --context-length 128 --val-fraction 0.1

# Equivalently, config-only (FR1: config file alone reproduces the run,
# including where artifacts land -- no --output flag needed):
forgelm tokenizer train --config configs/tokenizer_train.yaml
forgelm dataset build --config configs/dataset_build.yaml
```

## Repository layout

```
configs/                example YAML configs
src/forgelm/
  config.py, seed.py, constants.py   cross-cutting infrastructure
  tokenizer/             byte-level BPE (text <-> token ids)
  dataset/               text -> token arrays -> splits -> batches
                          (named `dataset/`, not `data/` — see ADR 0005)
  model/                 [Phase 2] architecture only
  training/              [Phase 3] optimizer/schedule/precision/checkpoint
  evaluation/             [Phase 4] loss/perplexity/sample reports
  generation/             [Phase 4] sampling from a checkpoint
  benchmarks/             [Phase 5] throughput/memory experiments
  cli/                    composes the above into `typer` commands
tests/unit/, tests/integration/
scripts/                 setup_env.sh, run_checks.sh
examples/                committed public-domain example corpus
benchmarks/results/, benchmarks/plots/, benchmarks/methodology.md  [Phase 5]
docs/architecture.md, docs/model-math.md, docs/reproducibility.md
docs/decisions/          ADRs
```

## Testing

See `docs/reproducibility.md` for the exact command and the last recorded
result. In short: `pytest -v` from an activated `.venv`. GPU-only tests
(`tests/integration/test_smoke_gpu.py`) skip automatically when no CUDA
device is visible.

## Engineering rules this repo follows

No notebook-only implementations; no copied full tokenizer/model
implementation from a mature library; tests ship with the functionality
that needs them; every open design decision (D1–D8 in the spec) gets an
ADR in `docs/decisions/` before the phase that depends on it; no benchmark
number is ever fabricated — only measured, committed, reproducible output.
