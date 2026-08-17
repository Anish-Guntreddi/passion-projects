# ForgeLM

A from-scratch decoder-only Transformer training stack — tokenizer,
architecture, training loop, checkpointing, evaluation, and generation,
built without high-level pretrained-model wrappers. Portfolio project 01 of
9 (Track A: ML systems); prerequisite for MiniPaged.

Full spec: `01-forgelm-spec.md` (portfolio repo root).

## Status

Roadmap phases implemented so far: **0–5** (repo foundation; tokenization
& data; Transformer architecture; training system; generation &
evaluation; scaling experiments). See `01-forgelm-spec.md` Part 3 for the
full 6-phase roadmap; **Phase 6** (portfolio hardening: polished
diagrams, release tag) is not yet implemented.

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
- **Decoder-only Transformer** (`forgelm.model`): token embeddings (tied
  to the output projection, ADR 0006), causal multi-head self-attention
  with RoPE, RMSNorm, a SwiGLU-gated MLP, and pre-norm residual blocks —
  built from plain `torch.nn`/tensor ops, no `transformers`-library model
  class anywhere. See `docs/decisions/0007-gqa-deferred.md` and
  `docs/decisions/0008-precision-strategy.md`.
- **Training system** (`forgelm.training`): AdamW with decay/no-decay
  parameter groups, linear-warmup + cosine-decay LR schedule, gradient
  clipping/accumulation, bf16 autocast on CUDA (fp32 fallback),
  checkpoint save/resume with full RNG-state capture and a `config_hash`
  guard against resuming onto a mismatched config or dataset. See
  `docs/decisions/0009-checkpoint-format.md`.
- **Generation** (`forgelm.generation`): load a trained model straight
  from a checkpoint (independent of any `Trainer`/optimizer state), then
  generate with greedy, temperature, top-k, and/or top-p (nucleus)
  decoding — reproducible from `(prompt, seed, decoding settings)` alone.
- **Evaluation** (`forgelm.evaluation`): token-weighted loss/perplexity
  over a configurable token budget, plus a combined JSON + Markdown
  "sample report" (checkpoint's loss/perplexity + documented-settings
  generations for a list of prompts) via `forgelm sample-report`.
- **Benchmark harness** (`forgelm.benchmarks`): tokens/sec, step time,
  peak GPU memory (`None` on CPU — FR10's CPU-fallback requirement),
  and a val-loss-vs-tokens-trained curve, measured by actually running
  real `Trainer` steps. See `docs/decisions/0010-phase5-compute-budget.md`
  and `benchmarks/methodology.md` for the Phase 5 scaling-experiment
  results.

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

# Train a tiny model (reproducible end-to-end demo, ~40s on an RTX 4090)
forgelm train --config configs/train_toy.yaml

# Generate from the trained checkpoint (FR8)
forgelm generate --checkpoint artifacts/checkpoints/final.pt \
    --tokenizer artifacts/tokenizer.json \
    --prompt "Alice was beginning to" --max-new-tokens 40 \
    --temperature 0.8 --top-k 40

# Loss/perplexity of the checkpoint against the val split (FR9)
forgelm evaluate --checkpoint artifacts/checkpoints/final.pt \
    --tokens artifacts/dataset/val_tokens.npy --batch-size 8

# Combined evaluation + documented-settings generations, written as
# JSON + Markdown (FR9's "sample report" deliverable)
forgelm sample-report --config configs/sample_report_example.yaml

# One throughput/memory/val-loss measurement (Phase 5)
forgelm benchmark run --config configs/benchmark_example.yaml
```

## Repository layout

```
configs/                example YAML configs (tokenizer/dataset/train/
                        generate/sample-report/benchmark)
src/forgelm/
  config.py, seed.py, constants.py   cross-cutting infrastructure
  tokenizer/             byte-level BPE (text <-> token ids)
  dataset/               text -> token arrays -> splits -> batches
                          (named `dataset/`, not `data/` — see ADR 0005)
  model/                 decoder-only Transformer architecture only
  training/              optimizer/schedule/precision/checkpoint
  evaluation/             loss/perplexity + sample-report format
  generation/             checkpoint loading + sampling (greedy/temp/top-k/top-p)
  benchmarks/             throughput/memory/val-loss measurement harness
  cli/                    composes the above into `typer` commands
tests/unit/, tests/integration/
scripts/                 setup_env.sh, run_checks.sh,
                        run_phase5_benchmarks.py, plot_benchmarks.py
examples/                committed public-domain example corpus
benchmarks/results/, benchmarks/plots/, benchmarks/methodology.md  (Phase 5)
docs/architecture.md, docs/model-math.md, docs/reproducibility.md
docs/decisions/          ADRs
```

## Benchmarks (Phase 5)

Three model sizes (164K / 1.12M / 6.43M parameters), a context-length
sweep, a batch-size sweep, and an optional `torch.compile` A/B — measured
on an RTX 4090, 14 runs in 1.54 min total wall-clock. Full results,
methodology, and plots: `benchmarks/methodology.md`,
`benchmarks/results/*.json`, `benchmarks/plots/*.png`.

| Config | Params | Tokens/sec | Peak GPU memory | Val loss (634,880 tokens trained) |
|---|---|---|---|---|
| small | 164,160 | 308,148 | 59.3 MiB | 4.029 |
| medium | 1,115,264 | 178,848 | 131.8 MiB | 3.416 |
| larger | 6,425,856 | 126,864 | 377.3 MiB | 3.042 |

`torch.compile` gave a measured **37.1%** throughput improvement on the
small config (276,061 → 378,474 tokens/sec) on this GPU/torch build.
Compute budget rationale: `docs/decisions/0010-phase5-compute-budget.md`.

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
