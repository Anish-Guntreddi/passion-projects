# Architecture

Status: reflects Phases 0–1 (repo foundation, tokenization, dataset
construction). Will be extended as Phases 2–6 land — the model/training/
evaluation/generation/benchmarks sections below are boundary placeholders
until then, not yet implemented.

## Module boundaries

Each top-level package under `src/forgelm/` owns exactly one concern. This
is enforced by convention (reviewed in `/review` passes and ADRs), not by
a runtime dependency-checker:

| Module | Owns | Must not contain |
|---|---|---|
| `forgelm.tokenizer` | text ↔ token id conversion (train/encode/decode/save/load) | dataset splitting, model code |
| `forgelm.dataset` | reading text, packing fixed-length examples, train/val split, batching | tokenizer algorithm internals, model/training code — see `docs/decisions/0005-dataset-module-naming.md` for why this isn't named `data/` |
| `forgelm.model` *(Phase 2)* | architecture modules only (embeddings, attention, RoPE, RMSNorm, MLP, block, full decoder) | optimizer, scheduler, checkpointing |
| `forgelm.training` *(Phase 3)* | optimization loop, LR schedule, clipping, mixed precision, grad accumulation, checkpointing | model architecture, benchmark harness |
| `forgelm.evaluation` *(Phase 4)* | loss/perplexity computation, sample-report generation | training loop, benchmark timing |
| `forgelm.generation` *(Phase 4)* | sampling (greedy/temperature/top-k/top-p) from a checkpoint | training logic |
| `forgelm.benchmarks` *(Phase 5)* | throughput/memory experiments | model-correctness assertions (kept separate per §1.7) |
| `forgelm.cli` | composes the above into `typer` commands | any of the above modules' actual logic — CLI code only builds configs, calls a library function, and prints the result |
| `forgelm.config` | typed dataclasses + YAML loader, used by every other module | — |
| `forgelm.seed` | global RNG seeding | — |

## Phase 0–1 data flow

```
                 ┌─────────────────────┐
 configs/*.yaml  │                     │
       or        │   forgelm.config    │  load_config() -> typed, validated
 CLI flags   ───▶│  (dataclasses)      │  dataclass instance
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
 raw text file   │  forgelm.tokenizer  │  train(): text -> merge rules
  (examples/,    │  ByteLevelBPE       │  encode(): text -> list[int]
   or any path)  │  Tokenizer          │  decode(): list[int] -> text (lossless)
                 └──────────┬──────────┘
                            │ tokenizer.json (merges + special tokens)
                            ▼
                 ┌─────────────────────┐
                 │  forgelm.dataset    │  tokenize_corpus() -> flat token array
                 │                     │  train_val_split() -> deterministic,
                 │                     │    contiguous split (no RNG needed)
                 │                     │  NextTokenDataset -> (x, y) windows
                 │                     │  make_dataloader() -> seeded batch order
                 └──────────┬──────────┘
                            │
                            ▼
                 (Phase 2+: forgelm.model consumes (x, y) batches)
```

Every arrow above is exercised end-to-end by
`tests/integration/test_pipeline.py`, including the property that
decoding the concatenation of train + val token arrays reproduces the
exact source text.

## Why the CLI has no logic of its own

`forgelm/cli/main.py` only builds a config dataclass (from flags or a
`--config` YAML file) and calls a plain function from `forgelm.cli.smoke`,
`forgelm.cli.tokenizer_cli`, or `forgelm.cli.dataset_cli` — each a thin
wrapper that itself only calls into `forgelm.tokenizer` / `forgelm.dataset`.
This keeps every non-CLI module directly unit-testable without spawning a
subprocess, while still getting subprocess-level integration coverage of
the actual command line (`tests/integration/test_cli_subprocess.py`).

## Seeding and determinism

`forgelm.seed.set_seed(seed)` is the single entry point every CLI command
and test that needs reproducibility calls before doing anything else. It
seeds Python's `random`, NumPy, and torch (CPU + all visible CUDA devices),
and requests deterministic algorithms from torch with `warn_only=True` (a
handful of ops used later in the project may not have a deterministic CUDA
implementation; we'd rather warn than crash on those specific ops when we
get there). See `docs/reproducibility.md` for the full reproducibility
story and `docs/decisions/0004-cpu-only-dev-path.md` for the CPU/GPU
policy.
