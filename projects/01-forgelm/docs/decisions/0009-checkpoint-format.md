# ADR 0009: Checkpoint format and RNG-state guarantees (D7)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D7 (`01-forgelm-spec.md` §1.9) — *"Checkpoint format +
RNG-state guarantees → default: torch.save dict with model/optimizer/
scheduler/RNG states + config hash; ADR required."*

## Decision

`forgelm.training.checkpoint.save_checkpoint` writes a single
`torch.save` dict (`CHECKPOINT_FORMAT_VERSION = 1`) containing:

| Key | Contents |
|---|---|
| `format_version` | integer, bumped on any breaking payload-shape change |
| `step` | optimizer-step counter at save time |
| `model_state_dict` | `model.state_dict()` |
| `optimizer_state_dict` | `optimizer.state_dict()` (AdamW moment buffers) |
| `scheduler_state_dict` | `LambdaLR.state_dict()` (schedule position) |
| `rng_state` | Python `random`, NumPy, torch CPU, torch CUDA (all visible devices) |
| `model_config` / `training_config` | `dataclasses.asdict(...)` of both, for inspection without reconstructing the dataclasses |
| `dataset_paths` | `[train_tokens_path, val_tokens_path]` as strings, or `None` when the caller didn't supply them |
| `config_hash` | SHA-256 of `model_config` + `training_config` + `dataset_paths` (`forgelm.training.checkpoint.config_hash`) |
| `val_loss` | last computed validation loss, or `None` |

`load_checkpoint` reads the raw payload back; `Trainer.load_checkpoint`
is the higher-level entry point that also restores it into a live
`Trainer` (model/optimizer/scheduler state, RNG state, step counter, and
a batch-stream position — see "Data-order determinism" below).

## Rationale for each piece

- **`torch.save` dict, not a custom binary format**: this is exactly the
  spec's recommended default, and it is what every real PyTorch training
  codebase does — no reason to build something bespoke.
- **`config_hash` gates resuming**: `Trainer.load_checkpoint` calls
  `verify_config_hash` before touching any state, so mismatched configs
  fail loudly and immediately with a clear message, instead of either (a)
  `load_state_dict` failing deep inside with a confusing tensor-shape
  error, or (b) — the worse case — *succeeding* while silently training
  under different hyperparameters than the checkpoint was produced with.
  `Trainer` itself only knows `model_config`/`training_config` (it's
  constructed from already-loaded `Dataset` objects, not paths), so
  `config_hash`'s optional `dataset_paths` argument exists for the one
  caller that *does* know the paths --
  `forgelm.cli.train_cli.run_training` always passes
  `(train_tokens_path, val_tokens_path)` through to both `Trainer(...)`
  and `Trainer.load_checkpoint(...)`, so `--resume-from` against a
  checkpoint produced from a different corpus/tokenizer run is also a
  loud `config_hash` mismatch, not a silent resume onto a different
  token stream. This is intentionally strict: resuming with a
  **different** config
  (e.g. a longer `max_steps` to continue training further than originally
  planned) is treated as an error, not a supported "extend training" flow,
  in this MVP. That is an explicit scope limit, not an oversight — the
  Phase 3 exit criterion is "interrupted training resumes with an
  equivalent trajectory" for the *same* planned run, which this satisfies
  exactly (see `tests/integration/test_checkpoint_resume.py`). A future
  phase that wants "resume and extend" would need to either exclude
  `max_steps` from the hashed payload or add an explicit
  `extend_max_steps` escape hatch — deliberately not built now.
- **Full RNG-state capture**: FR7 asks for this "where practical." All
  four RNGs this project seeds (see `forgelm.seed.set_seed`) are captured
  and restored, so any RNG consumption *inside* a training step (e.g.
  dropout masks, once `ModelConfig.dropout > 0`) reproduces identically
  after a resume, not just approximately.
- **`weights_only=False` on load**: torch's default `weights_only=True`
  unpickling only accepts tensors and a small allowlist of container
  types, and would reject this payload's RNG-state tuples and plain
  Python objects. This project's checkpoints are only ever produced by
  its own `save_checkpoint` and loaded from trusted local paths (D3:
  "prefer deterministic local development over external managed
  services" — no untrusted or networked checkpoint loading is in scope
  anywhere in this project), so the security risk `weights_only=True`
  guards against (arbitrary code execution via a malicious pickle) does
  not apply here. This is a deliberate, documented trade-off, not an
  oversight.

## Data-order determinism (batch stream position)

RNG state alone is not sufficient to reproduce an *equivalent trajectory*
across an interruption: the sequence of training batches also has to
match. `forgelm.dataset.iter_batches(dataset, batch_size, seed)` is
designed as a **pure function** of its three arguments (see its
docstring) — the entire infinite batch stream from step 0 onward is
reproducible from `(dataset, batch_size, seed)` alone, with no additional
hidden state. `Trainer.load_checkpoint` therefore does not need to
persist any batch-iterator state in the checkpoint itself: it recreates
the same generator from the checkpoint's `training_config.seed` and
`skip_batches = step * grad_accum_steps` (the number of micro-batches
already consumed), which reproduces the exact batches the uninterrupted
run would see from that point on.

## Alternatives considered

- **Persisting DataLoader/sampler iterator state directly** (e.g.
  `torch.utils.data.DataLoader`'s internal worker/sampler state):
  rejected — PyTorch's `DataLoader` does not expose a clean, versioned
  "resume this exact iterator" API, and rebuilding the equivalent
  guarantee from a pure seeded-generator function (as above) is simpler,
  smaller, and easier to test/explain than trying to serialize and
  restore that internal state.
- **Separate files per state component** (model/optimizer/RNG each their
  own file): rejected as unnecessary complexity for this project's scale
  — a single `torch.save` dict is simpler to move around, hash, and
  reason about as one atomic artifact.

## Consequences

- `tests/unit/test_training_checkpoint.py` covers: `config_hash`
  determinism and sensitivity to config *and* `dataset_paths` changes;
  RNG-state capture/restore round-tripping; a raw save/load round trip
  preserving every payload key (including `dataset_paths`);
  `verify_config_hash` raising on a mismatched config or a mismatched
  dataset.
- `tests/integration/test_checkpoint_resume.py::
  test_trainer_resume_reproduces_equivalent_loss_trajectory` is the
  Phase 3 exit-criterion proof: an uninterrupted N-step run and an
  interrupted-at-K-then-resumed run (same config throughout) produce
  matching per-step losses for steps K..N, within a numerical tolerance
  (never exact float equality, per the project-wide rule).
- `tests/integration/test_cli_subprocess.py::
  test_cli_train_resume_against_a_different_dataset_raises_a_clean_error`
  is the real-CLI proof of the `dataset_paths` guard above: `forgelm
  train --config ... --resume-from ...` against a checkpoint produced
  from a different corpus fails with a `config_hash` error instead of
  silently resuming onto the wrong token stream.
