# ADR 0001: Config layer — plain dataclasses, not Pydantic

**Status:** Accepted
**Date:** 2026-08-17
**Context:** Phase 0, Tech Stack Plan §"Config" row (`01-forgelm-spec.md`): *"Typed dataclasses or Pydantic (pick one, ADR it)."*

## Decision

Use plain `@dataclasses.dataclass` for every config object (`SmokeConfig`,
`TokenizerTrainConfig`, `DatasetBuildConfig`, and future `ModelConfig` /
`TrainingConfig`), loaded from YAML by a small hand-written generic loader
in `forgelm/config.py` (`load_config`, `config_from_dict`). No Pydantic
dependency.

## Alternatives considered

- **Pydantic** (`BaseModel`): gets type coercion, JSON-schema export, and
  richer validators for free. Rejected for this project because:
  - It is a third-party dependency the whole config surface would run
    through; part of ForgeLM's brief is that the *mechanics* be visible and
    explainable (§1.7 "no magic global config"). A dataclass +
    ~120-line loader is small enough to read end-to-end in an interview and
    shows exactly what "config validation" means, rather than delegating it
    to a library's metaclass machinery.
  - The validation ForgeLM actually needs (type-check a handful of scalar
    fields, catch typos, enforce a few numeric invariants like
    `vocab_size >= 256 + len(special_tokens)`) does not need Pydantic's
    coercion/serialization feature surface.
- **Hydra**: explicitly ruled out by the tech stack table ("no Hydra unless
  justified — keep it simple"). Not reconsidered.
- **argparse-only, no config files**: rejected because FR1 requires
  "reproducible... CLI + config driven" execution, and later phases
  (training, benchmarking) need a config *artifact* to record alongside
  results, not just a set of flags.

## Consequences

- `forgelm/config.py`'s generic loader (`_build_dataclass`) handles nested
  dataclasses, `list[T]` fields, and `X | None` fields, which covers what
  this project's config tree needs through Phase 5. If a future phase needs
  real cross-field coercion (e.g. dtype strings -> `torch.dtype`), that goes
  in each dataclass's `__post_init__`, not in the generic loader.
- Unknown YAML keys are a hard error (`ConfigError`), which catches typos
  early — a property we'd otherwise have needed a validator library for.
- `frozen=True` on every config dataclass: configs are immutable once
  built, which matters once they start getting hashed into checkpoint
  metadata (D7, Phase 3).
