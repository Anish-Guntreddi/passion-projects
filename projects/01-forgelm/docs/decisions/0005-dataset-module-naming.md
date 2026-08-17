# ADR 0005: `src/forgelm/dataset/` instead of `src/forgelm/data/`

**Status:** Accepted, flagged for orchestrator review
**Date:** 2026-08-17

## Context

`01-forgelm-spec.md`'s repository structure names the module that "owns
datasets/packing/splits" `data/`:

```
src/forgelm/
    tokenizer/  data/  model/  training/
    evaluation/ generation/ benchmarks/ cli/
```

The **portfolio root** `.gitignore` (outside this project's directory, not
editable by this implementer) contains:

```
projects/**/data/
```

Git's gitignore `**` glob matches any number of path segments, including
zero, between `projects/` and `data/`. That pattern therefore matches *any*
directory literally named `data` anywhere under `projects/`, including
`projects/01-forgelm/src/forgelm/data/` — not only a project's top-level
runtime data directory, which is what the comment above that line
("Large training/data artifacts... raw data and weights are not
[committed]") suggests it was meant to target. Once a directory itself is
excluded, git will not descend into it to re-include individual files even
with negating patterns in a nested `.gitignore`, so this would silently
exclude this module's *source code* from version control.

## Decision

Name the module `src/forgelm/dataset/` instead of `src/forgelm/data/`. All
other module boundaries match the spec exactly (`tokenizer/`, `model/`,
`training/`, `evaluation/`, `generation/`, `benchmarks/`, `cli/`). The
package's public API (`forgelm.dataset`) still owns exactly what the spec
assigns to the `data` boundary: reading text, tokenizing, building
fixed-length next-token examples, and deterministic train/val splitting.
This is a naming change only, not a scope or boundary change.

## Consequences / follow-up needed

- Anywhere this project's own docs (README, architecture doc) refer to "the
  data module," they mean `forgelm.dataset`.
- **Flagging for the orchestrator**: the root `.gitignore` pattern
  `projects/**/data/` likely needs to be narrowed (e.g. to
  `projects/*/data/`, matching only a *top-level* per-project `data/`
  directory, or renamed to target a more specific convention like
  `projects/**/raw_data/`) if other projects in this portfolio also want a
  source-code directory named `data`. This implementer did not modify the
  root `.gitignore`, per the hard constraint against touching anything
  outside its assigned project directory — see the final phase report.
