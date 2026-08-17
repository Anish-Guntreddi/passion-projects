# ADR 0003: Dataset choice for Phase 0/1 development (D2)

**Status:** Accepted for Phase 0/1; revisit before Phase 5
**Date:** 2026-08-17
**Open decision:** D2 (`01-forgelm-spec.md` §1.9) — *"Dataset: small, legal,
reproducible → default: TinyStories or a public-domain text corpus; document
license."*

## Decision

Use a **public-domain text corpus** — *Alice's Adventures in Wonderland* by
Lewis Carroll, sourced unmodified from Project Gutenberg — as the committed
example/dev/test dataset for Phases 0–1, in preference to TinyStories.

Committed at `examples/alice_in_wonderland.txt` (~150 KB); provenance and
license documented in `examples/README.md`.

## Rationale

- **No network dependency in CI or tests.** TinyStories is distributed via
  the Hugging Face Hub (`datasets` library or direct download), which would
  make `pytest` — and any GitHub Actions run — depend on network access and
  an external service's availability. Portfolio-wide rule 8 explicitly
  prefers "deterministic local development over external managed
  services." A corpus committed directly to the repo sidesteps this
  entirely.
- **Simpler licensing story.** Gutenberg public-domain texts have a
  well-understood, permissive redistribution policy (keep the
  start/end markers, which this file does). TinyStories is CDLA-Sharing-1.0
  licensed by Microsoft Research — usable, but a second license to track
  for no benefit at this phase.
- **Sufficient for Phase 0/1's job.** Phases 0–1 need a real (not
  single-sentence) corpus to exercise tokenizer training, dataset
  statistics, and the train/val split — not a large or diverse one. ~150 KB
  of continuous English prose exercises the byte-level BPE trainer and the
  `NextTokenDataset` windowing correctly.
- Unit tests additionally use a small synthetic string (`tests/conftest.py`
  `tiny_corpus`) so the fast unit suite doesn't depend on any file I/O or
  the corpus's exact content.

## Deferred: TinyStories for Phase 5

The spec's D2 recommends "TinyStories **or** a public-domain text corpus" —
either satisfies the open decision. TinyStories remains the better fit for
the Phase 5 scaling experiments (FR11), which need enough tokens across
≥3 model-size configurations to show a real loss curve; a single ~150 KB
book will not be enough there. When Phase 5 is implemented, add a
`scripts/download_tinystories.py` (network-dependent, run manually / not
part of `pytest`) and a corresponding ADR update recording the exact
dataset revision, license, and token count used, per the "no resume metric
without committed evidence" project rule.

## Consequences

- `examples/alice_in_wonderland.txt` is committed as source (not covered by
  the portfolio root `.gitignore`'s data-artifact patterns, since it lives
  under `examples/`, not a directory literally named `data`).
- Phase 5 will need a second ADR entry (or an amendment here) once the
  scaling-experiment dataset is actually chosen and downloaded — no dataset
  claim beyond Phase 0/1's example corpus is made by this ADR.
