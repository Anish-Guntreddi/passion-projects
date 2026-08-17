# ADR 0003: Dataset choice for Phase 0/1 development (D2)

**Status:** Accepted for Phase 0/1; **resolved for Phase 5 too — see
"Phase 5 resolution" below.**
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

## Phase 5 resolution: Alice in Wonderland reused, not TinyStories

This ADR originally deferred to TinyStories for Phase 5, reasoning that a
single ~150 KB book "will not be enough" for a real loss curve across
≥3 model sizes. Phase 5's actual compute budget (ADR 0010: toy-to-small
models, under an hour of GPU time total) revised that assumption:

- The existing `artifacts/dataset/` build of
  `examples/alice_in_wonderland.txt` (vocab_size 512, context_length 128)
  already has **84,583 tokens** (76,124 train / 8,459 val). At the
  Phase 5 scaling configs' scale (164K–6.4M parameters) and step budget
  ((10 warmup + 300 measured) steps × 16 × 128 = 634,880 tokens per
  config by the final recorded val loss, ~8 epochs over the training
  split), this produces a real, non-degenerate val-loss-vs-tokens curve
  per config — the actual deliverable FR11 asks for — without needing
  more raw tokens than the committed corpus already has. (The 300
  *measured* steps alone cover 614,400 tokens; the harness's 10 real,
  weight-updating warmup steps push the actual total to 634,880 --
  see `benchmarks/results/scaling_*.json`'s `val_loss_history[-1].tokens`.)
- Downloading TinyStories would add a network dependency
  (`scripts/download_tinystories.py`, as originally sketched above) and a
  second license to track, for a benefit (a bigger, more diverse corpus)
  that is real but not necessary to satisfy Phase 5's actual exit
  criterion at this compute budget. D3 ("prefer deterministic local
  development over external managed services") favors staying with the
  already-committed, already-offline corpus when it suffices.
- This keeps the dataset identical across Phases 0–5, so every
  tokenizer/dataset artifact and test built during Phases 0–3 remains
  valid input to the Phase 5 benchmark harness with zero re-tokenization.

TinyStories (or any larger corpus) remains a reasonable choice for a
*future*, larger-budget scaling study — this resolution is specific to
this project's Phase 5 MVP scope, not a permanent judgment that the
smaller corpus is always sufficient.

## Consequences

- `examples/alice_in_wonderland.txt` is committed as source (not covered by
  the portfolio root `.gitignore`'s data-artifact patterns, since it lives
  under `examples/`, not a directory literally named `data`).
- `examples/README.md`'s note about "the plan for a larger corpus... once
  Phase 5 scaling experiments need more tokens" is superseded by the
  resolution above — updated to point here instead of promising a corpus
  swap that didn't end up happening.
- `scripts/run_phase5_benchmarks.py` and
  `docs/decisions/0010-phase5-compute-budget.md` are the record of
  exactly how the existing corpus was used for the Phase 5 scaling
  experiment.
