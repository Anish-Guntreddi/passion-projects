# ADR 0011: Phase 6 documentation, diagram, and release tooling

**Status:** Accepted
**Date:** 2026-08-17
**Context:** Phase 6 ("portfolio hardening") introduces three choices the
spec leaves implicit rather than open-decision-labeled (D1–D8 are all
already resolved by ADRs 0001–0010) — this ADR records them per the
project's "record architecture/tooling decisions in ADRs" rule (spec Part
4, agent execution rule 7), rather than leaving them undocumented because
no `D`-numbered slot exists for them.

## Decision 1: diagrams are GitHub-native Mermaid, not a rendered-image toolchain

`docs/architecture.md`'s forward-pass and training-loop diagrams (the
spec §1.6 "architecture diagram of the forward/training path" deliverable)
are written as ```` ```mermaid ```` fenced code blocks, rendered natively
by GitHub's markdown viewer — not exported PNG/SVG files produced by a
separate diagramming tool (Excalidraw, draw.io, graphviz, etc.).

**Rationale:**
- **No new dependency.** This project's only rendering dependency anywhere
  is the already-existing optional `matplotlib` extra
  (`pip install -e ".[plots]"`) for the Phase 5 benchmark plots — those
  are *data* plots (numbers read from committed JSON), a fundamentally
  different artifact from a *structural* diagram, and don't belong to the
  same tool. Adding a diagramming binary/service just for two structural
  diagrams would violate the "prefer the simplest design" spec guidance
  (§1.9) for no functional gain.
- **Diagrams stay a diff-reviewable text file**, not a binary blob — a
  future architecture change (e.g. GQA, spec D5's deferred stretch goal)
  updates a few lines of Mermaid syntax in the same PR as the code change,
  the same way the existing ASCII data-flow diagrams in
  `docs/architecture.md` always have been kept in sync.
- **Renders correctly wherever this repo is read**: GitHub's web UI,
  GitHub's markdown preview, and most modern IDEs (VS Code with a Mermaid
  extension) render these fenced blocks without any build step; a reader
  without Mermaid support still sees valid, readable pseudo-code-shaped
  text (unlike a missing/broken image link).

**Alternative considered:** static SVG/PNG diagrams committed under a
`docs/assets/` folder. Rejected for this MVP — it would need either a
manual diagramming tool (not automatable, not text-diffable) or adding a
Mermaid-CLI/headless-browser dependency to render Mermaid to an image
file, neither of which is justified by two diagrams whose primary
consumer (GitHub's own markdown renderer) already renders the text form
natively.

## Decision 2: sample generations live in `benchmarks/sample_generations/`, not `artifacts/`

`forgelm sample-report`'s default output path
(`configs/sample_report_example.yaml`) writes into `artifacts/reports/`,
which is deliberately gitignored (Phase 4: "running this doesn't dirty
the repo" — see that config's comments). That is correct for local,
throwaway runs, but it means Phase 4's original sample report was never
committed evidence — which conflicts with spec §1.6's "sample generations
with documented decoding settings" being a listed **deliverable
artifact**, and with the portfolio-wide rule that every claim (here: "the
generation pipeline runs end-to-end and produces this output") traces to
a committed artifact, not a locally-regenerable-but-unverified one.

**Decision:** a new config, `configs/sample_report_portfolio.yaml` — byte
-for-byte identical to `configs/sample_report_example.yaml` except
`output_path: benchmarks/sample_generations/toy_sample_report` — writes
into `benchmarks/`, which is *not* gitignored (Phase 5 already committed
`benchmarks/results/*.json` and `benchmarks/plots/*.png` there as
measured evidence). `benchmarks/sample_generations/toy_sample_report.json`
and `.md` are the result, generated from this project's actual trained
toy checkpoint (`configs/train_toy.yaml`, 50 steps) with the exact
decoding settings recorded inside the file itself (temperature 0.8,
top-k 40, seeds 1337–1339) — real output, not edited or invented text.

**Honesty note carried into the README:** this checkpoint is
intentionally tiny (Phase 0-3's CPU-only, ~2.4-second demo config — see
`configs/train_toy.yaml`'s `device: cpu` — not a Phase 5
scaling-experiment size), so its generations are not fluent English — the
committed sample proves the checkpoint -> load -> decode pipeline runs
correctly end-to-end and is reproducible from documented settings, which
is what FR8/§1.6 ask for; it is not presented as a claim about generation
*quality*. `configs/sample_report_example.yaml` (writing to the
gitignored `artifacts/` path) remains available unchanged for local,
non-committed iteration.

## Decision 3: versioning is SemVer via `CHANGELOG.md`; the git tag itself is orchestrator-owned

`pyproject.toml`'s `version = "0.1.0"` (unchanged since Phase 0) is
declared the MVP release version, tracked in a new root `CHANGELOG.md`
(Keep a Changelog format) with one dated entry per roadmap phase. This
repo does **not** create the `v0.1.0` git tag itself — per this
repository's hard operating rule, only the multi-project orchestrator
runs state-mutating git commands (tag creation included); this ADR and
`CHANGELOG.md` are the "release tag prep" the spec's Phase 6 row asks for
(the tag's exact contents and message are fully determined by what's
committed at this point), not the tag-creation command itself.

## Consequences

- `tests/unit/test_release_metadata.py` asserts `pyproject.toml`'s
  `version` matches `CHANGELOG.md`'s latest (topmost) version heading, and
  that `docs/decisions/*.md` is sequentially numbered with no gaps — a
  small, cheap regression guard against the two documentation artifacts
  drifting apart, not a substitute for the manual fresh-clone
  verification below.
- `docs/reproducibility.md` records a real, from-scratch fresh-clone
  verification transcript (a separate `git clone` of this repository into
  a scratch directory, following `README.md` verbatim) as the concrete
  proof of Phase 6's exit criterion, per the spec's Definition of Done:
  "Fresh clone installs and passes tests; small training run
  reproducible."
