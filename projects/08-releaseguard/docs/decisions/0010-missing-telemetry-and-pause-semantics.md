# ADR 0010: `on_missing_telemetry` defaults to pause; PAUSED resolves via explicit Finalize

**Status:** Accepted; implemented in Phase 4-5 (`controller/internal/policy`, `controller/internal/rollout`)
**Decision ID:** none in spec §1.9 (D1-D7 are all already recorded); this
ADR documents two structural design choices Phase 4-5 had to make to
satisfy FR4 and Claude Code handoff rule 1, in the same spirit as D1-D7 --
recorded per the brief's instruction to write an ADR for open decisions
this phase introduces.
**Date:** 2026-08-17

## Context

FR4 requires a typed policy to configure "action on missing telemetry,"
and separately states as a hard rule (Claude Code handoff rule 1): "never
auto-promote with missing evidence." Two concrete design questions follow
directly from that requirement that spec §1.9 does not itself resolve:

1. What are the valid values of `on_missing_telemetry`, and which one is
   the default?
2. The spec's lifecycle chain ends `PAUSED -> COMPLETED/FAILED`, but this
   MVP (per D5/ADR 0006) has no human-in-the-loop resume API -- so what
   does a rollout actually *do* once it reaches PAUSED?

## Decision

**`on_missing_telemetry`** (`controller/internal/policy.MissingTelemetryAction`)
is a closed two-value enum: `pause` or `rollback`. There is no `promote`
value, and none was ever added as an option a policy author could reach
for -- rule 1 is enforced by the schema having no such escape hatch, not
by a runtime check that a future change could accidentally bypass. When a
policy document omits the field entirely, `Policy.ApplyDefaults` (called
from `Parse`/`Load` before validation) fills in `pause` -- the strictly
safer of the two valid choices: it takes no rollout action at all and
waits, versus `rollback`, which does take an action based on absent
evidence (defensible as a stricter posture some policies may want, but not
the default).

**PAUSED resolution**: `rollout.Rollout.Finalize` is the explicit,
separately-callable method that moves `PAUSED -> COMPLETED`. Nothing
inside `RecordWindow` auto-finalizes a paused rollout. The driver that
owns the rollout's lifecycle -- today, `internal/scenario.Run` for
simulations and the `simulate` CLI command; a live Phase 6 reconcile loop
later -- decides when to call `Finalize`, and today's callers all call it
immediately after observing `StatePaused`, because this MVP has no
operator-facing "resume" action to wait for. `PAUSED -> COMPLETED` is
therefore still a real, separately audited transition (visible as its own
JSONL line, `"reason_codes":["PAUSED_ROLLOUT_FINALIZED"]`), not something
folded silently into the pause transition itself.

## Consequences

- A policy file can never be configured to promote on missing evidence --
  this is enforced at three independent layers: the schema (no `promote`
  value exists), `policy.Validate` (rejects anything outside the two valid
  values), and the state machine's transition table (`allowedTransitions`
  in `controller/internal/rollout/state.go`), which has no edge from
  `StateInconclusive` directly to `StatePromoting` at all. A test
  (`TestAllowedTransitions_OnlyHealthyLeadsToPromoting`) checks the third
  layer exhaustively over every table entry, not just a spot check.
- Separating "reach PAUSED" from "resolve PAUSED" keeps the pause itself
  fully visible in the audit trail as a distinct, deliberate stopping
  point -- matching FR4's "why did ReleaseGuard make this decision"
  requirement -- rather than a state a human reading the JSONL trail could
  miss because it was collapsed into the next line.
- The most visible near-term cost: without a real resume API, every
  simulated/CLI-driven paused rollout finalizes to COMPLETED immediately,
  which reads a little oddly in isolation ("paused, then immediately
  completed") until the human-in-the-loop resume mechanism exists. That
  mechanism is out of MVP scope (the spec's stretch goals do not even
  explicitly list it, though it is implied by any real operational use of
  PAUSE) and is deferred rather than half-built now.
- `rollback` remains available for a policy author who has a specific,
  reviewed reason to treat "we can't observe this canary" as
  failure-equivalent rather than merely inconclusive (e.g. a team that
  considers observability itself part of the release contract). Choosing
  it is a per-policy decision, not a framework default.

## Alternatives considered

- **Auto-resume PAUSED after a fixed grace period**: rejected -- this
  would just be `max_rollout_duration` under a different name applied a
  second time, adding complexity without a new capability, and blurs the
  deadline's own "we ran out of time" semantics (see `RecordWindow`'s
  deadline check, which already routes to `StatePaused` with decision
  `INCONCLUSIVE` on its own, independent of `on_missing_telemetry`).
- **Three-value `on_missing_telemetry` including an explicit
  `promote-if-otherwise-healthy` escape hatch**: rejected outright as a
  direct violation of Claude Code handoff rule 1; not seriously
  considered as a default, and not offered as an option at all.
- **Fold `PAUSED -> COMPLETED` into the same transition as reaching
  `PAUSED`**: rejected because it would remove PAUSED as an independently
  observable point in the audit trail, weakening the "why" story FR4
  asks for.
