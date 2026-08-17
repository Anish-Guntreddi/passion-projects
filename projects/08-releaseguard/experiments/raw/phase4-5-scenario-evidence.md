# Phase 4-5 scenario evidence: `controller simulate` against the real default policy

Real, measured output from running all five spec §1.8 demo scenarios
(A-E) through `go run ./cmd/controller simulate`, using the actual policy
file [`controller/policy/examples/demo-service-default.yaml`](../../controller/policy/examples/demo-service-default.yaml)
and the actual fixture files under
[`controller/scenarios/`](../../controller/scenarios/) -- not summarized
from memory. Commands run from `projects/08-releaseguard/controller` in
WSL2 Ubuntu (Go 1.22.2), 2026-08-17. This is what Phase 4's exit criterion
("recorded telemetry fixtures produce deterministic decisions") and Phase
5's ("simulated rollout executes full state sequence") look like end to
end, using the same code path (`internal/scenario.Run`) the automated
tests in `controller/internal/scenario/scenario_test.go` assert on.

## Summary line per scenario

Each command's `# rollout=... final_state=... final_decision=...` summary
line, one per scenario, unedited:

```
$ go run ./cmd/controller simulate -policy policy/examples/demo-service-default.yaml -scenario scenarios/a-healthy-promotion.json -id demo-a
# rollout=demo-a policy=demo-service-canary final_state=COMPLETED final_decision=PROMOTE windows=3 promote_calls=1 rollback_calls=0

$ go run ./cmd/controller simulate -policy policy/examples/demo-service-default.yaml -scenario scenarios/b-latency-regression-rollback.json -id demo-b
# rollout=demo-b policy=demo-service-canary final_state=COMPLETED final_decision=ROLLBACK windows=2 promote_calls=0 rollback_calls=1

$ go run ./cmd/controller simulate -policy policy/examples/demo-service-default.yaml -scenario scenarios/c-error-regression-rollback.json -id demo-c
# rollout=demo-c policy=demo-service-canary final_state=COMPLETED final_decision=ROLLBACK windows=2 promote_calls=0 rollback_calls=1

$ go run ./cmd/controller simulate -policy policy/examples/demo-service-default.yaml -scenario scenarios/d-missing-telemetry-pause.json -id demo-d
# rollout=demo-d policy=demo-service-canary final_state=COMPLETED final_decision=PAUSE windows=2 promote_calls=0 rollback_calls=0

$ go run ./cmd/controller simulate -policy policy/examples/demo-service-default.yaml -scenario scenarios/e-transient-blip-promotion.json -id demo-e
# rollout=demo-e policy=demo-service-canary final_state=COMPLETED final_decision=PROMOTE windows=5 promote_calls=1 rollback_calls=0
```

Every scenario's `final_decision` matches what spec §1.8 asks that
scenario to demonstrate: A promotes, B and C roll back (latency and error
regressions respectively), D pauses without ever calling `Promote`
(`promote_calls=0`) despite missing telemetry, and E promotes despite one
degraded window (the "transient blip" never reaches
`consecutive_unhealthy_windows=2`, and 3 fresh consecutive healthy windows
follow it).

## Full audit trail: scenario A (healthy promotion) and scenario D (missing telemetry)

The complete state sequence for scenario A, showing the full
`PENDING -> DEPLOYING -> OBSERVING -> HEALTHY -> OBSERVING -> ... -> PROMOTING -> COMPLETED`
chain the state machine actually walks (`from_state`/`to_state` pairs,
reason codes only -- evidence payloads omitted here for length, see
`docs/slo-policy.md` for a full single-event example with evidence):

```
PENDING      -> DEPLOYING    reason_codes=[ROLLOUT_STARTED]
DEPLOYING    -> OBSERVING    reason_codes=[CANARY_DEPLOYED]
OBSERVING    -> HEALTHY      reason_codes=[ALL_SIGNALS_WITHIN_THRESHOLD]
HEALTHY      -> OBSERVING    reason_codes=[AWAITING_MORE_HEALTHY_WINDOWS]
OBSERVING    -> HEALTHY      reason_codes=[ALL_SIGNALS_WITHIN_THRESHOLD]
HEALTHY      -> OBSERVING    reason_codes=[AWAITING_MORE_HEALTHY_WINDOWS]
OBSERVING    -> HEALTHY      reason_codes=[ALL_SIGNALS_WITHIN_THRESHOLD]
HEALTHY      -> PROMOTING    reason_codes=[CONSECUTIVE_HEALTHY_WINDOWS_MET]  decision=PROMOTE
PROMOTING    -> COMPLETED    reason_codes=[ACTION_APPLIED]
```

Scenario D (missing telemetry -- decision PAUSE, never PROMOTE):

```
PENDING      -> DEPLOYING    reason_codes=[ROLLOUT_STARTED]
DEPLOYING    -> OBSERVING    reason_codes=[CANARY_DEPLOYED]
OBSERVING    -> INCONCLUSIVE reason_codes=[MISSING_TELEMETRY]
INCONCLUSIVE -> OBSERVING    reason_codes=[AWAITING_MORE_WINDOWS]
OBSERVING    -> INCONCLUSIVE reason_codes=[MISSING_TELEMETRY]
INCONCLUSIVE -> PAUSED       reason_codes=[PERSISTENT_INCONCLUSIVE_TELEMETRY, ON_MISSING_TELEMETRY_PAUSE]  decision=PAUSE
PAUSED       -> COMPLETED    reason_codes=[PAUSED_ROLLOUT_FINALIZED]
```

(Extracted from the real JSONL event stream `go run ./cmd/controller
simulate` printed for these two runs; `from_state`/`to_state`/
`reason_codes`/`decision` fields only, evidence payloads elided for
readability -- the full JSONL lines, with per-signal evidence attached to
every classification event, are reproduced by re-running the commands
above.)

## Full automated test suite backing this (Go, race-enabled)

```
$ go test ./... -race -cover -count=1
	releaseguard/controller/cmd/controller		coverage: 0.0% of statements
ok  	releaseguard/controller/internal/audit	1.079s	coverage: 85.7% of statements
ok  	releaseguard/controller/internal/evaluator	1.019s	coverage: 92.7% of statements
ok  	releaseguard/controller/internal/metrics	1.016s	coverage: 86.2% of statements
ok  	releaseguard/controller/internal/policy	1.019s	coverage: 92.1% of statements
ok  	releaseguard/controller/internal/rollout	1.010s	coverage: 83.9% of statements
ok  	releaseguard/controller/internal/scenario	1.078s	coverage: 81.0% of statements
```

80 top-level tests pass (`go test ./... -race -cover -count=1 -v 2>&1 | grep -c '^--- PASS'`
= 80; 110 including indented subtest lines), 0 fail. Transcript above
recaptured 2026-08-17 after the review-fix pass (evaluator coverage rose
88.1% → 92.7% with the fixer's added regression tests). `cmd/controller` shows 0.0% coverage because its two
subcommands are exercised via `go run` above (real process invocations,
not `go test` coverage instrumentation) rather than in-process Go tests --
the logic those subcommands call (`internal/policy`, `internal/scenario`,
`internal/rollout`, `internal/audit`) is what carries the package-level
coverage numbers above.

Re-run: `cd controller && go test ./... -race -cover -v` (from a fresh
clone; no cluster, no live Prometheus, no network access required for
Phase 4/5 -- everything here is fixture-driven).
