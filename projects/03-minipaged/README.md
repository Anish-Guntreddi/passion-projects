# MiniPaged

MiniPaged is a compact, understandable LLM inference runtime demonstrating
the systems ideas behind modern high-throughput serving: request
scheduling, prefill/decode separation, paged KV-cache allocation,
continuous batching, admission control, and streaming token output. It
deliberately does **not** clone vLLM — it is a smaller system whose data
structures and scheduling decisions can be explained in a technical
interview.

The full product/roadmap spec lives at `../../03-minipaged-spec.md` in the
portfolio repo root. This README covers what exists *today*.

## Status: Phase 0 + Phase 1 of 8

Per the spec's roadmap (strict dependency chain: simulator -> contiguous
baseline -> paged allocator -> scheduler -> sharing -> model adapter ->
API -> benchmark study -> portfolio hardening), only the first two phases
are implemented. Everything after Phase 1 (`src/minipaged/{scheduler,
model, runtime, server}/`, and `kv/{block,pool,table,manager}.py`) is a
documented boundary placeholder — an empty module with a docstring
explaining what will live there and which phase adds it. Nothing is
silently stubbed; see each placeholder's own docstring.

**Phase 0 — Domain simulator.** `SimulationEngine` drives synthetic
requests through `waiting -> running -> completed` on a deterministic
`SimClock`, with no real model and no KV-capacity constraint. Every
lifecycle transition and decode step is recorded to an `EventLog` for
inspection.

**Phase 1 — Contiguous KV baseline.** `ContiguousKVEngine` subclasses the
Phase 0 engine and adds `ContiguousMemoryModel`: a naive allocator that
reserves `prompt_len + max_new_tokens` contiguous "token slots" per
sequence up front (the only size a contiguous allocator can safely commit
to, since it cannot grow a live reservation in place). `reserved - used`
is measurable waste — exactly the fragmentation that Phase 2's paged
allocator exists to eliminate. See `examples/phase1_fragmentation_demo.py`
for a runnable demonstration with numbers.

## Quickstart (WSL / Linux / macOS)

```bash
bash scripts/setup_env.sh          # creates .venv, installs pinned deps
source .venv/bin/activate
bash scripts/run_checks.sh         # ruff (lint + format) -> pyright -> pytest
```

Run the Phase 0/1 demo scripts directly:

```bash
python examples/phase0_lifecycle_demo.py
python examples/phase1_fragmentation_demo.py
```

## Repository layout

```
src/minipaged/
  requests/    Request + SequenceState (FR2, FR9)
  simulation/  SimClock, EventLog, trace generator, SimulationEngine (Phase 0)
  kv/          ContiguousMemoryModel + ContiguousKVEngine (Phase 1);
               block/pool/table/manager.py land in Phase 2
  sampling/    SamplingConfig (data-only until Phase 5)
  scheduler/ model/ server/ runtime/ metrics/   boundary placeholders (Phase 3+)
tests/{unit,property,integration}/
benchmarks/{traces,raw,plots}/  benchmarks/methodology.md
docs/architecture.md  docs/scheduler.md  docs/kv-memory.md  docs/invariants.md  docs/decisions/
examples/
```

## Core invariants

See `docs/invariants.md` for the full list and where each is tested. The
two that are load-bearing for Phase 0-1 today:

- Core invariant #6: completed/cancelled requests cannot re-enter
  scheduling (`Request`'s state machine — `tests/unit/test_request.py`).
- Core invariant #7: deterministic simulated traces reproduce identical
  allocation history under a fixed seed/config
  (`tests/property/test_determinism.py`,
  `tests/integration/test_trace_replay.py`).

## Documented simplifications (spec S1.10: "document all simplifications
explicitly — never hide them")

- **No tokenizer.** Prompt token ids are placeholder integers
  (`range(prompt_len)`); FR2's "tokenizer boundary" exists but is a no-op
  until Phase 5.
- **Prefill costs zero simulated ticks.** It is marked done at the same
  timestamp as admission; a real per-token prefill cost is Phase 3's job.
- **KV capacity is simulated, not physical.** Capacity is abstract "token
  slots", not real tensors sized by layers/heads/head_dim/dtype — see
  `docs/decisions/0002-contiguous-baseline-simulated-capacity.md`.
- **Admission policy is FCFS with reject-on-exhaustion**, not a real
  scheduler; Phase 3 replaces it.

## License

MIT.
