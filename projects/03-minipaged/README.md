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

## Status: Phase 0-3 of 8

Per the spec's roadmap (strict dependency chain: simulator -> contiguous
baseline -> paged allocator -> scheduler -> sharing -> model adapter ->
API -> benchmark study -> portfolio hardening), the first four phases are
implemented. Everything after Phase 3 (`src/minipaged/{model, runtime,
server}/`, prefix sharing/COW in `kv/`) is a documented boundary
placeholder — an empty module with a docstring explaining what will live
there and which phase adds it. Nothing is silently stubbed; see each
placeholder's own docstring.

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

**Phase 2 — Paged KV allocator.** `PhysicalBlock` / `BlockPool` /
`BlockTable` (`kv/{block,pool,table}.py`) implement fixed-size-block
allocation with a free-list and refcount lifecycle (FR5). `PagedKVEngine`
admits on *current* need (`prompt_len`, not the worst case) and grows a
sequence's block table lazily, one block-boundary crossing at a time —
`examples/phase2_paged_kv_demo.py` replays the identical trace and total
capacity Phase 1's demo used and measures **31.5%** peak waste for paging
versus **63.7%** for the contiguous baseline. This admission policy has a
documented, deliberately-not-hidden limitation (it can deadlock under
adversarial concurrent admission — see
`docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md`)
that Phase 3 closes.

**Phase 3 — Scheduler.** `SchedulerEngine` subclasses `PagedKVEngine` and
adds continuous batching with a per-step token budget (decode always
prioritized over new admission) plus a block-quantized worst-case
admission ledger that makes Phase 2's deadlock scenario provably
unreachable — proven, not just avoided, by
`tests/property/test_scheduler_invariants.py` across 100 randomized
traces. `SchedulerDecision` (FR9) is recorded once per tick as the
"scheduling timeline" — see `examples/phase3_scheduler_demo.py` for a
runnable replay, and
`docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md` for
the full design.

## Quickstart (WSL / Linux / macOS)

```bash
bash scripts/setup_env.sh          # creates .venv, installs pinned deps
source .venv/bin/activate
bash scripts/run_checks.sh         # ruff (lint + format) -> pyright -> pytest
```

Run the demo scripts directly:

```bash
python examples/phase0_lifecycle_demo.py
python examples/phase1_fragmentation_demo.py
python examples/phase2_paged_kv_demo.py
python examples/phase3_scheduler_demo.py
```

## Repository layout

```
src/minipaged/
  requests/    Request + SequenceState (FR2, FR9)
  simulation/  SimClock, EventLog, trace generator, SimulationEngine (Phase 0)
  kv/          ContiguousMemoryModel + ContiguousKVEngine (Phase 1);
               PhysicalBlock/BlockPool/BlockTable + PagedKVManager/PagedKVEngine (Phase 2)
  scheduler/   SchedulerDecision + SchedulerEngine (Phase 3)
  sampling/    SamplingConfig (data-only until Phase 5)
  model/ server/ runtime/ metrics/   boundary placeholders (Phase 4+)
tests/{unit,property,integration}/
benchmarks/{traces,raw,plots}/  benchmarks/methodology.md
docs/architecture.md  docs/scheduler.md  docs/kv-memory.md  docs/invariants.md  docs/decisions/
examples/
```

## Core invariants

See `docs/invariants.md` for the full list and where each is tested.
Highlights as of Phase 3:

- Core invariant #1 (no block simultaneously free and referenced) and #3
  (a sequence's logical block count covers its KV length): literal,
  block-level tests as of Phase 2 (`tests/unit/test_pool.py`,
  `tests/unit/test_table.py`, `tests/property/test_paged_kv_invariants.py`).
- Core invariant #5 (scheduler never admits more than KV capacity
  permits): checked at the allocator level from Phase 1, and *proven* at
  the scheduler level from Phase 3
  (`tests/property/test_scheduler_invariants.py`) — Phase 2's
  `PagedKVEngine` alone does not fully satisfy it (documented, tested
  limitation; see `docs/decisions/0004-...md`).
- Core invariant #6: completed/cancelled requests cannot re-enter
  scheduling (`Request`'s state machine — `tests/unit/test_request.py`).
- Core invariant #7: deterministic simulated traces reproduce identical
  allocation history under a fixed seed/config, for every engine class
  (`tests/property/test_determinism.py`,
  `tests/integration/test_trace_replay.py`,
  `tests/integration/test_scheduler_replay.py`).

## Documented simplifications (spec S1.10: "document all simplifications
explicitly — never hide them")

- **No tokenizer.** Prompt token ids are placeholder integers
  (`range(prompt_len)`); FR2's "tokenizer boundary" exists but is a no-op
  until Phase 5.
- **Prefill costs zero simulated ticks.** It is marked done at the same
  timestamp as admission; a real per-token prefill cost (chunked prefill)
  remains a stretch goal, not implemented.
- **KV capacity is simulated, not physical.** Capacity is abstract "token
  slots"/blocks, not real tensors sized by layers/heads/head_dim/dtype —
  see `docs/decisions/0002-contiguous-baseline-simulated-capacity.md`.
- **Phase 2's paged admission can deadlock in isolation.** `PagedKVEngine`
  admits on current need, not worst case, which can (under adversarial
  concurrent admission) leave running sequences unable to ever complete —
  a deliberate, tested, documented Phase 2 limitation that Phase 3's
  scheduler closes with a KV-capacity-aware admission ledger. See
  `docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md`.
- **No preemption, no chunked prefill.** D7's MVP default (reject at
  admission, not swap/recompute) holds through Phase 3; a request is
  either admitted with a guarantee it can finish, or cancelled/deferred —
  never evicted mid-flight.

## License

MIT.
