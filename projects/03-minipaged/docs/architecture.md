# Architecture

Status: covers what is implemented today (Phase 0 through Phase 3 of the
roadmap in `../../03-minipaged-spec.md`). Updated as later phases land.

## Layering

```
Request / SequenceState / SamplingConfig      (requests/, sampling/)
            |
SimClock / EventLog / TraceRequestSpec        (simulation/)
            |
SimulationEngine                              (simulation/engine.py)   -- Phase 0
            |
   subclassed by
            |
ContiguousKVEngine + ContiguousMemoryModel     (kv/contiguous.py)      -- Phase 1
            |
   (independently) subclassed by
            |
PhysicalBlock / BlockPool / BlockTable         (kv/{block,pool,table}.py)
PagedKVManager + PagedKVEngine                 (kv/manager.py)         -- Phase 2
            |
   subclassed by
            |
SchedulerEngine + SchedulerDecision            (scheduler/*.py)        -- Phase 3
```

`SimulationEngine` owns the entire `waiting -> running -> completed`
lifecycle loop and exposes override points for subclasses:
`_try_admit` (admission policy), `_can_decode` (may a sequence's decode
step actually run this tick -- added Phase 2), `_on_decode_step`
(per-tick hook for running sequences), `_on_release` (hook when a
sequence finishes), `_begin_tick`/`_end_tick` (per-tick bracketing hooks
-- added Phase 3 for scheduler bookkeeping). `ContiguousKVEngine` reuses
the base loop unchanged and only overrides `_try_admit`/
`_on_decode_step`/`_on_release` to gate admission on
`ContiguousMemoryModel` capacity. `PagedKVEngine` overrides the same
three plus `_can_decode` to gate admission *and* decode-time growth on
`PagedKVManager`. `SchedulerEngine` subclasses `PagedKVEngine` and
overrides `_try_admit`/`_admit_arrivals` again (stricter, ledger- and
budget-gated) plus `_begin_tick`/`_end_tick` (per-step bookkeeping and
`SchedulerDecision` assembly) -- `_can_decode`, `_on_decode_step`, and
`_on_release`'s block-freeing half are inherited from `PagedKVEngine`
unchanged. This composition pattern (subclass + hook overrides, never a
rewritten loop) held across all three extensions so far -- see
`simulation/engine.py`'s module docstring for the exact hook contract.

## Request lifecycle (Phase 0-3)

```
                 arrival_time <= now
  WAITING  ------------------------------->  (admission attempt)
     |                                             |
     |  hard-cancel: worst case can never fit      | admitted (soft-rejected
     |  total capacity / this tick's token         |  attempts stay WAITING,
     v  budget (Phase 1+)                           v  retried next tick)
 CANCELLED                                       RUNNING
                                                     |
                                          decode ticks until
                                          tokens_generated >= target_output_len
                                          (Phase 2+: a tick's decode attempt
                                           can instead stall -- seq stays
                                           RUNNING, unchanged, retried next
                                           tick -- see _can_decode below)
                                                     |
                                                     v
                                                 COMPLETED
```

- A request sits in the public `waiting` dict from the moment
  `load_trace()` returns it, whether or not its `arrival_time` has passed
  yet (`RequestState.WAITING` covers both "queued" and "not yet
  arrived"). An `ARRIVED` event is recorded exactly once, the tick its
  `arrival_time` is first `<= clock.now()`.
- Each `step()` runs `_begin_tick()` (Phase 3: reset per-step budget
  bookkeeping), then attempts admission for every arrived-but-still-
  waiting request (oldest arrival first, FCFS -- Phase 3 additionally
  stops at the first *soft* rejection this tick, see
  `docs/scheduler.md`), then runs one decode tick for every currently
  *running* sequence -- excluding requests admitted this same tick via an
  explicit skip set, so a just-admitted request never also receives a
  `DECODE_STEP` event in the tick it was admitted, and (Phase 2+) gating
  each sequence's decode on `_can_decode` first. `_end_tick()` runs last
  (Phase 3: assembles that tick's `SchedulerDecision`). Admission runs
  before decode specifically to preserve `EventLog` timestamp
  monotonicity -- see `engine.py`'s docstring for the full rationale.
- `Request.transition_to` is the only way a request's `state` changes,
  and only `SimulationEngine`/its subclasses call it -- this is what keeps
  core invariant #6 ("completed/cancelled requests cannot remain
  scheduled") checkable in one place.
- Phase 2+ adds a third *tick-level* (not state-machine-level) outcome for
  an already-RUNNING sequence: `_can_decode` can say "not yet" for one
  tick without any state transition at all -- `tokens_generated` and
  `current_len` stay exactly as they were, a `DECODE_STALLED` event is
  recorded instead of `DECODE_STEP`, and the same sequence is retried
  unconditionally next tick (every tick re-attempts every running
  sequence). This is not a new `RequestState` -- the request is still
  RUNNING throughout -- it is purely an allocator-level "not this tick"
  signal. See `docs/decisions/0004-...md` for why this exists (Phase 2)
  and `docs/decisions/0005-...md` for why it is provably unreachable once
  a request is admitted through `SchedulerEngine` (Phase 3).

## Event log

Every lifecycle transition and KV-accounting change is recorded to an
append-only `EventLog` (`simulation/events.py`) as a timestamped `Event`.
This is the primary mechanism for the "make memory/accounting behavior
inspectable in logs/tests" agent execution rule: nothing the engine does
is a silent side effect. `EventLog.as_dicts()` gives a JSON-serializable
view used by the example scripts and (eventually) benchmark raw-result
dumps.

## Determinism

`SimClock` never reads wall-clock time; every notion of "now" flows
through one explicit instance, advanced only by `step()`. Request ids are
assigned by an `itertools.count` **owned by each `SimulationEngine`
instance** (not a shared module-level counter) specifically so that
loading and replaying the same trace through two independent engines in
the same process produces byte-for-byte identical request ids and event
logs — core invariant #7. See `docs/invariants.md` and
`tests/property/test_determinism.py`.

## What is not here yet

`kv/` prefix-sharing/COW (Phase 4 -- `BlockPool.retain`/`release` already
exist per FR5 and are unit-tested, but nothing calls `retain` yet, since
no block has more than one owner until then), `model/` (Phase 5),
`server/` (Phase 6), `runtime/` composition root (Phase 5-6), and
scheduler-level `metrics/` (Phase 3+, currently folded into
`SchedulerDecision` and `PagedKVMetrics` rather than a separate package)
are all present as empty packages (or, for `scheduler/metrics/`, not yet
broken out into their own reporting layer) with a docstring naming the
phase that fills them in — see each module's own docstring rather than
duplicating that list here where it will go stale.
