# Architecture

Status: covers what is implemented today (Phase 0 + Phase 1 of the
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
```

`SimulationEngine` owns the entire `waiting -> running -> completed`
lifecycle loop and exposes three override points for subclasses:
`_try_admit` (admission policy), `_on_decode_step` (per-tick hook for
running sequences), `_on_release` (hook when a sequence finishes).
`ContiguousKVEngine` reuses the base loop unchanged and only overrides
those three hooks to gate admission on `ContiguousMemoryModel` capacity.
This composition pattern (subclass + hook overrides, not a rewritten
loop) is intentional: Phase 2's paged allocator and Phase 3's scheduler
are expected to follow the same pattern rather than growing
`SimulationEngine` unboundedly. See `simulation/engine.py`'s module
docstring for the exact hook contract.

## Request lifecycle (Phase 0-1)

```
                 arrival_time <= now
  WAITING  ------------------------------->  (admission attempt)
     |                                             |
     |  CapacityError / insufficient capacity      | admitted
     v  (Phase 1 only)                              v
 CANCELLED                                       RUNNING
                                                     |
                                          decode ticks until
                                          tokens_generated >= target_output_len
                                                     |
                                                     v
                                                 COMPLETED
```

- A request sits in the public `waiting` dict from the moment
  `load_trace()` returns it, whether or not its `arrival_time` has passed
  yet (`RequestState.WAITING` covers both "queued" and "not yet
  arrived"). An `ARRIVED` event is recorded exactly once, the tick its
  `arrival_time` is first `<= clock.now()`.
- Each `step()` first attempts admission for every arrived-but-still-waiting
  request (oldest arrival first, FCFS), then runs one decode tick for every
  currently *running* sequence — excluding requests admitted this same tick
  via an explicit skip set, so a just-admitted request never also receives a
  `DECODE_STEP` event in the tick it was admitted. Admission runs before
  decode specifically to preserve `EventLog` timestamp monotonicity — see
  `engine.py`'s docstring for the full rationale.
- `Request.transition_to` is the only way a request's `state` changes,
  and only `SimulationEngine`/its subclasses call it — this is what keeps
  core invariant #6 ("completed/cancelled requests cannot remain
  scheduled") checkable in one place.

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

`kv/{block,pool,table,manager}.py` (Phase 2 paged allocator),
`scheduler/` (Phase 3), `kv/` prefix-sharing/COW (Phase 4), `model/`
(Phase 5), `server/` (Phase 6), `runtime/` composition root (Phase 5-6),
and scheduler-level `metrics/` (Phase 3+) are all present as empty
packages with a docstring naming the phase that fills them in — see each
module's own docstring rather than duplicating that list here where it
will go stale.
