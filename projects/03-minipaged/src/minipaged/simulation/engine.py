"""SimulationEngine: the Phase 0 domain simulator.

Drives synthetic requests waiting -> running -> completed with NO real
model and (in this base class) NO KV-capacity constraint -- every arrived
request is admitted immediately. This is deliberately the entire Phase 0
exit criterion: "synthetic requests move waiting -> running -> completed
without a model."

Phase 1's ``ContiguousKVEngine`` (``minipaged.kv.contiguous``) subclasses
this and overrides three hook methods (``_try_admit``, ``_on_decode_step``,
``_on_release``) to gate admission on reserved contiguous KV capacity,
without duplicating the waiting/running/completed loop itself. Later
phases (scheduler, paged allocator) are expected to follow the same
pattern rather than growing this class unboundedly.

Simplification documented candidly (per spec S1.10, "document all
simplifications explicitly"): prefill is not yet modeled as a distinct
timed step. It is marked done at the same simulated timestamp as
admission. A real per-token prefill cost, and interleaving prefill work
with decode work across a running batch, is Phase 3's job (continuous
batching / scheduler); Phase 0-1 only need "admitted" vs "still
generating" to be observable, which the ``PREFILL_DONE`` event still
records for downstream inspection even though it costs zero simulated
ticks today.

Two more things worth stating plainly since they are easy to get wrong:

* ``waiting`` holds *every* request from the moment ``load_trace`` returns
  it, including ones whose ``arrival_time`` is still in the future --
  ``RequestState.WAITING``'s own docstring is "queued, or not yet arrived
  in a simulated trace", so both cases live in the same public dict. What
  changes at arrival time is only that an ``ARRIVED`` event is recorded
  and the request becomes eligible for ``_try_admit``.
* A request admitted this tick does not also receive a ``DECODE_STEP`` in
  the same tick -- prefill (instant, per the simplification above) and a
  sequence's first decode are distinct events separated by at least one
  tick, matching how a real scheduler could not possibly run a decode
  step for a sequence before its KV cache exists. ``step()`` still runs
  admission *before* decode (not after) so that event timestamps stay
  monotonic: ``ARRIVED``/``ADMITTED`` events use each request's own
  ``arrival_time``/``now``, both ``<=`` the current tick's ``now``, and
  must therefore be appended before this tick's ``DECODE_STEP`` events
  (which use ``now`` directly) -- so newly admitted sequences are instead
  explicitly excluded from this tick's decode pass via a skip set, rather
  than achieved by reordering the two phases.

Phase 2/3 extension points (added alongside the paged allocator and
scheduler, not by rewriting this loop -- see ``kv/manager.py`` and
``scheduler/scheduler.py``):

* ``_can_decode(seq)`` -- checked immediately before a sequence's decode
  step actually runs. Base behavior (Phase 0/1): always ``True`` -- a
  contiguous reservation already committed the sequence's entire
  worst-case span at admission, so a decode step can never fail for lack
  of space. Phase 2's ``PagedKVEngine`` overrides this to ask the paged
  allocator whether it can grow the sequence's block table by one more
  token; if not, the sequence is left running and unchanged this tick (a
  ``DECODE_STALLED`` event is recorded instead of ``DECODE_STEP``) and is
  retried next tick -- this is what lets "KV exhaustion mid-decode" be a
  safe, observable outcome instead of a crash or silent corruption.
* ``_begin_tick()`` / ``_end_tick()`` -- no-op hooks bracketing each
  ``step()`` call, before admission runs and after decode runs
  respectively. Phase 3's ``SchedulerEngine`` uses ``_begin_tick`` to
  reset its per-step token budget and per-tick bookkeeping, and
  ``_end_tick`` to assemble that tick's ``SchedulerDecision`` (the
  scheduling-timeline entry) once both admission and decode outcomes for
  the tick are known.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from minipaged.requests.request import Request, RequestState
from minipaged.requests.sequence import SequenceState
from minipaged.sampling.config import SamplingConfig
from minipaged.simulation.clock import SimClock
from minipaged.simulation.events import EventLog, EventType
from minipaged.simulation.trace import TraceRequestSpec


class SimulationEngine:
    """Phase 0 engine: request lifecycle simulation without a model.

    Public, test-facing state: ``waiting`` / ``running`` (keyed by
    request_id) and ``completed`` / ``cancelled`` (lists), plus
    ``event_log`` and ``clock``. Nothing outside this class ever calls
    ``Request.transition_to`` directly -- the engine is the single place
    lifecycle transitions happen, which is what keeps them inspectable and
    testable as a unit.
    """

    def __init__(self, clock: SimClock | None = None, decode_tick: float = 1.0) -> None:
        if decode_tick <= 0:
            raise ValueError(f"decode_tick must be > 0, got {decode_tick}")
        self.clock = clock if clock is not None else SimClock()
        self.event_log = EventLog()
        self.decode_tick = decode_tick

        # Instance-scoped, not the shared module-level counter in
        # ``minipaged.requests.request`` -- keeps request-id assignment (and
        # therefore the whole event log, per core invariant #7) deterministic
        # for *this engine's* trace regardless of how many other engines or
        # traces have run earlier in the same process.
        self._id_counter = itertools.count(1)
        self._spec_by_id: dict[str, TraceRequestSpec] = {}
        self._arrived_ids: set[str] = set()

        self.waiting: dict[str, Request] = {}
        self.running: dict[str, SequenceState] = {}
        self.completed: list[SequenceState] = []
        self.cancelled: list[Request] = []

    # -- trace loading --------------------------------------------------

    def _next_request_id(self) -> str:
        return f"req-{next(self._id_counter)}"

    def load_trace(self, specs: Sequence[TraceRequestSpec]) -> list[Request]:
        """Materialize ``TraceRequestSpec``s into ``Request``s and place
        them directly into ``waiting`` -- including ones that have not
        arrived yet, per this module's docstring. Prompt token ids are
        placeholder integers (``range(prompt_len)``) since Phase 0-1 has no
        tokenizer -- FR2's "tokenizer boundary" is intentionally a no-op
        boundary here, documented rather than faked as real token ids.
        """
        requests = []
        for spec in specs:
            req = Request(
                request_id=self._next_request_id(),
                prompt_token_ids=tuple(range(spec.prompt_len)),
                max_new_tokens=spec.max_new_tokens,
                sampling_config=SamplingConfig(),
                arrival_time=spec.arrival_time,
            )
            self._spec_by_id[req.request_id] = spec
            requests.append(req)
            self.waiting[req.request_id] = req
        return requests

    # -- one simulated tick ----------------------------------------------

    def step(self) -> None:
        """Advance the simulation by exactly one tick: admit this tick's
        arrivals (and retry any still-waiting requests), then decode every
        *already* running sequence by one step -- sequences admitted this
        tick are excluded from this tick's decode pass. See the module
        docstring for why admission runs first (event-log monotonicity)
        while still not decoding newly admitted sequences.

        ``_begin_tick``/``_end_tick`` bracket the whole thing as no-op
        hooks for subclasses (Phase 3's scheduler) that need per-tick
        bookkeeping -- see the module docstring."""
        self.clock.advance(self.decode_tick)
        self._begin_tick()
        just_admitted = self._admit_arrivals()
        self._decode_step(skip=just_admitted)
        self._end_tick()

    def run_to_completion(self, max_ticks: int = 1_000_000) -> None:
        """Repeatedly ``step()`` until every loaded request has reached a
        terminal state (or ``max_ticks`` is exceeded, which raises -- a
        safety net against an infinite loop from a misconfigured trace or
        subclass)."""
        ticks = 0
        while self.waiting or self.running:
            if ticks >= max_ticks:
                raise RuntimeError(f"run_to_completion exceeded max_ticks={max_ticks}")
            self.step()
            ticks += 1

    def is_complete(self) -> bool:
        return not self.waiting and not self.running

    # -- internals (override points for subclasses) ----------------------

    def _record_new_arrivals(self, now: float) -> None:
        """Record an ``ARRIVED`` event, exactly once, for every waiting
        request whose ``arrival_time`` has now passed. Split out of
        ``_admit_arrivals`` so Phase 3's ``SchedulerEngine`` can reuse it
        verbatim while overriding the admission-attempt loop itself."""
        for req_id, req in self.waiting.items():
            if req.arrival_time <= now and req_id not in self._arrived_ids:
                self._arrived_ids.add(req_id)
                self.event_log.record(req.arrival_time, req_id, EventType.ARRIVED)

    def _arrived_waiting_fcfs(self) -> list[str]:
        """Request ids currently in ``waiting`` that have arrived, oldest
        arrival first (FCFS, D5's default fairness policy)."""
        return sorted(
            (rid for rid in self.waiting if rid in self._arrived_ids),
            key=lambda rid: self.waiting[rid].arrival_time,
        )

    def _admit_arrivals(self) -> set[str]:
        """Record this tick's ``ARRIVED`` events and retry admission for
        everything that has arrived. Returns the set of request ids
        admitted *during this call* so ``step()`` can exclude them from
        this tick's decode pass."""
        now = self.clock.now()
        self._record_new_arrivals(now)

        # Retry admission for everything that has arrived, oldest arrival
        # first (FCFS). A base-class admission never fails, but Phase 1's
        # capacity check can leave requests queued across ticks, so this
        # must re-attempt previously-stuck requests too, not just new
        # arrivals. Requests that have not arrived yet are left untouched
        # in ``waiting``.
        admitted_this_tick: set[str] = set()
        for req_id in self._arrived_waiting_fcfs():
            req = self.waiting.get(req_id)
            if req is None:
                continue  # admitted or rejected earlier in this same pass
            if self._try_admit(req, self._spec_by_id[req_id]):
                admitted_this_tick.add(req_id)
        return admitted_this_tick

    def _try_admit(self, request: Request, spec: TraceRequestSpec) -> bool:
        """Admission policy. Base Phase 0 behavior: unconditional
        admission (no KV-capacity model). Returns True iff admitted."""
        self._admit(request, spec)
        return True

    def _admit(self, request: Request, spec: TraceRequestSpec) -> None:
        del self.waiting[request.request_id]
        request.transition_to(RequestState.RUNNING)
        seq = SequenceState(
            request=request,
            target_output_len=spec.target_output_len,
            started_at=self.clock.now(),
        )
        seq.prefill_done = True  # see module docstring's documented simplification
        self.running[request.request_id] = seq
        self.event_log.record(self.clock.now(), request.request_id, EventType.ADMITTED)
        self.event_log.record(self.clock.now(), request.request_id, EventType.PREFILL_DONE)

    def _decode_step(self, skip: AbstractSet[str] = frozenset()) -> None:
        finished: list[str] = []
        for req_id, seq in self.running.items():
            if req_id in skip:
                continue
            if not self._can_decode(seq):
                # KV exhaustion this tick (Phase 2+ only -- base default is
                # always True): seq is left exactly as it was and retried
                # next tick, not crashed and not silently advanced without
                # backing KV space. See module docstring.
                self.event_log.record(self.clock.now(), req_id, EventType.DECODE_STALLED)
                continue
            seq.step_decode()
            self._on_decode_step(seq)
            self.event_log.record(
                self.clock.now(),
                req_id,
                EventType.DECODE_STEP,
                tokens_generated=seq.tokens_generated,
            )
            if seq.is_finished:
                finished.append(req_id)

        for req_id in finished:
            seq = self.running.pop(req_id)
            seq.completed_at = self.clock.now()
            seq.request.transition_to(RequestState.COMPLETED)
            self.completed.append(seq)
            self.event_log.record(self.clock.now(), req_id, EventType.COMPLETED)
            self._on_release(seq)

    def _can_decode(self, seq: SequenceState) -> bool:
        """Hook: may ``seq`` receive a decode step this tick? Base default
        (Phase 0/1): always True. Phase 2's ``PagedKVEngine`` overrides
        this to check whether the paged allocator can grow the sequence's
        block table by one more token; see module docstring."""
        return True

    def _on_decode_step(self, seq: SequenceState) -> None:
        """Hook for subclasses (Phase 1: update reserved-memory "used" accounting;
        Phase 2: physically grow the sequence's paged block table). Only called
        when ``_can_decode`` returned True and ``seq.step_decode()`` already ran."""

    def _on_release(self, seq: SequenceState) -> None:
        """Hook for subclasses (Phase 1: free reserved contiguous capacity;
        Phase 2: free the sequence's physical blocks back to the pool)."""

    def _begin_tick(self) -> None:
        """Hook: per-tick setup, run once at the start of ``step()`` before
        admission or decode. No-op by default; Phase 3's ``SchedulerEngine``
        resets its per-step token budget and decision-tracking state here."""

    def _end_tick(self) -> None:
        """Hook: per-tick teardown, run once at the end of ``step()`` after
        both admission and decode have run. No-op by default; Phase 3's
        ``SchedulerEngine`` assembles this tick's ``SchedulerDecision``
        (the scheduling-timeline entry) here, once outcomes are known."""
