"""SchedulerEngine: Phase 3's continuous-batching scheduler (FR4, FR9).

Subclasses Phase 2's ``PagedKVEngine`` unchanged -- same ``BlockPool`` /
``BlockTable`` / lazy physical allocation, same ``_can_decode`` stall
mechanism -- and adds exactly two things on top, per D3/D4/D5 (see
``docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md``
for the full argument):

1. **A per-step token budget bounding how much *prefill* work (admission)
   happens in a single tick** (D3 default: token budget; D4 default:
   "decode-first with bounded prefill chunk per step" -- decode itself is
   never throttled by this budget, only new admission is, since decode
   work is O(1) per sequence per tick while prefill work scales with
   ``prompt_len``). This is what makes batch composition change
   dynamically tick to tick instead of admitting everything at once
   (continuous batching, FR4).

2. **A block-quantized worst-case admission ledger**, reusing Phase 1's
   ``ContiguousMemoryModel`` at block granularity (capacity =
   ``num_blocks``, not tokens) purely as an admission *gate* -- physical
   allocation stays lazy via the inherited ``PagedKVManager``, so paging's
   memory-efficiency benefit is not lost. This is what makes Phase 2's
   documented deadlock scenario
   (``docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md``)
   provably unreachable once a request is admitted: the ledger guarantees
   ``sum(watermark_i) <= num_blocks`` for every currently-running
   sequence's *worst-case* remaining footprint, and a sequence's actual
   physical usage is always ``<=`` its own watermark, so total physical
   demand can never exceed total supply. Core invariant #5 ("scheduler
   never admits work requiring more KV capacity than policy permits")
   becomes airtight at the scheduler level, not just the allocator level
   -- exactly what ``docs/scheduler.md`` previewed before Phase 2 landed.

Fairness (D5 default: FCFS) is strict: admission is attempted oldest-
arrival-first every tick, and the very first candidate this tick's
ledger/budget cannot currently satisfy stops the attempt loop for the
rest of this tick -- a later, smaller request is never allowed to jump
the queue ahead of an earlier, larger one just because it happens to fit
leftover budget/capacity. See the ADR for the starvation tradeoff this
implies and why it was chosen over a greedier bin-packing policy.
"""

from __future__ import annotations

from minipaged.kv.contiguous import ContiguousMemoryModel
from minipaged.kv.manager import PagedKVEngine
from minipaged.requests.request import Request, RequestState
from minipaged.requests.sequence import SequenceState
from minipaged.scheduler.decision import SchedulerDecision
from minipaged.simulation.clock import SimClock
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceRequestSpec

__all__ = ["SchedulerEngine"]


class SchedulerEngine(PagedKVEngine):
    """Phase 3 engine: continuous-batching scheduler over Phase 2's paged
    KV allocator. See module docstring for the two additions over
    ``PagedKVEngine``."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        token_budget: int = 64,
        clock: SimClock | None = None,
        decode_tick: float = 1.0,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks, block_size=block_size, clock=clock, decode_tick=decode_tick
        )
        if token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}")
        self.token_budget = token_budget
        # Block-quantized worst-case admission ledger -- see module
        # docstring. Units are BLOCKS, not tokens, despite reusing
        # ContiguousMemoryModel's token-shaped API.
        self._ledger = ContiguousMemoryModel(capacity_tokens=num_blocks)
        self.decisions: list[SchedulerDecision] = []

        self._prefill_budget_remaining = token_budget
        self._tick_admitted: list[str] = []
        self._tick_rejected: list[str] = []
        self._tick_cancelled: list[str] = []

    # -- per-tick bookkeeping (SimulationEngine hooks) -----------------------

    def _begin_tick(self) -> None:
        self._prefill_budget_remaining = self.token_budget
        self._tick_admitted = []
        self._tick_rejected = []
        self._tick_cancelled = []

    def _end_tick(self) -> None:
        now = self.clock.now()
        tick_events = [e for e in self.event_log if e.timestamp == now]
        decoded = tuple(e.request_id for e in tick_events if e.event_type == EventType.DECODE_STEP)
        stalled = tuple(
            e.request_id for e in tick_events if e.event_type == EventType.DECODE_STALLED
        )
        completed = tuple(e.request_id for e in tick_events if e.event_type == EventType.COMPLETED)
        kv_allocated_blocks = self.manager.pool.num_allocated
        # FR9's "expected KV growth": the admission ledger holds a worst-case
        # watermark per admitted/running sequence (blocks_needed_for(prompt_len
        # + max_new_tokens)), while physical allocation stays lazy -- the gap
        # between the two is exactly how many more blocks are already
        # committed, in the worst case, before any running sequence can
        # complete. See module docstring's `sum(watermark_i) <= num_blocks`
        # argument for why this can never go negative.
        expected_kv_growth_blocks = self._ledger.reserved_tokens - kv_allocated_blocks
        assert expected_kv_growth_blocks >= 0, (
            f"ledger reserved ({self._ledger.reserved_tokens}) fell below physically "
            f"allocated ({kv_allocated_blocks}) -- ledger/pool accounting mismatch"
        )
        self.decisions.append(
            SchedulerDecision(
                tick=now,
                admitted=tuple(self._tick_admitted),
                rejected=tuple(self._tick_rejected),
                cancelled=tuple(self._tick_cancelled),
                decoded=decoded,
                stalled=stalled,
                completed=completed,
                prefill_tokens_used=self.token_budget - self._prefill_budget_remaining,
                prefill_token_budget=self.token_budget,
                kv_free_blocks=self.manager.pool.num_free,
                kv_allocated_blocks=kv_allocated_blocks,
                expected_kv_growth_blocks=expected_kv_growth_blocks,
            )
        )

    # -- admission: strict FCFS, budget- and ledger-gated ---------------------

    def _admit_arrivals(self) -> set[str]:
        """Overrides ``PagedKVEngine``'s inherited ``_admit_arrivals``
        (itself inherited from ``SimulationEngine``): still FCFS by
        arrival time, but a soft rejection (ledger or budget insufficient
        *right now*) stops the attempt loop for the rest of this tick --
        see module docstring's fairness note. A hard cancellation never
        blocks the rest of the queue, matching every earlier phase's
        CapacityError-style handling."""
        now = self.clock.now()
        self._record_new_arrivals(now)
        admitted_this_tick: set[str] = set()
        for req_id in self._arrived_waiting_fcfs():
            req = self.waiting.get(req_id)
            if req is None:
                continue
            if self._try_admit(req, self._spec_by_id[req_id]):
                admitted_this_tick.add(req_id)
                self._tick_admitted.append(req_id)
                continue
            if req.is_terminal():
                self._tick_cancelled.append(req_id)
                continue  # unrecoverable -- never blocks the rest of the FCFS queue
            self._tick_rejected.append(req_id)
            break  # soft rejection: strict FCFS, stop rather than let a later request jump ahead
        return admitted_this_tick

    def _try_admit(self, request: Request, spec: TraceRequestSpec) -> bool:
        watermark = self.manager.blocks_needed_for(request.prompt_len + request.max_new_tokens)

        # Two distinct, both unrecoverable ("no amount of waiting helps")
        # conditions -- checked before anything soft/retryable, so an
        # impossible request is cancelled immediately rather than blocking
        # the rest of the strict-FCFS queue behind it forever (see
        # module docstring's fairness note and _admit_arrivals).
        cancel_reason: str | None = None
        if watermark > self.manager.pool.num_blocks:
            cancel_reason = "exceeds_total_kv_capacity"
        elif request.prompt_len > self.token_budget:
            cancel_reason = "prompt_exceeds_step_token_budget"

        if cancel_reason is not None:
            del self.waiting[request.request_id]
            request.transition_to(RequestState.CANCELLED)
            self.cancelled.append(request)
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                reason=cancel_reason,
                requested_blocks=watermark,
                total_blocks=self.manager.pool.num_blocks,
                prompt_len=request.prompt_len,
                token_budget=self.token_budget,
            )
            self.event_log.record(self.clock.now(), request.request_id, EventType.CANCELLED)
            return False

        if request.prompt_len > self._prefill_budget_remaining:
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                reason="prefill_budget_exhausted_this_tick",
                requested=request.prompt_len,
                remaining_budget=self._prefill_budget_remaining,
            )
            return False

        prompt_blocks = self.manager.blocks_needed_for(request.prompt_len)
        reserved = self._ledger.try_reserve(request.request_id, size=watermark, used=prompt_blocks)
        if not reserved:
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                reason="insufficient_free_kv_capacity",
                requested_blocks=watermark,
                free_blocks=self._ledger.free_tokens,
            )
            return False

        admitted = self.manager.admit(request.request_id, request.prompt_len)
        assert admitted, (
            "admission ledger reserved capacity but the paged manager could not admit -- "
            "ledger/pool accounting mismatch"
        )
        self._prefill_budget_remaining -= request.prompt_len
        self._admit(request, spec)
        return True

    # -- release: also frees the ledger's watermark reservation ---------------

    def _on_release(self, seq: SequenceState) -> None:
        super()._on_release(seq)
        self._ledger.release(seq.request_id)
