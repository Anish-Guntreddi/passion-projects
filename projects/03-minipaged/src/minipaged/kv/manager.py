"""PagedKVManager + PagedKVMetrics + PagedKVEngine: Phase 2's paged KV
subsystem tied into the simulation engine, mirroring
``minipaged.kv.contiguous`` (Phase 1) exactly at the seam that matters:
same ``SimulationEngine`` base class, same three-hook composition
pattern, so the *same trace* can be replayed through either allocator and
compared -- literally the Phase 2 exit criterion ("same trace runs paged
and produces utilization statistics").

``PagedKVManager`` owns one ``BlockPool`` (``pool.py``) and one
``BlockTable`` (``table.py``) per admitted request; it is the thing
``PagedKVEngine``'s hooks call into, exactly as ``ContiguousKVEngine``
calls into ``ContiguousMemoryModel``.

Admission policy (deliberate contrast with Phase 1, and the actual point
of paging): a request is admitted based on its *current* need
(``prompt_len``, rounded up to whole blocks), never its worst case
(``prompt_len + max_new_tokens``, what Phase 1 must commit to up front).
Physical blocks are then grown lazily, one ``block_size``-token boundary
crossing at a time, as the sequence actually generates.

This has a real, documented cost: since admission does not reserve a
sequence's worst-case future growth, it is possible for multiple
concurrently-admitted sequences to simultaneously need a new block when
the pool has none free -- a ``BlockPoolExhausted`` mid-decode. This
manager never lets that propagate as an exception (``can_grow_to`` is a
non-mutating pre-check the engine uses via ``_can_decode``, so a decode
that cannot proceed simply does not happen this tick --
``EventType.DECODE_STALLED``, retried next tick). But *repeated,
permanent* stalling across every running sequence simultaneously -- true
deadlock, no sequence ever able to progress or complete and therefore
never releasing a block either -- is reachable under adversarial
concurrent admission with this policy alone. See
``docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md``
for the full argument and a constructed example
(``tests/unit/test_manager.py::test_two_sequences_can_deadlock_under_optimistic_admission``),
and ``docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md``
for how Phase 3's scheduler closes this gap with KV-capacity-aware
admission control, matching docs/scheduler.md's preview of that split.
"""

from __future__ import annotations

import dataclasses

from minipaged.kv.pool import BlockPool
from minipaged.kv.table import BlockTable, blocks_needed_for
from minipaged.requests.request import Request, RequestState
from minipaged.requests.sequence import SequenceState
from minipaged.simulation.clock import SimClock
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceRequestSpec

__all__ = [
    "PagedKVEngine",
    "PagedKVManager",
    "PagedKVMetrics",
]


@dataclasses.dataclass(frozen=True)
class PagedKVMetrics:
    """A point-in-time snapshot of the paged allocator's accounting.

    Mirrors ``ContiguousMemoryMetrics`` field-for-field where the concepts
    correspond (``reserved``/``used``/``waste``/``utilization``), plus
    ``num_blocks``/``block_size``/``free_blocks``/``allocated_blocks``
    which only make sense for a block-granular allocator. ``reserved_tokens``
    is *block-quantized* (``allocated_blocks * block_size``), unlike
    Phase 1's ``reserved`` (exact worst-case token count) -- comparing the
    two ``waste``/``utilization`` numbers on the same trace and capacity is
    exactly what demonstrates paging's benefit (see
    ``examples/phase2_paged_kv_demo.py``).
    """

    num_blocks: int
    block_size: int
    free_blocks: int
    allocated_blocks: int
    capacity_tokens: int  # num_blocks * block_size
    reserved_tokens: int  # allocated_blocks * block_size -- physically committed
    used_tokens: int  # sum of current_len across active sequences (exact, not rounded)
    waste: int  # reserved_tokens - used_tokens: block-rounding overhead only
    utilization: float  # used_tokens / capacity_tokens
    waste_ratio: float  # waste / capacity_tokens

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class PagedKVManager:
    """Owns the ``BlockPool`` and one ``BlockTable`` per admitted request."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        self.pool = BlockPool(num_blocks, block_size)
        self.block_size = block_size
        self._tables: dict[str, BlockTable] = {}
        self._used_tokens: dict[str, int] = {}  # request_id -> current_len, for metrics only

    def blocks_needed_for(self, num_tokens: int) -> int:
        return blocks_needed_for(num_tokens, self.block_size)

    def is_admitted(self, request_id: str) -> bool:
        return request_id in self._tables

    def capacity_tokens_for(self, request_id: str) -> int:
        """How many tokens ``request_id``'s currently-allocated blocks can
        hold (its ``BlockTable.capacity_tokens``) -- always ``>=`` the
        length it was last grown/admitted to (core invariant #3)."""
        return self._tables[request_id].capacity_tokens

    def can_admit(self, prompt_len: int) -> bool:
        """Would ``admit`` currently succeed for a sequence with this
        prompt length? Pure check, no side effects."""
        return self.blocks_needed_for(prompt_len) <= self.pool.num_free

    def admit(self, request_id: str, prompt_len: int) -> bool:
        """Allocate blocks to cover ``prompt_len`` tokens for a newly
        admitted sequence. Returns False (no partial allocation, no
        mutation) if currently-free blocks are insufficient -- the caller
        leaves the request queued and retries later, same soft-rejection
        contract as ``ContiguousMemoryModel.try_reserve``. Raises
        ``ValueError`` if ``request_id`` is already admitted."""
        if request_id in self._tables:
            raise ValueError(f"{request_id} is already admitted")
        if not self.can_admit(prompt_len):
            return False
        table = BlockTable(self.block_size, self.pool)
        table.grow_to(prompt_len)
        self._tables[request_id] = table
        self._used_tokens[request_id] = prompt_len
        return True

    def can_grow_to(self, request_id: str, num_tokens: int) -> bool:
        """Would ``grow`` currently succeed in extending ``request_id``'s
        block table to cover ``num_tokens`` tokens? Pure check, no side
        effects -- this is what ``PagedKVEngine._can_decode`` calls before
        ``step_decode()`` runs, so a decode that cannot get KV space is
        never attempted in the first place (see module docstring)."""
        table = self._tables[request_id]
        delta = self.blocks_needed_for(num_tokens) - table.num_blocks
        return delta <= self.pool.num_free

    def grow(self, request_id: str, num_tokens: int) -> int:
        """Grow ``request_id``'s block table to cover ``num_tokens``
        tokens. Returns the number of newly allocated blocks (0 if
        already sufficient). Raises ``BlockPoolExhausted`` if the pool
        cannot satisfy it -- callers that want the safe, non-raising path
        should check ``can_grow_to`` first (which ``PagedKVEngine``
        always does)."""
        table = self._tables[request_id]
        grown = table.grow_to(num_tokens)
        self._used_tokens[request_id] = num_tokens
        return grown

    def release(self, request_id: str) -> int:
        """Free every block ``request_id`` holds back to the pool. Returns
        the number of blocks freed."""
        table = self._tables.pop(request_id)
        self._used_tokens.pop(request_id, None)
        return table.free_all()

    def snapshot(self) -> PagedKVMetrics:
        allocated = self.pool.num_allocated
        capacity_tokens = self.pool.num_blocks * self.block_size
        reserved_tokens = allocated * self.block_size
        used_tokens = sum(self._used_tokens.values())
        waste = reserved_tokens - used_tokens
        return PagedKVMetrics(
            num_blocks=self.pool.num_blocks,
            block_size=self.block_size,
            free_blocks=self.pool.num_free,
            allocated_blocks=allocated,
            capacity_tokens=capacity_tokens,
            reserved_tokens=reserved_tokens,
            used_tokens=used_tokens,
            waste=waste,
            utilization=used_tokens / capacity_tokens,
            waste_ratio=waste / capacity_tokens,
        )


class PagedKVEngine(SimulationEngine):
    """Phase 2 engine: Phase 0's lifecycle simulator plus a
    ``PagedKVManager`` gating admission and decode-time growth, mirroring
    ``ContiguousKVEngine`` (Phase 1) at every hook.

    Two outcomes at admission, matching ``ContiguousKVEngine`` exactly so
    the two engines' behavior is directly comparable on the same trace:

    * The request's worst-case footprint (``prompt_len + max_new_tokens``,
      rounded up to whole blocks) exceeds *total* pool capacity -> no
      amount of waiting ever admits it -> immediate ``CANCELLED`` (D7
      default: reject at admission).
    * Otherwise, admission needs only ``prompt_len`` blocks (not the
      worst case -- see module docstring) -> soft rejection (stays
      queued, retried next tick) if currently-free blocks are
      insufficient, admitted otherwise.

    ``_can_decode`` additionally gates each decode tick on paged growth
    succeeding -- see module docstring for the documented tradeoff this
    introduces (decode-time stalls are possible; Phase 3 closes the gap).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        clock: SimClock | None = None,
        decode_tick: float = 1.0,
    ) -> None:
        super().__init__(clock=clock, decode_tick=decode_tick)
        self.manager = PagedKVManager(num_blocks, block_size)

    def _try_admit(self, request: Request, spec: TraceRequestSpec) -> bool:
        worst_case_blocks = self.manager.blocks_needed_for(
            request.prompt_len + request.max_new_tokens
        )
        if worst_case_blocks > self.manager.pool.num_blocks:
            del self.waiting[request.request_id]
            request.transition_to(RequestState.CANCELLED)
            self.cancelled.append(request)
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                requested_blocks=worst_case_blocks,
                total_blocks=self.manager.pool.num_blocks,
                reason="exceeds_total_kv_capacity",
            )
            self.event_log.record(self.clock.now(), request.request_id, EventType.CANCELLED)
            return False

        if not self.manager.admit(request.request_id, request.prompt_len):
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                requested_blocks=self.manager.blocks_needed_for(request.prompt_len),
                free_blocks=self.manager.pool.num_free,
                reason="insufficient_free_blocks",
            )
            return False

        self._admit(request, spec)
        return True

    def _can_decode(self, seq: SequenceState) -> bool:
        return self.manager.can_grow_to(seq.request_id, seq.current_len + 1)

    def _on_decode_step(self, seq: SequenceState) -> None:
        self.manager.grow(seq.request_id, seq.current_len)

    def _on_release(self, seq: SequenceState) -> None:
        self.manager.release(seq.request_id)
