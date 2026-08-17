"""Phase 1: contiguous (unpaged) KV-memory baseline.

Models the naive "reserve one contiguous span sized to the worst case"
allocation strategy that paged KV-cache designs (Phase 2) exist to fix.
Capacity is measured in abstract "token slots" (D2: simulate first, real
tensors later -- see
``docs/decisions/0002-contiguous-baseline-simulated-capacity.md`` and
``docs/kv-memory.md``). One slot conceptually holds one token's K/V
vectors across all layers/heads for this toy model; real per-token KV
footprint (layers x heads x head_dim x 2 x dtype_bytes) is out of scope
until a real model is wired in (Phase 5).

Each running sequence reserves ``prompt_len + max_new_tokens`` slots up
front -- the only size a contiguous allocator can safely commit to, since
it has no way to grow a reservation in place without risking collision
with a neighboring sequence's memory (no gaps between reservations, by
construction of this model). ``used`` tracks the sequence's actual
current KV length (``prompt_len + tokens_generated`` so far).
``reserved - used`` is memory the allocator is holding but nothing is
using yet -- exactly the fragmentation/waste that paged allocation
(Phase 2) is designed to eliminate.
"""

from __future__ import annotations

import dataclasses

from minipaged.requests.request import Request, RequestState
from minipaged.requests.sequence import SequenceState
from minipaged.simulation.clock import SimClock
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.events import EventType
from minipaged.simulation.trace import TraceRequestSpec


class CapacityError(ValueError):
    """A request's worst-case reservation exceeds *total* capacity.

    Distinct from a normal (temporary) admission-rejection: no amount of
    waiting for other sequences to complete will ever free enough
    contiguous space for a request this large, so the engine treats it as
    an unrecoverable rejection (D7 default: reject at admission) rather
    than queuing it forever.
    """


@dataclasses.dataclass
class _Reservation:
    request_id: str
    reserved: int
    used: int


@dataclasses.dataclass(frozen=True)
class ContiguousMemoryMetrics:
    """A point-in-time snapshot of the contiguous allocator's accounting.

    All fields are in "token slots" (see module docstring). ``waste`` is
    capacity reserved by an active sequence but not currently holding real
    KV data for that sequence -- capacity nobody else can use either,
    since a contiguous span can't be subdivided and lent out. ``free`` is
    capacity not reserved by anyone at all, and *can* be handed to a new
    arrival.
    """

    capacity: int
    reserved: int
    used: int
    free: int
    waste: int
    utilization: float  # used / capacity
    waste_ratio: float  # waste / capacity

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class ContiguousMemoryModel:
    """Fixed-capacity contiguous reservation pool. See module docstring."""

    def __init__(self, capacity_tokens: int) -> None:
        if capacity_tokens < 1:
            raise ValueError(f"capacity_tokens must be >= 1, got {capacity_tokens}")
        self.capacity_tokens = capacity_tokens
        self._reservations: dict[str, _Reservation] = {}

    @property
    def reserved_tokens(self) -> int:
        return sum(r.reserved for r in self._reservations.values())

    @property
    def used_tokens(self) -> int:
        return sum(r.used for r in self._reservations.values())

    @property
    def free_tokens(self) -> int:
        return self.capacity_tokens - self.reserved_tokens

    def try_reserve(self, request_id: str, size: int, used: int) -> bool:
        """Attempt to reserve ``size`` contiguous slots for ``request_id``.

        Returns ``False`` (does not raise) when free capacity is
        insufficient -- callers are expected to leave the request queued
        and retry on a later tick, per core invariant #5 ("scheduler
        never admits work requiring more KV capacity than policy
        permits"). Raises ``CapacityError`` only when ``size`` exceeds
        *total* capacity: see that class's docstring.
        """
        if request_id in self._reservations:
            raise ValueError(f"{request_id} already holds a reservation")
        if size > self.capacity_tokens:
            raise CapacityError(
                f"{request_id} requests {size} slots but total capacity is only "
                f"{self.capacity_tokens} -- can never be admitted"
            )
        if used > size:
            raise ValueError(f"{request_id}: used ({used}) exceeds requested size ({size})")
        if size > self.free_tokens:
            return False
        self._reservations[request_id] = _Reservation(request_id, reserved=size, used=used)
        return True

    def update_used(self, request_id: str, used: int) -> None:
        res = self._reservations[request_id]
        if used > res.reserved:
            raise ValueError(
                f"{request_id}: used ({used}) exceeds its own reservation ({res.reserved})"
            )
        res.used = used

    def release(self, request_id: str) -> int:
        """Free ``request_id``'s reservation; returns the number of slots freed."""
        res = self._reservations.pop(request_id)
        return res.reserved

    def is_reserved(self, request_id: str) -> bool:
        return request_id in self._reservations

    def snapshot(self) -> ContiguousMemoryMetrics:
        reserved = self.reserved_tokens
        used = self.used_tokens
        free = self.free_tokens
        waste = reserved - used
        return ContiguousMemoryMetrics(
            capacity=self.capacity_tokens,
            reserved=reserved,
            used=used,
            free=free,
            waste=waste,
            utilization=used / self.capacity_tokens,
            waste_ratio=waste / self.capacity_tokens,
        )


class ContiguousKVEngine(SimulationEngine):
    """Phase 1 engine: Phase 0's lifecycle simulator plus a
    ``ContiguousMemoryModel`` gating admission.

    Composition + three hook overrides (``_try_admit``,
    ``_on_decode_step``, ``_on_release``) -- no duplication of the base
    waiting/running/completed loop, which is exactly what lets
    Phase 2's paged allocator later reuse the same base class instead of
    reimplementing scheduling from scratch.
    """

    def __init__(
        self, capacity_tokens: int, clock: SimClock | None = None, decode_tick: float = 1.0
    ) -> None:
        super().__init__(clock=clock, decode_tick=decode_tick)
        self.memory = ContiguousMemoryModel(capacity_tokens)

    def _try_admit(self, request: Request, spec: TraceRequestSpec) -> bool:
        # Worst-case contiguous reservation: a contiguous allocator cannot
        # grow a live span, so it must commit to the sequence's maximum
        # possible length up front.
        reserve_size = request.prompt_len + request.max_new_tokens
        try:
            admitted = self.memory.try_reserve(
                request.request_id, reserve_size, used=request.prompt_len
            )
        except CapacityError:
            del self.waiting[request.request_id]
            request.transition_to(RequestState.CANCELLED)
            self.cancelled.append(request)
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                requested=reserve_size,
                capacity=self.memory.capacity_tokens,
                reason="exceeds_total_capacity",
            )
            self.event_log.record(self.clock.now(), request.request_id, EventType.CANCELLED)
            return False

        if not admitted:
            self.event_log.record(
                self.clock.now(),
                request.request_id,
                EventType.ADMISSION_REJECTED,
                requested=reserve_size,
                free=self.memory.free_tokens,
                reason="insufficient_free_capacity",
            )
            return False

        self._admit(request, spec)
        return True

    def _on_decode_step(self, seq: SequenceState) -> None:
        self.memory.update_used(seq.request_id, seq.current_len)

    def _on_release(self, seq: SequenceState) -> None:
        self.memory.release(seq.request_id)
