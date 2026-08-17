"""Runtime event log: an append-only, queryable record of everything the
simulator does to a request, timestamped by SimClock.

This is the primary mechanism keeping scheduling/memory behavior
"inspectable in logs/tests" per the agent execution rules -- every
lifecycle transition and KV-accounting change becomes an ``Event`` instead
of a side effect nobody can observe after the fact.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterator


class EventType(enum.Enum):
    ARRIVED = "arrived"
    ADMITTED = "admitted"  # WAITING -> RUNNING
    PREFILL_DONE = "prefill_done"
    DECODE_STEP = "decode_step"
    COMPLETED = "completed"  # RUNNING -> COMPLETED
    CANCELLED = "cancelled"  # -> CANCELLED
    ADMISSION_REJECTED = "admission_rejected"  # Phase 1+: insufficient/impossible KV capacity
    DECODE_STALLED = "decode_stalled"  # Phase 2+: a running sequence wanted a decode step this
    # tick but the paged KV allocator could not grow its block table (no free physical
    # blocks) -- the sequence stays RUNNING, unchanged, and is retried next tick. Distinct
    # from ADMISSION_REJECTED (which applies to a still-WAITING request).


@dataclasses.dataclass(frozen=True)
class Event:
    timestamp: float
    request_id: str
    event_type: EventType
    detail: dict[str, object] = dataclasses.field(default_factory=dict)


class EventLog:
    """Append-only ordered event store with simple query helpers."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(
        self,
        timestamp: float,
        request_id: str,
        event_type: EventType,
        **detail: object,
    ) -> Event:
        event = Event(
            timestamp=timestamp, request_id=request_id, event_type=event_type, detail=detail
        )
        self._events.append(event)
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def for_request(self, request_id: str) -> list[Event]:
        return [e for e in self._events if e.request_id == request_id]

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]

    def is_monotonic(self) -> bool:
        """True iff recorded timestamps never decrease in append order."""
        return all(
            a.timestamp <= b.timestamp for a, b in zip(self._events, self._events[1:], strict=False)
        )

    def as_dicts(self) -> list[dict[str, object]]:
        """JSON-serializable view, used by benchmark/example artifact dumps."""
        return [
            {
                "timestamp": e.timestamp,
                "request_id": e.request_id,
                "event_type": e.event_type.value,
                **e.detail,
            }
            for e in self._events
        ]
