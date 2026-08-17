"""The Request object and its explicit lifecycle state machine.

Per FR2, a Request carries a request_id, prompt token ids, max_new_tokens,
a sampling config, an arrival timestamp, and an explicit lifecycle state.
This module also owns the state machine (which transitions are legal) so
"what can a request legally do next" lives in exactly one place instead of
being reimplemented by every caller (the simulation engine, later the
scheduler, later the HTTP layer).
"""

from __future__ import annotations

import dataclasses
import enum
import itertools

from minipaged.sampling.config import SamplingConfig


class RequestState(enum.Enum):
    """A request's position in its lifecycle.

    WAITING   -- created, not yet admitted to run (queued, or not yet
                 arrived in a simulated trace).
    RUNNING   -- admitted; prefill and/or decode work is in flight for it.
    COMPLETED -- finished normally (reached its target/max output length).
    CANCELLED -- terminated before completion (client cancel, admission
                 rejection, shutdown, ...).

    COMPLETED and CANCELLED are terminal: core invariant #6 (see
    docs/invariants.md) requires that a request in either state can never
    re-enter scheduling.
    """

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({RequestState.COMPLETED, RequestState.CANCELLED})

# The complete legal-transition table. Anything not listed here is illegal
# and `transition_to` raises InvalidStateTransition for it.
#   WAITING   -> RUNNING    : admitted by the engine/scheduler
#   WAITING   -> CANCELLED  : cancelled (or rejected at admission) before running
#   RUNNING   -> COMPLETED  : finished normally
#   RUNNING   -> CANCELLED  : cancelled mid-flight
_ALLOWED_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.WAITING: frozenset({RequestState.RUNNING, RequestState.CANCELLED}),
    RequestState.RUNNING: frozenset({RequestState.COMPLETED, RequestState.CANCELLED}),
    RequestState.COMPLETED: frozenset(),
    RequestState.CANCELLED: frozenset(),
}


class InvalidStateTransition(ValueError):
    """Raised when a Request is asked to move to an illegal next state."""

    def __init__(self, current: RequestState, target: RequestState) -> None:
        self.current = current
        self.target = target
        super().__init__(f"illegal transition: {current.value} -> {target.value}")


def is_legal_transition(current: RequestState, target: RequestState) -> bool:
    """Pure predicate shared by ``Request.transition_to`` and property tests."""
    return target in _ALLOWED_TRANSITIONS[current]


_id_counter = itertools.count(1)


def next_request_id() -> str:
    """Deterministic, monotonically increasing request id ("req-1", "req-2", ...).

    Used by the trace generator so request ids in a generated trace are
    reproducible run-to-run given a fresh counter (tests reset it via
    ``reset_request_id_counter``).
    """
    return f"req-{next(_id_counter)}"


def reset_request_id_counter(start: int = 1) -> None:
    """Test/simulation hook: reset the monotonic id counter to ``start``."""
    global _id_counter
    _id_counter = itertools.count(start)


@dataclasses.dataclass
class Request:
    """A single inference request moving through MiniPaged's lifecycle.

    Fields match FR2 exactly. ``state`` is mutable but only ever changes
    via ``transition_to``, which enforces ``_ALLOWED_TRANSITIONS`` -- there
    is no other way to move a Request between states in this codebase.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling_config: SamplingConfig
    arrival_time: float
    state: RequestState = RequestState.WAITING

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.arrival_time < 0:
            raise ValueError(f"arrival_time must be >= 0, got {self.arrival_time}")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def transition_to(self, target: RequestState) -> None:
        """Move this request to ``target``, or raise ``InvalidStateTransition``."""
        if not is_legal_transition(self.state, target):
            raise InvalidStateTransition(self.state, target)
        self.state = target

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
