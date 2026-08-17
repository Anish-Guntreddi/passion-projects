"""Request/sequence domain types: the objects that move through MiniPaged.

``Request`` is the client-facing object (FR2); ``SequenceState`` is the
runtime's per-request generation progress (FR9). See their module
docstrings for the split rationale.
"""

from __future__ import annotations

from minipaged.requests.request import (
    TERMINAL_STATES,
    InvalidStateTransition,
    Request,
    RequestState,
    is_legal_transition,
    next_request_id,
    reset_request_id_counter,
)
from minipaged.requests.sequence import SequenceState

__all__ = [
    "TERMINAL_STATES",
    "InvalidStateTransition",
    "Request",
    "RequestState",
    "SequenceState",
    "is_legal_transition",
    "next_request_id",
    "reset_request_id_counter",
]
