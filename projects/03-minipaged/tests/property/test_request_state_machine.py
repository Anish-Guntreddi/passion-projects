"""Property test for core invariant #6: "completed/cancelled requests
cannot remain scheduled." ``tests/unit/test_request.py`` already checks
this exhaustively over the (small, fixed) set of state pairs via
parametrize; this generalizes it to arbitrary *sequences* of transition
attempts, per spec S1.9's "state transitions" test-plan item.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minipaged.requests.request import (
    TERMINAL_STATES,
    InvalidStateTransition,
    Request,
    RequestState,
    is_legal_transition,
)
from minipaged.sampling.config import SamplingConfig


def _new_request() -> Request:
    return Request(
        request_id="req-prop",
        prompt_token_ids=(1,),
        max_new_tokens=1,
        sampling_config=SamplingConfig(),
        arrival_time=0.0,
    )


@given(targets=st.lists(st.sampled_from(list(RequestState)), min_size=0, max_size=12))
def test_random_transition_sequences_never_escape_a_terminal_state_or_violate_the_table(
    targets: list[RequestState],
) -> None:
    """Whatever sequence of transition attempts is thrown at a fresh
    ``Request``: (a) every transition that succeeds matches
    ``is_legal_transition``'s table exactly, (b) a rejected transition
    never mutates ``state``, and (c) once ``state`` is terminal
    (COMPLETED/CANCELLED) no later transition in the sequence ever
    succeeds."""
    req = _new_request()
    for target in targets:
        prev_state = req.state
        was_terminal = prev_state in TERMINAL_STATES
        legal = is_legal_transition(prev_state, target)

        if legal:
            req.transition_to(target)
            assert req.state == target
            assert not was_terminal  # nothing legally leaves a terminal state
        else:
            with pytest.raises(InvalidStateTransition) as excinfo:
                req.transition_to(target)
            assert excinfo.value.current == prev_state
            assert excinfo.value.target == target
            assert req.state == prev_state  # rejected transition never mutates state
