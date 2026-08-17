from __future__ import annotations

import pytest

from minipaged.requests.request import (
    InvalidStateTransition,
    Request,
    RequestState,
    is_legal_transition,
    next_request_id,
    reset_request_id_counter,
)
from minipaged.sampling.config import SamplingConfig


def _make_request(
    *,
    request_id: str = "req-1",
    prompt_token_ids: tuple[int, ...] = (1, 2, 3),
    max_new_tokens: int = 8,
    sampling_config: SamplingConfig | None = None,
    arrival_time: float = 0.0,
) -> Request:
    """Test builder with per-field defaults. Explicit keyword parameters
    (rather than **overrides unpacked into a heterogeneous dict) keep this
    type-checkable: a dict mixing str/tuple/int/float/SamplingConfig values
    defeats pyright's per-parameter narrowing when splatted into
    ``Request(**fields)``.
    """
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        sampling_config=sampling_config if sampling_config is not None else SamplingConfig(),
        arrival_time=arrival_time,
    )


# -- construction validation --------------------------------------------


def test_new_request_starts_waiting():
    req = _make_request()
    assert req.state == RequestState.WAITING
    assert not req.is_terminal()


def test_prompt_len_property():
    req = _make_request(prompt_token_ids=(1, 2, 3, 4, 5))
    assert req.prompt_len == 5


def test_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt_token_ids"):
        _make_request(prompt_token_ids=())


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_rejects_non_positive_max_new_tokens(max_new_tokens):
    with pytest.raises(ValueError, match="max_new_tokens"):
        _make_request(max_new_tokens=max_new_tokens)


def test_rejects_negative_arrival_time():
    with pytest.raises(ValueError, match="arrival_time"):
        _make_request(arrival_time=-1.0)


# -- state machine: legal transitions ------------------------------------


def test_waiting_to_running_is_legal():
    req = _make_request()
    req.transition_to(RequestState.RUNNING)
    assert req.state == RequestState.RUNNING


def test_waiting_to_cancelled_is_legal():
    req = _make_request()
    req.transition_to(RequestState.CANCELLED)
    assert req.state == RequestState.CANCELLED
    assert req.is_terminal()


def test_running_to_completed_is_legal():
    req = _make_request()
    req.transition_to(RequestState.RUNNING)
    req.transition_to(RequestState.COMPLETED)
    assert req.state == RequestState.COMPLETED
    assert req.is_terminal()


def test_running_to_cancelled_is_legal():
    req = _make_request()
    req.transition_to(RequestState.RUNNING)
    req.transition_to(RequestState.CANCELLED)
    assert req.state == RequestState.CANCELLED


# -- state machine: illegal transitions (core invariant #6) --------------


@pytest.mark.parametrize(
    "current,target",
    [
        (RequestState.WAITING, RequestState.COMPLETED),
        (RequestState.WAITING, RequestState.WAITING),
        (RequestState.RUNNING, RequestState.WAITING),
        (RequestState.RUNNING, RequestState.RUNNING),
        (RequestState.COMPLETED, RequestState.WAITING),
        (RequestState.COMPLETED, RequestState.RUNNING),
        (RequestState.COMPLETED, RequestState.CANCELLED),
        (RequestState.COMPLETED, RequestState.COMPLETED),
        (RequestState.CANCELLED, RequestState.WAITING),
        (RequestState.CANCELLED, RequestState.RUNNING),
        (RequestState.CANCELLED, RequestState.COMPLETED),
        (RequestState.CANCELLED, RequestState.CANCELLED),
    ],
)
def test_illegal_transitions_raise_and_do_not_mutate_state(current, target):
    req = _make_request()
    # Drive to `current` via legal transitions where necessary.
    if current == RequestState.RUNNING:
        req.transition_to(RequestState.RUNNING)
    elif current == RequestState.COMPLETED:
        req.transition_to(RequestState.RUNNING)
        req.transition_to(RequestState.COMPLETED)
    elif current == RequestState.CANCELLED:
        req.transition_to(RequestState.CANCELLED)
    assert req.state == current

    assert not is_legal_transition(current, target)
    with pytest.raises(InvalidStateTransition) as excinfo:
        req.transition_to(target)
    assert excinfo.value.current == current
    assert excinfo.value.target == target
    assert req.state == current  # rejected transition must not mutate state


def test_completed_requests_cannot_be_rescheduled():
    """Core invariant #6, direct statement: a COMPLETED request can never
    re-enter WAITING or RUNNING."""
    req = _make_request()
    req.transition_to(RequestState.RUNNING)
    req.transition_to(RequestState.COMPLETED)
    for target in (RequestState.WAITING, RequestState.RUNNING):
        with pytest.raises(InvalidStateTransition):
            req.transition_to(target)


def test_cancelled_requests_cannot_be_rescheduled():
    """Core invariant #6, direct statement: a CANCELLED request can never
    re-enter WAITING or RUNNING."""
    req = _make_request()
    req.transition_to(RequestState.CANCELLED)
    for target in (RequestState.WAITING, RequestState.RUNNING):
        with pytest.raises(InvalidStateTransition):
            req.transition_to(target)


# -- request id generation -------------------------------------------------


def test_next_request_id_is_monotonic_and_reproducible():
    reset_request_id_counter()
    assert next_request_id() == "req-1"
    assert next_request_id() == "req-2"
    reset_request_id_counter()
    assert next_request_id() == "req-1"


def test_reset_request_id_counter_accepts_custom_start():
    reset_request_id_counter(start=100)
    assert next_request_id() == "req-100"
