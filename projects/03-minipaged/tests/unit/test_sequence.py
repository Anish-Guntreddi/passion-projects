from __future__ import annotations

import pytest

from minipaged.requests.request import Request
from minipaged.requests.sequence import SequenceState
from minipaged.sampling.config import SamplingConfig


def _make_request(prompt_len: int = 4, max_new_tokens: int = 8) -> Request:
    return Request(
        request_id="req-1",
        prompt_token_ids=tuple(range(prompt_len)),
        max_new_tokens=max_new_tokens,
        sampling_config=SamplingConfig(),
        arrival_time=0.0,
    )


def test_current_len_starts_at_prompt_len():
    req = _make_request(prompt_len=5)
    seq = SequenceState(request=req, target_output_len=3)
    assert seq.current_len == 5
    assert seq.tokens_generated == 0
    assert not seq.is_finished


def test_step_decode_increments_tokens_generated_and_current_len():
    req = _make_request(prompt_len=5)
    seq = SequenceState(request=req, target_output_len=3)
    seq.step_decode()
    assert seq.tokens_generated == 1
    assert seq.current_len == 6
    assert not seq.is_finished


def test_is_finished_true_exactly_at_target_output_len():
    req = _make_request(prompt_len=2, max_new_tokens=5)
    seq = SequenceState(request=req, target_output_len=2)
    assert not seq.is_finished
    seq.step_decode()
    assert not seq.is_finished
    seq.step_decode()
    assert seq.is_finished


def test_step_decode_after_finished_raises():
    req = _make_request(prompt_len=2, max_new_tokens=5)
    seq = SequenceState(request=req, target_output_len=1)
    seq.step_decode()
    assert seq.is_finished
    with pytest.raises(RuntimeError, match="after completion"):
        seq.step_decode()


@pytest.mark.parametrize("target_output_len", [0, -1])
def test_rejects_non_positive_target_output_len(target_output_len):
    req = _make_request()
    with pytest.raises(ValueError, match="target_output_len"):
        SequenceState(request=req, target_output_len=target_output_len)


def test_rejects_target_output_len_exceeding_max_new_tokens():
    req = _make_request(max_new_tokens=4)
    with pytest.raises(ValueError, match="target_output_len"):
        SequenceState(request=req, target_output_len=5)


def test_target_output_len_equal_to_max_new_tokens_is_allowed():
    req = _make_request(max_new_tokens=4)
    seq = SequenceState(request=req, target_output_len=4)
    assert seq.target_output_len == 4


def test_request_id_property_delegates_to_request():
    req = _make_request()
    seq = SequenceState(request=req, target_output_len=1)
    assert seq.request_id == req.request_id
