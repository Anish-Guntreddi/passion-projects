from __future__ import annotations

import pytest

from minipaged.simulation.trace import TraceConfig, TraceRequestSpec, generate_trace

# -- TraceRequestSpec validation ------------------------------------------


def test_valid_spec_constructs():
    spec = TraceRequestSpec(
        prompt_len=10, max_new_tokens=20, target_output_len=15, arrival_time=1.0
    )
    assert spec.prompt_len == 10


@pytest.mark.parametrize("prompt_len", [0, -1])
def test_rejects_non_positive_prompt_len(prompt_len):
    with pytest.raises(ValueError, match="prompt_len"):
        TraceRequestSpec(
            prompt_len=prompt_len, max_new_tokens=5, target_output_len=1, arrival_time=0.0
        )


@pytest.mark.parametrize("target_output_len", [0, 21])
def test_rejects_target_output_len_out_of_bounds(target_output_len):
    with pytest.raises(ValueError, match="target_output_len"):
        TraceRequestSpec(
            prompt_len=5, max_new_tokens=20, target_output_len=target_output_len, arrival_time=0.0
        )


def test_rejects_negative_arrival_time():
    with pytest.raises(ValueError, match="arrival_time"):
        TraceRequestSpec(prompt_len=5, max_new_tokens=5, target_output_len=1, arrival_time=-1.0)


# -- TraceConfig validation ------------------------------------------------


def test_default_config_is_valid():
    TraceConfig()


@pytest.mark.parametrize("num_requests", [0, -1])
def test_rejects_non_positive_num_requests(num_requests):
    with pytest.raises(ValueError, match="num_requests"):
        TraceConfig(num_requests=num_requests)


def test_rejects_negative_mean_interarrival():
    with pytest.raises(ValueError, match="mean_interarrival"):
        TraceConfig(mean_interarrival=-1.0)


def test_zero_mean_interarrival_is_allowed():
    # All requests arrive at t=0 -- a valid "burst arrival" configuration.
    TraceConfig(mean_interarrival=0.0)


def test_rejects_inverted_prompt_len_range():
    with pytest.raises(ValueError, match="prompt_len_range"):
        TraceConfig(prompt_len_range=(10, 5))


def test_rejects_inverted_max_new_tokens_range():
    with pytest.raises(ValueError, match="max_new_tokens_range"):
        TraceConfig(max_new_tokens_range=(10, 5))


@pytest.mark.parametrize("bad_range", [(0.0, 0.5), (0.5, 1.5), (0.8, 0.2)])
def test_rejects_invalid_completion_fraction_range(bad_range):
    with pytest.raises(ValueError, match="completion_fraction_range"):
        TraceConfig(completion_fraction_range=bad_range)


# -- generate_trace ----------------------------------------------------------


def test_generate_trace_produces_requested_count():
    trace = generate_trace(TraceConfig(num_requests=17), seed=1)
    assert len(trace) == 17


def test_generate_trace_is_deterministic_given_seed():
    config = TraceConfig(num_requests=25, mean_interarrival=2.0)
    trace_a = generate_trace(config, seed=123)
    trace_b = generate_trace(config, seed=123)
    assert trace_a == trace_b


def test_generate_trace_differs_for_different_seeds_with_high_probability():
    config = TraceConfig(num_requests=25, mean_interarrival=2.0)
    trace_a = generate_trace(config, seed=1)
    trace_b = generate_trace(config, seed=2)
    assert trace_a != trace_b


def test_generate_trace_arrival_times_are_nondecreasing():
    trace = generate_trace(TraceConfig(num_requests=50, mean_interarrival=3.0), seed=5)
    arrivals = [spec.arrival_time for spec in trace]
    assert arrivals == sorted(arrivals)


def test_generate_trace_zero_mean_interarrival_means_all_arrive_at_zero():
    trace = generate_trace(TraceConfig(num_requests=10, mean_interarrival=0.0), seed=9)
    assert all(spec.arrival_time == 0.0 for spec in trace)


def test_generate_trace_respects_prompt_and_token_ranges():
    config = TraceConfig(
        num_requests=100,
        prompt_len_range=(5, 9),
        max_new_tokens_range=(10, 12),
        completion_fraction_range=(0.5, 0.5),
    )
    trace = generate_trace(config, seed=17)
    for spec in trace:
        assert 5 <= spec.prompt_len <= 9
        assert 10 <= spec.max_new_tokens <= 12
        assert 1 <= spec.target_output_len <= spec.max_new_tokens


def test_generate_trace_full_completion_fraction_uses_entire_budget():
    config = TraceConfig(
        num_requests=20,
        max_new_tokens_range=(30, 30),
        completion_fraction_range=(1.0, 1.0),
    )
    trace = generate_trace(config, seed=4)
    assert all(spec.target_output_len == spec.max_new_tokens for spec in trace)
