from __future__ import annotations

import pytest

from minipaged.sampling.config import SamplingConfig, SamplingConfigError


def test_defaults_are_valid():
    cfg = SamplingConfig()
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k is None
    assert cfg.seed is None


@pytest.mark.parametrize("temperature", [-0.1, -1.0])
def test_rejects_negative_temperature(temperature):
    with pytest.raises(SamplingConfigError, match="temperature"):
        SamplingConfig(temperature=temperature)


def test_zero_temperature_is_allowed():
    # Temperature 0 means "greedy" once sampling logic exists (Phase 5);
    # it is a valid config value even though nothing samples yet.
    SamplingConfig(temperature=0.0)


@pytest.mark.parametrize("top_p", [0.0, -0.5, 1.1])
def test_rejects_out_of_range_top_p(top_p):
    with pytest.raises(SamplingConfigError, match="top_p"):
        SamplingConfig(top_p=top_p)


def test_top_p_of_one_is_allowed():
    SamplingConfig(top_p=1.0)


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_non_positive_top_k(top_k):
    with pytest.raises(SamplingConfigError, match="top_k"):
        SamplingConfig(top_k=top_k)


def test_top_k_none_is_allowed():
    SamplingConfig(top_k=None)


def test_config_is_frozen():
    cfg = SamplingConfig()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        cfg.temperature = 0.5  # type: ignore[misc]
