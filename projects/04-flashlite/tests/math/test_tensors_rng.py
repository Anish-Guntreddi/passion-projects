"""Deterministic seeded RNG (FlashLite spec SS1.7: 'random seeds +
deterministic cases'): same seed -> byte-identical Q/K/V, every time, so a
test failure is reproducible and the SAME input data feeds every variant
under comparison.
"""

from __future__ import annotations

import pytest
import torch

from flashlite.reference.tensors import DEFAULT_SEED, make_extreme_qkv, make_qkv


def test_same_seed_reproduces_identical_tensors() -> None:
    q1, k1, v1 = make_qkv(batch=2, heads=3, seq_len=11, head_dim=16, seed=42)
    q2, k2, v2 = make_qkv(batch=2, heads=3, seq_len=11, head_dim=16, seed=42)
    torch.testing.assert_close(q1, q2, atol=0, rtol=0)
    torch.testing.assert_close(k1, k2, atol=0, rtol=0)
    torch.testing.assert_close(v1, v2, atol=0, rtol=0)


def test_different_seeds_produce_different_tensors() -> None:
    q1, _, _ = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=1)
    q2, _, _ = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=2)
    assert not torch.allclose(q1, q2)


def test_q_k_v_are_independent_draws() -> None:
    """q, k, v must not be identical to each other (they are three
    sequential draws from the same generator, in a fixed order)."""
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED)
    assert not torch.allclose(q, k)
    assert not torch.allclose(k, v)
    assert not torch.allclose(q, v)


def test_default_seed_matches_kernelforge_convention() -> None:
    assert DEFAULT_SEED == 123456789


def test_values_within_requested_range() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=64, head_dim=32, seed=DEFAULT_SEED, lo=-2.0, hi=2.0)
    for t in (q, k, v):
        assert torch.all(t >= -2.0)
        assert torch.all(t < 2.0)


def test_dtype_and_device_and_contiguity() -> None:
    q, k, v = make_qkv(batch=1, heads=2, seq_len=8, head_dim=16, seed=DEFAULT_SEED, dtype=torch.float32, device="cpu")
    for t in (q, k, v):
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"
        assert t.is_contiguous()
        assert t.shape == (1, 2, 8, 16)


@pytest.mark.parametrize("bad_kwargs", [
    {"batch": 0}, {"heads": -1}, {"seq_len": 0}, {"head_dim": -4},
])
def test_non_positive_dims_raise_clearly(bad_kwargs: dict) -> None:
    kwargs = {"batch": 1, "heads": 1, "seq_len": 4, "head_dim": 8}
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        make_qkv(**kwargs)


def test_extreme_qkv_produces_large_magnitude_scores() -> None:
    q, k, _ = make_extreme_qkv(batch=1, heads=1, seq_len=16, head_dim=32, seed=DEFAULT_SEED, magnitude=80.0)
    assert float(q.abs().max()) > 10.0
    assert float(k.abs().max()) > 10.0
