from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from forgelm.seed import set_seed


def test_set_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_set_seed_makes_python_random_reproducible() -> None:
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_cpu_reproducible() -> None:
    set_seed(42)
    a = torch.randn(5, 5)
    set_seed(42)
    b = torch.randn(5, 5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_diverge() -> None:
    set_seed(1)
    a = torch.randn(8)
    set_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_set_seed_sets_pythonhashseed_env_var() -> None:
    import os

    set_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
