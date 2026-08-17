"""Global seed handling.

A single entry point (:func:`set_seed`) seeds every RNG ForgeLM touches
(Python's ``random``, NumPy, and torch on both CPU and CUDA) and opts into
deterministic algorithms where torch supports them. Every CLI command and
test that needs reproducibility calls this before doing anything else.

``torch.use_deterministic_algorithms`` is requested with ``warn_only=True``:
a handful of CUDA ops (notably some scatter/index-based kernels we are not
using yet) have no deterministic implementation, and we would rather log a
warning than hard-crash a run that doesn't touch them.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs ForgeLM uses so a run is reproducible given the same seed.

    Args:
        seed: the seed value.
        deterministic: when True (default), also request deterministic
            algorithms from torch/cuDNN. Set to False only for throughput
            benchmarking, where determinism can cost meaningful performance.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # PYTHONHASHSEED only affects hash randomization for *new* processes; we
    # still set it so subprocesses spawned by this one (e.g. DataLoader
    # workers that fork) inherit a fixed value.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``torch.utils.data.DataLoader`` with num_workers > 0.

    Combines the base torch seed (already derived deterministically from the
    generator passed to the DataLoader) with the worker id so each worker
    gets a distinct-but-deterministic seed.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
