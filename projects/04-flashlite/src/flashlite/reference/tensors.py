"""Deterministic seeded tensor generation (mirrors KernelForge's
src/common/rng.hpp: same seed -> same input data, on any machine, forever,
so a test failure is reproducible and the *same* Q/K/V feed every variant
under comparison -- the input data must not be a hidden second variable).

std.mt19937_64-equivalent determinism here is `torch.Generator` seeded with
`manual_seed`. Generation always happens on the CPU generator, then the
result is moved to the requested device -- this keeps Q/K/V byte-identical
across CPU and CUDA runs (CUDA's philox RNG state is not guaranteed to
reproduce the same stream as the CPU generator for the same seed, so
generating on CPU and transferring is what actually guarantees the
reference (which may run on CPU or CUDA) and the CUDA kernel see the exact
same numbers).
"""

from __future__ import annotations

import torch

# Fixed default seed, matching kernelforge::kDefaultSeed exactly (portfolio
# convention: the same default seed value is reused project-to-project).
DEFAULT_SEED: int = 123456789


def make_qkv(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    *,
    seed: int = DEFAULT_SEED,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    lo: float = -1.0,
    hi: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates deterministic (q, k, v), each [batch, heads, seq_len, head_dim].

    q, k, v are drawn as three independent, sequential samples from one
    CPU generator seeded with `seed` (q first, then k, then v) -- so the
    seed alone fully determines all three tensors, in this fixed order,
    reproducibly across processes and machines.

    Values are drawn Uniform(lo, hi) (default [-1, 1)), matching
    kernelforge::make_random_vector's default range.
    """
    for name, val in (("batch", batch), ("heads", heads), ("seq_len", seq_len), ("head_dim", head_dim)):
        if val <= 0:
            raise ValueError(f"make_qkv: {name} must be positive, got {val}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    shape = (batch, heads, seq_len, head_dim)
    q = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi, generator=gen)
    k = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi, generator=gen)
    v = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi, generator=gen)

    q = q.to(dtype=dtype, device=device).contiguous()
    k = k.to(dtype=dtype, device=device).contiguous()
    v = v.to(dtype=dtype, device=device).contiguous()
    return q, k, v


def make_extreme_qkv(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    *,
    seed: int = DEFAULT_SEED,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    magnitude: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic Q/K/V scaled to produce large-magnitude pre-softmax
    logits (Q, K entries in [-magnitude, magnitude] rather than [-1, 1]).

    Used to stress-test softmax numerical stability (spec SS1.7: "extreme
    logits to stress softmax stability"): a naive softmax that exponentiates
    raw scores without subtracting the row max overflows float32 at scores
    of a few hundred; this makes it easy to construct exactly that scenario
    on demand. V is left at the normal [-1, 1) range since only Q/K feed the
    part of the computation (the score, pre-softmax) that stability depends
    on.
    """
    q, k, v = make_qkv(batch, heads, seq_len, head_dim, seed=seed, dtype=torch.float32, device="cpu")
    q = q * magnitude
    k = k * magnitude
    q = q.to(dtype=dtype, device=device).contiguous()
    k = k.to(dtype=dtype, device=device).contiguous()
    v = v.to(dtype=dtype, device=device).contiguous()
    return q, k, v
