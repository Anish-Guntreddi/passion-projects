"""Spec SS1.7: 'Unsupported shapes must fail clearly (documented
supported-shape contract per variant).' This file exercises every
documented rejection path for both V0 (flashlite.reference.attention,
pure Python -- raises ValueError/NotImplementedError/TypeError) and V1
(flashlite._cuda_naive, pybind11 -- TORCH_CHECK raises RuntimeError with a
specific message), so a caller always gets a clear, specific exception
instead of a segfault, a silent wrong answer, or an opaque CUDA error.
"""

from __future__ import annotations

import pytest
import torch

from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import DEFAULT_SEED, make_qkv

# --- V0 (reference_attention) ------------------------------------------------


def test_reference_rejects_3d_input() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED)
    with pytest.raises(ValueError, match="4-D"):
        reference_attention(q.squeeze(0), k, v)


def test_reference_rejects_mismatched_qkv_shapes() -> None:
    q, _, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED)
    k_wrong = torch.zeros(1, 1, 4, 16)  # wrong head_dim
    with pytest.raises(ValueError, match="must share one shape"):
        reference_attention(q, k_wrong, v)


def test_reference_rejects_non_positive_dims() -> None:
    q = torch.zeros(1, 1, 0, 8)
    k = torch.zeros(1, 1, 0, 8)
    v = torch.zeros(1, 1, 0, 8)
    with pytest.raises(ValueError, match="non-positive"):
        reference_attention(q, k, v)


def test_reference_rejects_fp16_with_specific_message() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="ADR 0004"):
        reference_attention(q, k, v)


def test_reference_rejects_float64() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float64)
    with pytest.raises(ValueError, match="unsupported dtype"):
        reference_attention(q, k, v)


def test_reference_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        reference_attention([1, 2, 3], [1, 2, 3], [1, 2, 3])  # type: ignore[arg-type]


# --- V1 (flashlite._cuda_naive.attention_naive_forward) --------------------

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    from flashlite import _cuda_naive

    CUDA_NAIVE_EXT_AVAILABLE = True
except ImportError:
    CUDA_NAIVE_EXT_AVAILABLE = False

gpu_test = pytest.mark.gpu


@gpu_test
def test_naive_rejects_cpu_tensors() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        _cuda_naive.attention_naive_forward(q, k, v, False)


@gpu_test
def test_naive_rejects_non_contiguous_input() -> None:
    # Slice every other row out of a seq_len=16 tensor: same target shape
    # [1, 1, 8, 8] as k/v (so the shape-equality check passes), but a
    # stride-2 view along the sequence dimension is not contiguous.
    q_big, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k, v = k[:, :, :8, :].contiguous(), v[:, :, :8, :].contiguous()
    q_noncontig = q_big[:, :, ::2, :]
    assert q_noncontig.shape == k.shape
    assert not q_noncontig.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        _cuda_naive.attention_naive_forward(q_noncontig, k, v, False)


@gpu_test
def test_naive_rejects_mismatched_shapes() -> None:
    q, _k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k_wrong, _, _ = make_qkv(batch=1, heads=1, seq_len=9, head_dim=8, seed=DEFAULT_SEED + 1, device="cuda")
    with pytest.raises(RuntimeError, match="must share one shape"):
        _cuda_naive.attention_naive_forward(q, k_wrong, v, False)


@gpu_test
def test_naive_rejects_float64() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _cuda_naive.attention_naive_forward(q, k, v, False)


@gpu_test
def test_naive_rejects_wrong_ndim() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    with pytest.raises(RuntimeError, match="4-D"):
        _cuda_naive.attention_naive_forward(q[0], k, v, False)
