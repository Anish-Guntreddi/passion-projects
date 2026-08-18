"""Spec SS1.7: 'Unsupported shapes must fail clearly (documented
supported-shape contract per variant).' This file exercises every
documented rejection path for V0 (flashlite.reference.attention, pure
Python -- raises ValueError/NotImplementedError/TypeError), V1
(flashlite._cuda_naive, pybind11 -- TORCH_CHECK raises RuntimeError with a
specific message), V2 (flashlite._cuda_tiled, same TORCH_CHECK contract via
cuda_common/shape_validate.hpp), V3 (flashlite._cuda_online_softmax, same
shared contract), and V4 (flashlite._cuda_fused, same shared contract PLUS
its own head_dim<=128 bound, ADR 0011), so a caller always gets a clear,
specific exception instead of a segfault, a silent wrong answer, or an
opaque CUDA error.
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


# --- V2 (flashlite._cuda_tiled.attention_tiled_forward) --------------------
# Same rejection paths as V1, verified against V2 too since both bindings
# route through the same shared cuda_common/shape_validate.hpp -- these
# tests are the guarantee that sharing the validator did not silently
# change or drop a check for one of the two variants.

try:
    from flashlite import _cuda_tiled

    CUDA_TILED_EXT_AVAILABLE = True
except ImportError:
    CUDA_TILED_EXT_AVAILABLE = False


@gpu_test
def test_tiled_rejects_cpu_tensors() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        _cuda_tiled.attention_tiled_forward(q, k, v, False)


@gpu_test
def test_tiled_rejects_non_contiguous_input() -> None:
    q_big, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k, v = k[:, :, :8, :].contiguous(), v[:, :, :8, :].contiguous()
    q_noncontig = q_big[:, :, ::2, :]
    assert q_noncontig.shape == k.shape
    assert not q_noncontig.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        _cuda_tiled.attention_tiled_forward(q_noncontig, k, v, False)


@gpu_test
def test_tiled_rejects_mismatched_shapes() -> None:
    q, _k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k_wrong, _, _ = make_qkv(batch=1, heads=1, seq_len=9, head_dim=8, seed=DEFAULT_SEED + 1, device="cuda")
    with pytest.raises(RuntimeError, match="must share one shape"):
        _cuda_tiled.attention_tiled_forward(q, k_wrong, v, False)


@gpu_test
def test_tiled_rejects_float64() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _cuda_tiled.attention_tiled_forward(q, k, v, False)


@gpu_test
def test_tiled_rejects_wrong_ndim() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    with pytest.raises(RuntimeError, match="4-D"):
        _cuda_tiled.attention_tiled_forward(q[0], k, v, False)


# --- V3 (flashlite._cuda_online_softmax.attention_online_softmax_forward) --
# Same rejection paths as V1/V2, verified against V3 too since it routes
# through the same shared cuda_common/shape_validate.hpp.

try:
    from flashlite import _cuda_online_softmax

    CUDA_ONLINE_SOFTMAX_EXT_AVAILABLE = True
except ImportError:
    CUDA_ONLINE_SOFTMAX_EXT_AVAILABLE = False


@gpu_test
def test_online_softmax_rejects_cpu_tensors() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        _cuda_online_softmax.attention_online_softmax_forward(q, k, v, False)


@gpu_test
def test_online_softmax_rejects_non_contiguous_input() -> None:
    q_big, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k, v = k[:, :, :8, :].contiguous(), v[:, :, :8, :].contiguous()
    q_noncontig = q_big[:, :, ::2, :]
    assert q_noncontig.shape == k.shape
    assert not q_noncontig.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        _cuda_online_softmax.attention_online_softmax_forward(q_noncontig, k, v, False)


@gpu_test
def test_online_softmax_rejects_mismatched_shapes() -> None:
    q, _k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k_wrong, _, _ = make_qkv(batch=1, heads=1, seq_len=9, head_dim=8, seed=DEFAULT_SEED + 1, device="cuda")
    with pytest.raises(RuntimeError, match="must share one shape"):
        _cuda_online_softmax.attention_online_softmax_forward(q, k_wrong, v, False)


@gpu_test
def test_online_softmax_rejects_float64() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _cuda_online_softmax.attention_online_softmax_forward(q, k, v, False)


@gpu_test
def test_online_softmax_rejects_wrong_ndim() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    with pytest.raises(RuntimeError, match="4-D"):
        _cuda_online_softmax.attention_online_softmax_forward(q[0], k, v, False)


# --- V4 (flashlite._cuda_fused.attention_fused_forward) --------------------
# Same shared rejection paths as V1/V2/V3, PLUS one V4-specific bound
# (head_dim <= kMaxHeadDimFused, ADR 0011).

try:
    from flashlite import _cuda_fused

    CUDA_FUSED_EXT_AVAILABLE = True
except ImportError:
    CUDA_FUSED_EXT_AVAILABLE = False


@gpu_test
def test_fused_rejects_cpu_tensors() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        _cuda_fused.attention_fused_forward(q, k, v, False)


@gpu_test
def test_fused_rejects_non_contiguous_input() -> None:
    q_big, k, v = make_qkv(batch=1, heads=1, seq_len=16, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k, v = k[:, :, :8, :].contiguous(), v[:, :, :8, :].contiguous()
    q_noncontig = q_big[:, :, ::2, :]
    assert q_noncontig.shape == k.shape
    assert not q_noncontig.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        _cuda_fused.attention_fused_forward(q_noncontig, k, v, False)


@gpu_test
def test_fused_rejects_mismatched_shapes() -> None:
    q, _k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    k_wrong, _, _ = make_qkv(batch=1, heads=1, seq_len=9, head_dim=8, seed=DEFAULT_SEED + 1, device="cuda")
    with pytest.raises(RuntimeError, match="must share one shape"):
        _cuda_fused.attention_fused_forward(q, k_wrong, v, False)


@gpu_test
def test_fused_rejects_float64() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _cuda_fused.attention_fused_forward(q, k, v, False)


@gpu_test
def test_fused_rejects_wrong_ndim() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=8, seed=DEFAULT_SEED, device="cuda")
    with pytest.raises(RuntimeError, match="4-D"):
        _cuda_fused.attention_fused_forward(q[0], k, v, False)


@gpu_test
def test_fused_rejects_head_dim_above_max() -> None:
    """V4-specific bound (attention_fused.cuh's kMaxHeadDimFused=128,
    ADR 0011) -- not shared by V1/V2/V3, which accept any positive
    head_dim; this must fail clearly (a specific TORCH_CHECK message
    naming ADR 0011), not silently corrupt memory via an out-of-bounds
    fixed-size register array write.
    """
    q, k, v = make_qkv(batch=1, heads=1, seq_len=8, head_dim=256, seed=DEFAULT_SEED, device="cuda")
    with pytest.raises(RuntimeError, match="kMaxHeadDimFused"):
        _cuda_fused.attention_fused_forward(q, k, v, False)
