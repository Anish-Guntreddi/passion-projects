"""V0: PyTorch reference attention (standard tensor ops).

Mathematical invariant (docs/attention-math.md derives this in full):

    S = (Q K^T) / sqrt(d)         # [B, H, Sq, Sk]
    S = mask(S)                   # causal: S[..., i, j] = -inf for j > i
    P = softmax(S, dim=-1)        # row-wise, [B, H, Sq, Sk]
    O = P V                       # [B, H, Sq, D]

This is the ladder's V0: no custom kernel, no tiling, no online softmax --
just `torch.matmul` + `torch.softmax`, which is itself already numerically
stable (PyTorch's softmax subtracts the row max internally), so this
function is the ground truth every later CUDA variant (V1 naive, V2 tiled,
V3 online-softmax, V4 fused) is checked against (hard constraint 4:
"correctness and stability precede speed").

Supported shape/layout contract (ADR 0002, ADR 0003):
    - q, k, v: 4-D tensors, shape [batch, heads, seq_len, head_dim], the
      SAME shape for all three (self-attention: Sq == Sk == Sv == seq_len).
    - Any positive batch/heads/seq_len/head_dim. Awkward, non-tile-multiple
      seq_len values are fully supported here (this reference performs no
      tiling); tile-multiple constraints are introduced starting Phase 3 and
      documented on those variants specifically, not here.
    - dtype: float32 (ADR 0004: FP32 first). float16/bfloat16 raise a clear
      NotImplementedError rather than silently running in fp32 or producing
      quietly-wrong low-precision output.
    - Both CPU and CUDA tensors are accepted (this is a pure-PyTorch
      function; it has no CUDA-specific code path).

Unsupported input (wrong ndim, mismatched shapes, non-floating dtype,
fp16/bf16 for now) raises ValueError/NotImplementedError with a specific
message -- "fail clearly" per the spec's test-matrix requirement
(FlashLite spec Part 1 SS1.7: "Unsupported shapes must fail clearly").
"""

from __future__ import annotations

import math

import torch

_SUPPORTED_DTYPES = (torch.float32,)
_DEFERRED_DTYPES = (torch.float16, torch.bfloat16)


def _check_shape_contract(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"reference_attention: {name} must be a torch.Tensor, got {type(t)!r}")
        if t.dim() != 4:
            raise ValueError(
                f"reference_attention: {name} must be 4-D [batch, heads, seq_len, head_dim], "
                f"got shape {tuple(t.shape)} (ADR 0002: [B, H, S, D] layout only)"
            )

    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            "reference_attention: q, k, v must share one shape "
            f"[batch, heads, seq_len, head_dim] for this self-attention MVP; got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)} "
            "(cross-attention with Sq != Sk is out of scope for Phases 0-1)"
        )

    for name, t in (("q", q), ("k", k), ("v", v)):
        if any(s <= 0 for s in t.shape):
            raise ValueError(f"reference_attention: {name} has a non-positive dimension: {tuple(t.shape)}")

    if q.dtype in _DEFERRED_DTYPES:
        raise NotImplementedError(
            f"reference_attention: dtype {q.dtype} is deferred (ADR 0004: FP32 first, then FP16/BF16 "
            "in a later phase); cast inputs to torch.float32 explicitly."
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"reference_attention: unsupported dtype {q.dtype}; only torch.float32 is supported in "
            "Phases 0-1 (ADR 0004)."
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"reference_attention: q, k, v must share one dtype; got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Standard (non-fused) scaled dot-product attention, V0 of the ladder.

    Args:
        q, k, v: [batch, heads, seq_len, head_dim], same shape, float32.
        causal: if True, query position i may only attend to key positions
            j <= i (upper-triangular mask, j > i set to -inf before softmax).
        scale: score scale; defaults to 1/sqrt(head_dim) (the standard
            scaled-dot-product-attention convention).
        return_probs: if True, also return the [B, H, S, S] post-softmax
            probability matrix P (useful for tests/debugging; the fused V4
            variant will not materialize this, which is the whole point of
            Phase 5 -- this reference intentionally can, since V0 is meant
            to be "obviously correct by inspection", not memory-efficient).

    Returns:
        O with shape [batch, heads, seq_len, head_dim], or (O, P) if
        return_probs is True.
    """
    _check_shape_contract(q, k, v)

    _, _, seq_len, head_dim = q.shape
    effective_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) * effective_scale  # [B, H, S, S]

    if causal:
        # mask[i, j] = True where j > i (strictly-future keys, disallowed).
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)

    if return_probs:
        return out, probs
    return out
