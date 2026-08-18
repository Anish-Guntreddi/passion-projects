"""Variant dispatcher: selects an attention implementation by name.

This is the one place that knows about every variant in the ladder
(spec Part 1 SS1.3). Phases 0-1 registered "reference" (V0) and "naive"
(V1); Phase 3 added "tiled" (V2); Phase 4 adds "online_softmax" (V3);
Phase 5 adds "fused" (V4) -- all unchanged in shape, so the
benchmark/comparison harness (Phase 7) can sweep every registered variant
without knowing their internals.
"""

from __future__ import annotations

import torch

from flashlite.reference.attention import reference_attention

VARIANTS = ("reference", "naive", "tiled", "online_softmax", "fused")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    variant: str = "reference",
) -> torch.Tensor:
    """Dispatches to the named attention variant.

    variant="reference": V0, pure PyTorch (flashlite.reference.attention),
        runs on CPU or CUDA.
    variant="naive":     V1, custom CUDA kernel (flashlite._cuda_naive),
        requires CUDA float32 contiguous [B, H, S, S==Sk] inputs (raises
        RuntimeError with a specific message otherwise -- see
        src/flashlite/cuda_naive/bindings.cpp).
    variant="tiled":     V2, shared-memory-tiled CUDA kernel
        (flashlite._cuda_tiled), same input contract as "naive" (raises the
        same class of RuntimeError with a specific message -- see
        src/flashlite/cuda_tiled/bindings.cpp and
        cuda_common/shape_validate.hpp).
    variant="online_softmax": V3, shared-memory-tiled QK^T/PV (unchanged
        from "tiled") with a single-combined-pass online-softmax row
        normalization (flashlite._cuda_online_softmax; docs/online-softmax.md),
        same input contract as "naive"/"tiled".
    variant="fused":    V4, single fused kernel with no [B,H,S,S] scores
        buffer at all (flashlite._cuda_fused; attention_fused.cuh), same
        input contract as "naive"/"tiled"/"online_softmax" PLUS
        head_dim <= 128 (ADR 0011; raises RuntimeError otherwise).
    """
    if variant == "reference":
        return reference_attention(q, k, v, causal=causal)
    if variant == "naive":
        from flashlite import (
            _cuda_naive,  # compiled extension (Phase 1); import deferred
        )

        return _cuda_naive.attention_naive_forward(q, k, v, causal)
    if variant == "tiled":
        from flashlite import (
            _cuda_tiled,  # compiled extension (Phase 3); import deferred
        )

        return _cuda_tiled.attention_tiled_forward(q, k, v, causal)
    if variant == "online_softmax":
        from flashlite import (
            _cuda_online_softmax,  # compiled extension (Phase 4); import deferred
        )

        return _cuda_online_softmax.attention_online_softmax_forward(q, k, v, causal)
    if variant == "fused":
        from flashlite import (
            _cuda_fused,  # compiled extension (Phase 5); import deferred
        )

        return _cuda_fused.attention_fused_forward(q, k, v, causal)
    raise ValueError(f"ops.attention: unknown variant {variant!r}; expected one of {VARIANTS}")
