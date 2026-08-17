// Shared shape/dtype/layout validation for every custom-kernel attention
// variant (V1 naive, V2 tiled, and later V3/V4). This is deliberately NOT
// part of the "one optimization variable per variant" ladder story (spec
// SS1.3) -- every variant in this repo accepts the exact SAME input
// contract (ADR 0002: [B, H, S, D] contiguous layout; ADR 0003: causal via
// flag; ADR 0004: float32 first); only what happens inside the kernel
// launch changes between variants. Extracting this one shared piece avoids
// N copies of the same TORCH_CHECK block silently drifting out of sync
// (different wording, a forgotten check) as variants are added -- the
// opposite of the kernel files themselves, which stay deliberately
// duplicated per variant (attention_naive.cu vs attention_tiled.cu) because
// THAT duplication is the point (spec: "Variants live side-by-side so Git
// history and the benchmark report show the optimization ladder").
//
// Originally inline in cuda_naive/bindings.cpp (Phase 1); extracted here in
// Phase 3 when cuda_tiled/bindings.cpp needed the identical contract.
#pragma once

#include <torch/extension.h>

namespace flashlite {

// Throws (via TORCH_CHECK, translated by pybind11 to a Python RuntimeError)
// if q, k, v do not satisfy the shared shape/dtype/layout contract.
// `variant_label` is folded into every message (e.g. "naive", "tiled") so a
// caller can tell which extension raised it without inspecting a traceback.
inline void validate_attention_inputs(const torch::Tensor& q, const torch::Tensor& k,
                                       const torch::Tensor& v, const char* variant_label) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flashlite ", variant_label,
              " attention: q, k, v must be CUDA tensors (got devices ", q.device(), ", ", k.device(),
              ", ", v.device(), ")");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 &&
                  v.scalar_type() == torch::kFloat32,
              "flashlite ", variant_label,
              " attention: q, k, v must be float32 (ADR 0004: FP32 first); got q=", q.scalar_type(),
              " k=", k.scalar_type(), " v=", v.scalar_type());
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "flashlite ", variant_label,
              " attention: q, k, v must be 4-D [batch, heads, seq_len, head_dim]; got q.dim()=",
              q.dim(), " k.dim()=", k.dim(), " v.dim()=", v.dim());
  TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), "flashlite ", variant_label,
              " attention: q, k, v must share one shape [B, H, S, D] (self-attention MVP, ADR 0002); "
              "got q=",
              q.sizes(), " k=", k.sizes(), " v=", v.sizes());
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "flashlite ", variant_label,
              " attention: q, k, v must be contiguous (ADR 0002: [B, H, S, D] contiguous layout only; "
              "call .contiguous() first)");
  for (int64_t d = 0; d < q.dim(); ++d) {
    TORCH_CHECK(q.size(d) > 0, "flashlite ", variant_label,
                " attention: all dims must be positive, got shape ", q.sizes());
  }
}

}  // namespace flashlite
