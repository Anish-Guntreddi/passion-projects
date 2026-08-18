// pybind11 / torch.utils.cpp_extension bindings for the V4 fused attention
// kernel (ADR 0005: CUDA-extension <-> Python integration route, same
// route as V1/V2/V3). Shape/dtype/layout validation is shared with every
// other variant via cuda_common/shape_validate.hpp, PLUS one V4-specific
// addition: `head_dim <= kMaxHeadDimFused` (attention_fused.cuh's header;
// ADR 0011), enforced here rather than in the shared validator since it is
// a V4-only limitation (every earlier variant supports any positive
// head_dim), consistent with cuda_common/shape_validate.hpp's own header
// comment: shared checks live there, per-variant-specific ones live in
// that variant's own bindings.cpp.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "attention_fused.cuh"
#include "../cuda_common/shape_validate.hpp"

namespace {

torch::Tensor attention_fused_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  flashlite::validate_attention_inputs(q, k, v, "fused");

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= flashlite::kMaxHeadDimFused, "flashlite fused attention: head_dim=", D,
              " exceeds the V4 fused kernel's supported maximum (kMaxHeadDimFused=",
              flashlite::kMaxHeadDimFused,
              "; ADR 0011) -- every head_dim this repo's test matrix uses (32/64/128) is well within "
              "this bound; use variant=\"tiled\" or \"online_softmax\" for larger head_dim.");

  // NOTE: unlike V1/V2/V3's bindings, there is NO `torch::empty({B, H, S,
  // S}, ...)` scores-scratch allocation here at all -- this is the literal
  // "full attention-matrix materialization removed" (roadmap Phase 5)
  // this variant exists to demonstrate; see attention_fused.cuh's header
  // and tests/correctness/test_fused_attention.py's peak-memory test.
  auto out = torch::empty_like(q);

  flashlite::launch_attention_fused(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                     out.data_ptr<float>(), static_cast<int>(B), static_cast<int>(H),
                                     static_cast<int>(S), static_cast<int>(D), causal,
                                     at::cuda::getCurrentCUDAStream());

  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_fused_forward", &attention_fused_forward,
        "FlashLite V4 fused attention forward (CUDA): single kernel, no [B,H,S,S] scores buffer -- "
        "QK^T computed tile-by-tile directly into a running online-softmax output accumulator "
        "(docs/online-softmax.md), checked against the V0 PyTorch reference.",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
