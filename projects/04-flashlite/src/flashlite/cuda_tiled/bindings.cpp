// pybind11 / torch.utils.cpp_extension bindings for the V2 tiled attention
// kernel (ADR 0005: CUDA-extension <-> Python integration route, same
// route as V1). Shape/dtype/layout validation is shared with V1 via
// cuda_common/shape_validate.hpp -- see that file's header for why this
// one piece is shared while the kernels themselves are not.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "attention_tiled.cuh"
#include "../cuda_common/shape_validate.hpp"

namespace {

torch::Tensor attention_tiled_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  flashlite::validate_attention_inputs(q, k, v, "tiled");

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);

  auto out = torch::empty_like(q);
  auto scores = torch::empty({B, H, S, S}, q.options());

  flashlite::launch_attention_tiled(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                     out.data_ptr<float>(), scores.data_ptr<float>(),
                                     static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
                                     static_cast<int>(D), causal, at::cuda::getCurrentCUDAStream());

  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_tiled_forward", &attention_tiled_forward,
        "FlashLite V2 tiled attention forward (CUDA): shared-memory-tiled QK^T and PV, ordinary "
        "(non-online) row softmax retained, checked against the V0 PyTorch reference.",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
