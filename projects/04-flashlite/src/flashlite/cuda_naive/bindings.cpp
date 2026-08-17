// pybind11 / torch.utils.cpp_extension bindings for the V1 naive attention
// kernel (ADR 0005: CUDA-extension <-> Python integration route).
//
// Shape/dtype/layout validation (before any kernel launches) is shared with
// every other variant's bindings via cuda_common/shape_validate.hpp (moved
// out of this file in Phase 3, when cuda_tiled/bindings.cpp needed the
// identical contract) -- the kernels themselves (attention_naive.cu) assume
// valid input, per the same separation of concerns KernelForge uses
// (launch_validate.hpp validates before kernel launch, kernels themselves
// stay simple). TORCH_CHECK failures become Python RuntimeError with the
// exact message documented there (spec: "unsupported shapes must fail
// clearly").
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "attention_naive.cuh"
#include "../cuda_common/shape_validate.hpp"

namespace {

torch::Tensor attention_naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  flashlite::validate_attention_inputs(q, k, v, "naive");

  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);

  auto out = torch::empty_like(q);
  auto scores = torch::empty({B, H, S, S}, q.options());

  flashlite::launch_attention_naive(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                     out.data_ptr<float>(), scores.data_ptr<float>(),
                                     static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
                                     static_cast<int>(D), causal, at::cuda::getCurrentCUDAStream());

  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_naive_forward", &attention_naive_forward,
        "FlashLite V1 naive attention forward (CUDA): materialized scores + two-pass softmax + "
        "weighted sum, checked against the V0 PyTorch reference.",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
