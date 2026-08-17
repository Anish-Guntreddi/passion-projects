// pybind11 / torch.utils.cpp_extension bindings for the V1 naive attention
// kernel (ADR 0005: CUDA-extension <-> Python integration route).
//
// This is the ONLY place that validates a torch::Tensor's shape/dtype/
// layout before any kernel launches -- the kernels themselves
// (attention_naive.cu) assume valid input, per the same separation of
// concerns KernelForge uses (launch_validate.hpp validates before kernel
// launch, kernels themselves stay simple). TORCH_CHECK failures become
// Python RuntimeError with the exact message below (spec: "unsupported
// shapes must fail clearly").
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "attention_naive.cuh"

namespace {

torch::Tensor attention_naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "flashlite naive attention: q, k, v must be CUDA tensors (got devices ", q.device(), ", ",
              k.device(), ", ", v.device(), ")");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 &&
                  v.scalar_type() == torch::kFloat32,
              "flashlite naive attention: q, k, v must be float32 (ADR 0004: FP32 first); got q=",
              q.scalar_type(), " k=", k.scalar_type(), " v=", v.scalar_type());
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "flashlite naive attention: q, k, v must be 4-D [batch, heads, seq_len, head_dim]; got "
              "q.dim()=",
              q.dim(), " k.dim()=", k.dim(), " v.dim()=", v.dim());
  TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
              "flashlite naive attention: q, k, v must share one shape [B, H, S, D] (self-attention "
              "MVP, ADR 0002); got q=",
              q.sizes(), " k=", k.sizes(), " v=", v.sizes());
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "flashlite naive attention: q, k, v must be contiguous (ADR 0002: [B, H, S, D] "
              "contiguous layout only; call .contiguous() first)");
  for (int64_t d = 0; d < q.dim(); ++d) {
    TORCH_CHECK(q.size(d) > 0, "flashlite naive attention: all dims must be positive, got shape ",
                q.sizes());
  }

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
