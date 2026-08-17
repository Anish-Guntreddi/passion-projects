#include "kernels/gemm/gemm_cublas.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#include "common/error_check.cuh"
#include "common/gpu_timer.cuh"

namespace kernelforge::kernels {

namespace {

void cublas_check(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS call failed: ") + expr +
                              " (status=" + std::to_string(static_cast<int>(status)) + ")");
  }
}

// Lazily-created, process-lifetime cuBLAS handle (Meyer's singleton,
// never explicitly destroyed). Deliberate: cublasCreate() does nontrivial
// one-time setup (allocates internal workspace, picks kernels for this
// GPU's architecture) that has nothing to do with the SGEMM call this
// file measures. Creating the handle once, on first use, and reusing it
// for every later call (including every warmup AND every measured
// repetition in bench_gemm_main.cpp's timed loop) keeps that one-time
// setup cost OUT of every GpuTimer-measured interval below, the same way
// this repo's other GEMM variants allocate their device buffers once
// before their timed loop rather than inside it. A never-destroyed handle
// is a benign one-time leak reclaimed at process exit -- this file backs
// a benchmark binary, not a long-running service.
cublasHandle_t& get_cublas_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    cublas_check(cublasCreate(&h), "cublasCreate");
    return h;
  }();
  return handle;
}

} // namespace

void gemm_cublas_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                         cudaStream_t stream, dim3& out_grid, dim3& out_block) {
  // cuBLAS picks its own internal launch geometry, opaque to this repo;
  // (0,0,0) is an explicit "not applicable" sentinel (see
  // src/kernels/gemm/README.md), not a real 0-thread launch.
  out_grid = dim3(0, 0, 0);
  out_block = dim3(0, 0, 0);
  if (m == 0 || n == 0) return;
  if (k == 0) {
    // cuBLAS requires ldb >= max(1, k_cb) for this call's column-major
    // B_cb operand (our d_a, with ldb == k -- see the derivation below);
    // k == 0 would pass ldb == 0, which cublasSgemm rejects outright
    // ("parameter number 10 had an illegal value", verified empirically)
    // rather than treating it as the mathematically well-defined "sum
    // over zero terms" every other GEMM variant here computes naturally
    // via its K-loop not executing. K == 0 legitimately means C == 0
    // (FR6: 0-sized edge dimensions are legal), so produce that directly.
    KF_CUDA_CHECK(cudaMemsetAsync(d_c, 0, static_cast<std::size_t>(m) * static_cast<std::size_t>(n) *
                                              sizeof(float), stream));
    return;
  }

  cublasHandle_t handle = get_cublas_handle();
  cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major C(MxN) = A(MxK) * B(KxN) via cuBLAS's column-major API:
  // reading our row-major buffers as column-major gives their transpose
  // (A_buf -> A^T (KxM), B_buf -> B^T (NxK), C_buf -> C^T (NxM)), and
  // C^T = B^T * A^T -- so this ONE cublasSgemm call, with A and B swapped
  // and M/N swapped, computes exactly the row-major product this repo's
  // other GEMM variants compute. See docs/decisions/
  // 0013-cublas-ceiling-methodology.md for the full derivation.
  cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            /*m_cb=*/n, /*n_cb=*/m, /*k_cb=*/k, &alpha,
                            /*A_cb=*/d_b, /*lda=*/n,
                            /*B_cb=*/d_a, /*ldb=*/k, &beta,
                            /*C_cb=*/d_c, /*ldc=*/n),
               "cublasSgemm");
}

float gemm_cublas_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {
  const std::size_t c_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  if (c_elems == 0) return 0.0f;

  const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(float);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t c_bytes = c_elems * sizeof(float);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  KF_CUDA_CHECK(cudaMalloc(&d_a, a_bytes > 0 ? a_bytes : sizeof(float)));
  KF_CUDA_CHECK(cudaMalloc(&d_b, b_bytes > 0 ? b_bytes : sizeof(float)));
  KF_CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  if (a_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
  if (b_bytes > 0) KF_CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;
  dim3 grid, block;
  timer.start();
  gemm_cublas_launch(d_a, d_b, d_c, m, n, k, 0, grid, block);
  const float elapsed_ms = timer.stop_ms();

  KF_CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
  KF_CUDA_CHECK(cudaFree(d_a));
  KF_CUDA_CHECK(cudaFree(d_b));
  KF_CUDA_CHECK(cudaFree(d_c));
  return elapsed_ms;
}

} // namespace kernelforge::kernels
