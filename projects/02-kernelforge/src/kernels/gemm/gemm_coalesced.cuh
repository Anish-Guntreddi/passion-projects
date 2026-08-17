// GEMM V2 (coalesced cleanup). ONE variable changes versus
// gemm_naive.cuh: which axis (M or N) maps to threadIdx.x (the
// warp-varying index). Block/grid geometry, the per-thread K-loop, and the
// lack of shared memory are all otherwise identical to V1.
//
// HYPOTHESIS (written before this kernel existed):
// V1 mapped `row` (M) to threadIdx.x, leaving `col` (N) warp-uniform. This
// kernel swaps that: `col` maps to threadIdx.x, `row` to threadIdx.y.
// Consequences per K-step, for one warp (32 consecutive threadIdx.x
// values, same threadIdx.y):
//   - A[row*K+k] read: `row` is now warp-uniform -> broadcast (same as
//     V1's B read was -- no change in kind, just which operand benefits).
//   - B[k*N+col] read: `col` now varies consecutively across the warp ->
//     32 lanes request 32 CONSECUTIVE addresses -> one coalesced 128-byte
//     transaction instead of 32 separate ones.
//   - C[row*N+col] write: `col` varies consecutively -> coalesced write,
//     same mechanism as the B read above.
// Expect a substantial speedup over V1 at every size, growing with matrix
// size (more K-steps and more output elements amplify the same per-access
// saving), with NO change to arithmetic work or occupancy -- this rung
// isolates "global memory coalescing" as its own measurable mechanism,
// same story as Phase 1's transpose naive-vs-tiled comparison but for
// GEMM's read pattern specifically. See src/kernels/gemm/README.md for
// the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void gemm_coalesced_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                            cudaStream_t stream, dim3& out_grid, dim3& out_block);

float gemm_coalesced_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k);

} // namespace kernelforge::kernels
