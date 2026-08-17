// GEMM V1 (naive, one thread per output element, no shared memory).
//
// C = A * B. A is M x K row-major, B is K x N row-major, C (output) is
// M x N row-major (spec FR2's literal GEMM ladder "V1 one thread/output"
// rung; the CPU reference `kernelforge::reference::gemm` is this ladder's
// V0).
//
// HYPOTHESIS (written before this kernel existed):
// Every one of the M*N output elements gets its own thread, which reads a
// full K-length row of A and a full K-length column of B straight from
// global memory (no reuse across threads, no shared memory, no tiling) and
// accumulates their dot product directly into a register, then writes one
// element of C. This kernel's thread-to-output mapping is DELIBERATELY the
// "wrong way round" for this GPU's warp-level memory coalescing: `row`
// (the M-dimension index) varies across a warp's 32 lanes (threadIdx.x),
// while `col` (the N-dimension index) is constant within a warp
// (threadIdx.y). Consequences per K-step, for one warp:
//   - A[row*K+k] read: `row` differs per lane -> 32 lanes touch 32
//     different, K-elements-apart cache lines -> badly uncoalesced.
//   - B[k*N+col] read: `col` is warp-uniform -> every lane requests the
//     SAME address -> serviced as one broadcast (this one is actually
//     already efficient, incidentally -- V2 does not "fix" B's read, it
//     fixes A's read and C's write by making `col`, not `row`, the
//     warp-varying index).
//   - C[row*N+col] write: `row` differs per lane -> 32 lanes write to 32
//     addresses N elements apart -> badly uncoalesced.
// Expect this to be the ladder's slowest rung, dominated by uncoalesced A
// reads and C writes; V2 changes exactly one variable (which axis maps to
// threadIdx.x) to fix both at once. See src/kernels/gemm/README.md for the
// measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void gemm_naive_launch(const float* d_a, const float* d_b, float* d_c, int m, int n, int k,
                        cudaStream_t stream, dim3& out_grid, dim3& out_block);

// Host convenience wrapper: allocates device buffers, copies h_a/h_b in,
// launches, synchronizes, copies h_c out, frees. Returns kernel-only
// elapsed ms (H2D/D2H copy excluded, FR4). m == 0, n == 0, or k == 0 are
// all legal and short-circuit without touching the GPU for a zero-size
// result (k == 0 still allocates/zeros a non-empty C, since M x N can be
// nonzero even when K is 0 -- the sum over an empty K range is 0).
float gemm_naive_run_host(const float* h_a, const float* h_b, float* h_c, int m, int n, int k);

} // namespace kernelforge::kernels
