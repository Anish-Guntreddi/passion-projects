// Softmax V3 (fused online max+sum, bounded). ONE variable changes versus
// softmax_warp_shuffle.cuh: the row max and exp-sum are computed TOGETHER
// in a single pass over the row (the "online softmax" streaming-rescale
// technique -- a well-known public numerical-methods trick, not sourced
// from or copied out of any specific external kernel implementation),
// cutting global-memory reads of each row from three down to two. The
// warp-shuffle block-reduction MECHANISM carries over unchanged from V2 --
// this rung's variable is the pass count / memory traffic, not the
// reduction mechanism.
//
// HYPOTHESIS (written before this kernel existed):
// V1/V2 both read every row three times: once to find the max, once (now
// that the max is known) to accumulate the exp-sum, once more to produce
// the final normalized output. The first two passes can be fused: track a
// RUNNING max `m` and a running, CONTINUOUSLY RESCALED sum `l` together as
// each element is visited once. When a new element `x` arrives:
//   m_new = max(m, x)
//   l_new = l * exp(m - m_new) + exp(x - m_new)
// This is mathematically exact (not an approximation) and, critically,
// ASSOCIATIVE -- combining two partial (m, l) pairs from disjoint element
// sets uses the identical rescale formula, so the same warp-shuffle
// block-reduction shape V2 already uses works unchanged here, just
// reducing PAIRS instead of single floats. This removes the max-only pass
// entirely: TWO passes over the row (one combined max+sum pass, one
// normalize-and-write pass) instead of three -- a disclosed, BOUNDED
// fusion (it fuses what can be fused without assuming the whole row fits
// on-chip, so it stays correct for any `cols`; fully caching a row in
// shared memory to also avoid the second pass is explicit future work,
// not silently included here). Expect roughly a 1/3 reduction in this
// kernel's global-memory read traffic versus V1/V2, and a real (if
// smaller than a full 1.5x) speedup, since compute (the reductions
// themselves) is not free either. See src/kernels/softmax/README.md for
// the measured numbers.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

void softmax_fused_online_launch(const float* d_in, float* d_out, int rows, int cols,
                                  int block_size, cudaStream_t stream, dim3& out_grid,
                                  dim3& out_block);

float softmax_fused_online_run_host(const float* h_in, float* h_out, int rows, int cols,
                                     int block_size = 256);

} // namespace kernelforge::kernels
