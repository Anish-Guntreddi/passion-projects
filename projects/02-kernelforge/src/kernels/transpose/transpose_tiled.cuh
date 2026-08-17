// Transpose V2 (shared-memory tiled, coalesced read AND write): stages
// each TILE x TILE block through shared memory so that both the global
// read and the global write are coalesced. This is the ONE variable
// changed versus transpose_naive.cuh (same TILE=32x32 launch shape,
// same rows/cols, same input data) — see docs/optimization-method.md.
//
// Known, deliberately-NOT-fixed-yet side effect: reading the shared-memory
// tile back out in transposed order (`tile[threadIdx.x][threadIdx.y]`)
// causes a shared-memory bank conflict (all 32 threads of a warp land on
// the same bank), because this variant does not pad the tile. That fix is
// V4 in the spec's ladder (bank-conflict padding) and is explicitly out of
// scope for Phase 1 (see docs/decisions and the Phase 1 postmortem in
// benchmarks/methodology.md) — recorded here so the omission is documented,
// not silently missing.
#pragma once

#include <cuda_runtime.h>

namespace kernelforge::kernels {

inline constexpr int kTransposeTileDim = 32;

void transpose_tiled_launch(const float* d_in, float* d_out, int rows, int cols,
                             cudaStream_t stream, dim3& out_grid, dim3& out_block);

float transpose_tiled_run_host(const float* h_in, float* h_out, int rows, int cols);

} // namespace kernelforge::kernels
