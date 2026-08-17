// Deterministic CPU reference implementations (FR3: "deterministic
// reference/test utilities"; hard constraint 1: correctness before
// performance). Every GPU kernel in this repo is checked against one of
// these before any timing is trusted.
//
// These are intentionally the simplest possible correct implementation of
// each operation — no blocking, no vectorization, nothing clever. Their
// only job is to be obviously correct by inspection.
#pragma once

#include <cstddef>

namespace kernelforge::reference {

// out[i] = x[i] + y[i], for i in [0, n).
void vector_add(const float* x, const float* y, float* out, std::size_t n);

// SAXPY: y[i] = a * x[i] + y[i], for i in [0, n). Matches BLAS semantics
// (in-place on y). `y_out` may alias `y_in` for a true in-place reference,
// but the GPU kernels/tests here use separate in/out buffers for clarity.
void saxpy(float a, const float* x, const float* y_in, float* y_out, std::size_t n);

// out[c * rows + r] = in[r * cols + c], for r in [0, rows), c in [0, cols).
// `in` is rows x cols row-major; `out` is cols x rows row-major.
void transpose(const float* in, float* out, int rows, int cols);

// Reduction (Phase 2): sum of all n elements. Accumulates in DOUBLE
// precision internally (see reference_ops.cpp), then casts down to float.
// This makes the reference deliberately more accurate than any FP32
// GPU tree-reduction it is compared against, so a correctness check
// isolates "does the kernel compute the right sum" from "double vs float
// summation order/rounding" (hard constraint 1; FR6). n == 0 is legal
// (sum of the empty set is the additive identity, 0).
float reduce_sum(const float* in, std::size_t n);

// Scan (Phase 2): inclusive prefix sum, out[i] = sum(in[0..i]) for
// i in [0, n). Same double-accumulation rationale as reduce_sum. n == 0
// is legal and produces an empty output.
void inclusive_scan(const float* in, float* out, std::size_t n);

// Histogram (Phase 3): hist_out[bin_of(in[i])] += 1 for i in [0, n), where
// bin_of(x) = clamp(floor(x * num_bins), 0, num_bins - 1). hist_out must
// point to num_bins ints; this function zeroes them itself. This bin
// formula is independently re-implemented (not shared/linked code) in
// kernels/histogram/histogram_common.cuh's `histogram_bin_of` for the GPU
// kernels -- the two must agree exactly for "correct" to be well-defined;
// see that header's comment for why they are two separate implementations
// rather than one shared function (same rationale as every other
// reference:: function in this file vs its GPU kernel counterpart).
//
// Throws std::invalid_argument if num_bins <= 0 (hard constraint 8: fail
// loudly). Without this guard, num_bins == 0 makes the clamp formula's
// `bin >= num_bins` branch produce bin = num_bins - 1 = -1, and
// `hist_out[-1] += 1` writes out of bounds -- undefined behavior (heap
// corruption for a zero-size backing allocation, e.g. `std::vector<int>
// ref(num_bins)` in apps/bench_histogram_main.cpp) reachable from a
// malformed `--num_bins 0` (or negative) CLI invocation, before the GPU
// kernels' own `validate_num_bins_fits_shared_mem`
// (kernels/histogram/histogram_common.cuh) ever runs.
void histogram(const float* in, std::size_t n, int* hist_out, int num_bins);

// GEMM (Phase 4): C = A * B. A is M x K row-major, B is K x N row-major,
// C (output) is M x N row-major: C[m*N+n] = sum_k A[m*K+k] * B[k*N+n].
// Accumulates in DOUBLE precision internally (same rationale as
// reduce_sum -- see that function's doc comment), then casts each element
// down to float, so a correctness check isolates "does the kernel compute
// the right dot products" from "fp32 accumulation order/rounding
// differences between this reference's simple row-major triple loop and
// whatever order a given GPU rung accumulates its K-length reduction in
// (sequential in V1/V2, tile-chunked in V3, tile-chunked-with-register-
// reuse in V4)". M == 0, N == 0, or K == 0 are all legal (an M x 0, 0 x N,
// or all-zero-K product is a legitimately empty/zero result; see
// src/kernels/gemm/README.md's edge-size test list) and short-circuit to
// "nothing to compute" for any resulting zero-size dimension.
void gemm(const float* a, const float* b, float* c, int m, int n, int k);

// Softmax (Phase 5): row-wise softmax of an (rows x cols) row-major
// matrix. out[r][c] = exp(in[r][c] - max_c(in[r])) / sum_c(exp(in[r][c] -
// max_c(in[r]))). The max-subtraction is the standard numerically-stable
// formulation (FR2: "numerical stability" is an explicit ladder concern
// for this family, not an afterthought) -- without it, exp() of even
// moderately large inputs overflows float32 before the softmax is ever
// computed, corrupting rows that would otherwise be perfectly well-
// defined. The row-max and the exp-sum are both accumulated in DOUBLE
// precision internally (same rationale as reduce_sum), then the final
// division casts down to float. `cols == 0` (empty row) is legal and
// produces no output for that row; `rows == 0` is legal and is a no-op.
void softmax_rows(const float* in, float* out, int rows, int cols);

// RMSNorm (Phase 5): out[r][c] = in[r][c] / sqrt(mean_c(in[r][c]^2) + eps)
// * gamma[c]. Unlike LayerNorm, RMSNorm does NOT re-center by subtracting
// the row mean first -- it normalizes purely by root-mean-square
// magnitude, which is the whole point of the "RMS" name and the reason
// RMSNorm is cheaper than LayerNorm (one reduction instead of two, and no
// mean to carry through the affine step). `gamma` is the per-column
// (per-feature) learned scale every real RMSNorm layer applies after
// normalizing (there is deliberately no additive "beta"/bias term --
// RMSNorm has none, unlike LayerNorm); pass an all-ones vector of length
// `cols` to get pure normalization with no learned rescaling. The
// mean-of-squares reduction is accumulated in DOUBLE precision internally
// (same rationale as reduce_sum). `cols == 0` (empty row) is legal and
// produces no output for that row; `rows == 0` is legal and is a no-op.
void rmsnorm_rows(const float* in, const float* gamma, float* out, int rows, int cols, float eps);

} // namespace kernelforge::reference
