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

} // namespace kernelforge::reference
