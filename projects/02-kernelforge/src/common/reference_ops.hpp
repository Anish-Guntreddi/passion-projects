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

} // namespace kernelforge::reference
