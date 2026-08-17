# ADR 0013: cuBLAS Ceiling-Comparison Methodology (Phase 4 GEMM)

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
Spec Part 1.3 (FR2) requires the GEMM ladder to include "cuBLAS ceiling/
reference comparison where available", and states explicitly: **"cuBLAS is
a ceiling/reference, never a target to beat."** This repo's dev machine has
cuBLAS available (`/usr/local/cuda/lib64/libcublas.so.12`, CUDA 12.6
toolkit — verified 2026-08-17), so "where available" resolves to "yes,
measure it." Two decisions are needed before writing any code: (1) how to
correctly call a column-major BLAS library's `sgemm` from this repo's
row-major buffers, and (2) how to keep the "never a target to beat" spec
constraint from silently eroding as ladder numbers accumulate (nothing in
the code itself stops a future benchmark script or README edit from
quietly comparing V4 favorably or unfavorably against cuBLAS as if it were
just another rung).

## Decision
1. **Row-major invocation via the standard transpose-and-swap trick.**
   Reading a row-major M x K buffer as column-major (with leading
   dimension K) yields exactly that buffer's transpose, as a K x M
   column-major matrix (a direct consequence of how row-major and
   column-major storage relate — the memory bytes are identical, only the
   *interpretation* of which axis is contiguous changes). Given row-major
   `C (M x N) = A (M x K) * B (K x N)`, transposing both sides gives `C^T =
   B^T * A^T` — and `B^T`/`A^T`, read this way, are exactly what this
   repo's `B`/`A` buffers already are when reinterpreted column-major. So
   ONE `cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, /*m=*/N, /*n=*/M,
   /*k=*/K, alpha, /*A=*/B_buf, /*lda=*/N, /*B=*/A_buf, /*ldb=*/K, beta,
   /*C=*/C_buf, /*ldc=*/N)` call — A and B swapped, M and N swapped —
   produces exactly the row-major product this repo's other GEMM variants
   compute, with no data reformatting/transposition kernel needed. This is
   standard, well-known cuBLAS usage (not sourced from or copied out of
   any specific external kernel implementation — hard constraint 2 governs
   optimized *kernel* code, not correct invocation of a vendor API's
   documented calling convention), and is unit-tested against the same CPU
   reference every other GEMM variant is (`tests/test_gemm.cpp`).
   `K == 0` is special-cased to a direct `cudaMemset` of `C` to zero rather
   than calling `cublasSgemm` — that call's transformed `ldb` argument
   equals `K`, and cuBLAS rejects `ldb == 0` outright (`CUBLAS_STATUS_
   INVALID_VALUE`, "parameter number 10 had an illegal value" — verified
   empirically while implementing this rung), even though the
   mathematically correct result (a sum over zero terms) is well-defined
   and every other rung's K-loop already produces it by simply not
   executing.
2. **Handle lifecycle: one process-lifetime `cublasHandle_t`, created
   lazily on first use, outside any timed interval.** `cublasCreate()`
   does one-time internal setup unrelated to the SGEMM call being
   measured; timing it would unfairly penalize cuBLAS's reported number
   relative to this repo's other variants, which likewise allocate their
   device buffers once before their own timed loop rather than inside it
   (see `apps/bench_gemm_main.cpp`).
3. **cuBLAS is measured, appears in `benchmarks/raw/gemm.jsonl` and the
   family's benchmark config, and is discussed in `src/kernels/gemm/
   README.md` — but is never included in this ladder's rung-over-rung
   speedup arithmetic (V1→V2→V3→V4), and its `--variant` identifier
   (`cublas_sgemm_ceiling`) says so in its own name.** Every place cuBLAS's
   number is reported states explicitly that it is context, not a rung —
   e.g. "V4 reaches N% of cuBLAS's throughput at this size" is the correct
   framing; "cuBLAS is only Nx faster than V4" or treating cuBLAS as "V5"
   is not. This is the concrete, code-level enforcement of the spec's
   "never a target to beat" instruction: a reader of the benchmark data
   cannot mistake cuBLAS for a laddered variant even without reading this
   ADR, because nothing in the data model presents it as one.

## Consequences
- `apps/bench_gemm_main.cpp`'s `ops_for()` documents cuBLAS's non-rung
  status inline (`description` field), so it is visible in every
  committed `BenchResult.description` too, not just this ADR.
- `gemm_cublas_launch`'s `out_grid`/`out_block` are reported as `(0,0,0)`
  — cuBLAS picks its own internal launch geometry, which this repo has no
  visibility into and does not attempt to reverse-engineer; `(0,0,0)` is
  an explicit "not applicable" sentinel, not a claim that cuBLAS launches
  zero threads.
- Future kernel families that gain a vendor-library ceiling comparison
  (e.g. a cuDNN comparison for softmax/rmsnorm, if ever added — currently
  out of scope, see spec §1.4 non-goals) should follow the same pattern:
  a `*_ceiling` variant name, excluded from ladder-speedup arithmetic,
  documented inline.
