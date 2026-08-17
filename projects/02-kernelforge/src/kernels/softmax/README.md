# Softmax Ladder (Phase 5)

Row-wise softmax of an (rows x cols) row-major FP32 matrix. Three GPU
variants, each changing exactly one variable from the previous rung (hard
constraint 4), all validated against the same CPU reference
(`kernelforge::reference::softmax_rows`, `src/common/reference_ops.cpp` —
max-subtracted and double-accumulated for accuracy/stability) before any
timing is trusted (hard constraint 1). Every rung's hypothesis below was
written into that rung's `.cuh` header **before** its `.cu`
implementation, per `docs/optimization-method.md`'s loop and hard
constraint 10.

Spec FR2 frames this family's ladder concerns as "reduction strategy,
memory traffic, bounded fusion, numerical stability" — this ladder's three
rungs are exactly the first three, in that order; numerical stability is
the fourth and is discussed separately below because it is held constant
across all three, not itself a distinguishing rung.

| Rung | File | What changes vs. previous rung | Hypothesis (full text in the `.cuh` header) |
|---|---|---|---|
| V1 `v1_naive` | `softmax_naive.cuh/.cu` | — (baseline) | One block per row, shared-memory tree reductions (max, then sum), three full global-memory passes over each row. "Simple CUDA first." |
| V2 `v2_warp_shuffle` | `softmax_warp_shuffle.cuh/.cu` | Reduction mechanism | Same three-pass structure; block-wide max/sum reductions use `__shfl_down_sync` (ADR 0004) instead of a shared-memory tree — fewer `__syncthreads()` barriers, no change in memory traffic. |
| V3 `v3_fused_online` | `softmax_fused_online.cuh/.cu` | Pass count / memory traffic | Online-softmax fusion: max and exp-sum computed together in ONE pass via a streaming rescale (`m_new=max(m,x); l_new=l*exp(m-m_new)+exp(x-m_new)`), cutting global reads from three to two. Reduction mechanism (warp shuffle) carries over unchanged from V2. |

## Numerical stability (held constant, not a ladder variable)

Every rung subtracts the row max before exponentiating — without it,
`exp()` of even moderately large inputs overflows FP32 (`exp(89)` already
exceeds FP32's ~3.4e38 max) before softmax is ever computed. This is a
correctness *prerequisite* present identically in V1, V2, and V3, the same
way GEMM's coalesced-read mapping is present in V2-V4 rather than being
its own separate rung. `tests/test_softmax.cpp`'s
`LargeMagnitudeInputStaysNumericallyStable` exercises input in `[-1000,
1000)` specifically to make this mechanism's necessity concrete (without
max-subtraction, this input would produce `inf`/`NaN` well before reaching
the reference's own — identically max-subtracting — output).

## V3's "bounded" fusion

V3 fuses the max and exp-sum passes but still makes a SECOND pass to
normalize and write output — it does not attempt to cache an entire row in
shared memory to also eliminate that second pass, since that would only be
valid for rows small enough to fit on-chip (bounded by this GPU's 49,152
shared-memory bytes/block), silently failing or falling back for larger
`cols` otherwise. This kernel stays correct for ANY `cols` by design; a
shared-memory-row-caching variant for rows that fit is explicit future
work, not silently included or omitted.

## Correctness

`tests/test_softmax.cpp` checks all three variants against
`reference::softmax_rows` on: rows = 0, cols = 0 (both legal, FR6), cols =
1 (softmax is trivially 1.0 regardless of the single value), small
non-power-of-two cols, cols exactly one block, cols larger than one block
(multi-stride), cols one more than a block multiple, a large-magnitude
input (numerical-stability case above), a realistic transformer-activation
shape (2048 x 768), a block-size sweep, and multiple seeds. See that file
for the exact cases and tolerances used.

## Benchmarks

`benchmarks/configs/softmax.json` sweeps all three variants across
representative transformer hidden/feature-dimension widths (`cols`, from
128 through 8192) at a fixed, representative `rows=4096`; results are in
`benchmarks/raw/softmax.jsonl` (21 records, all `correctness_passed:
true`). See `benchmarks/methodology.md` §12 for the full as-run table and
interpretation. Headline, and it is a genuinely mixed one, reported as
measured: **V1→V2 (reduction mechanism) is a real win at small-to-medium
`cols`** (up to 1.51x at cols=128), **fading to parity by cols=4096-8192**
(fewer `__syncthreads()` rounds is a fixed saving whose share of total
time shrinks as work-per-thread grows) — but **V2→V3 (the "bounded
fusion" rung) does NOT show a consistent win: at six of the seven sizes
tested, V3 is the same speed or measurably slower than V2**, contradicting
the hypothesis in `softmax_fused_online.cuh`. See §12 for the full
per-size table and a plausible (not yet profiler-confirmed) mechanism:
V3's online-combine issues two `expf()` calls per element versus V1/V2's
one `expf()` call total across their two separate passes, so this rung
trades a memory-traffic saving for a transcendental-throughput cost that,
at these row sizes, is a wash or a net loss rather than the assumed win.
