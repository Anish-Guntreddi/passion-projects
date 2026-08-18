# ADR 0011: V4 Fused Kernel Design (One Thread Per Query Row, `kFusedTileDim=32`, `kMaxHeadDimFused=128`)

- **Status:** Accepted (tile-size/register-pressure choices provisional --
  see "Revisit" below, matching ADR 0007's own framing)
- **Date:** 2026-08-17
- **Decision driver:** Roadmap Phase 5 ("Full attention-matrix
  materialization removed; output accumulated tile-by-tile" -> exit
  criterion "Peak-memory scales as designed; output matches reference").
  Spec hard constraints 1 and 3 apply directly here: "never copy the
  FlashAttention production implementation wholesale" and "keep the
  optimized implementation readable."

## Context
V4 (`src/flashlite/cuda_fused/`) is the single largest design decision in
this project: unlike every earlier variant (which changes exactly one
mechanism versus its predecessor), V4 has to combine V2's tiling, V3's
online-softmax combine identity, AND a completely different kernel
structure (no intermediate scores buffer to hand off between separate
kernel launches) into one kernel. Several structurally different
parallelization strategies were possible (e.g. one warp per query row with
warp-shuffle reductions; one thread per output element with cross-thread
shared-memory softmax state, mirroring V1/V2/V3's kernel-2 structure but
fused; a two-dimensional thread block covering multiple query rows AND
head-dim columns per thread, closer to V2's own kernel-1/3 tile shape).

## Decision
**One thread per query row.** Each thread block covers `kFusedTileDim`
query rows (a "Q tile"); each thread within the block owns exactly one
query row's entire computation -- its own private online-softmax state
(`m`, `l`) and its own private, full-`D`-length output accumulator
(`acc[kMaxHeadDimFused]`), for the whole kernel's duration. The block loops
sequentially over K/V in `kFusedTileDim`-row tiles, cooperatively staging
each tile into shared memory once per iteration (reused by every thread in
the block, the same cooperative-load idea V2/V3's kernel 1/3 already use),
and each thread updates its own private state against that shared tile
before the block moves to the next one.

**Why this over the alternatives:**

1. **No cross-thread communication needed for the online-softmax update
   itself.** Because one thread owns one row's ENTIRE running `(m, l,
   acc)` state, applying the online recurrence per key column
   (`docs/online-softmax.md` SS3/SS5) is a purely sequential, single-thread
   operation -- no `__syncthreads()`-gated reduction is needed for the
   softmax math itself (only for the K/V tile's cooperative shared-memory
   load/reuse, exactly as in V2/V3). A one-thread-per-output-element or
   cross-thread-reduction design would need to re-derive V3's per-block
   combine step (`online_softmax_rows_kernel`'s Phase B) INSIDE the fused
   kernel's tile loop, once per KV tile, adding real complexity for no
   clear benefit at this project's scale -- directly against hard
   constraint 3 ("keep the optimized implementation readable").
2. **Directly reuses the exact recurrence already proven correct on CPU.**
   `src/flashlite/online_softmax.py`'s `online_softmax_attention_row`
   (tested independently, `tests/math/test_online_softmax.py`, BEFORE any
   CUDA kernel existed) is already written as "one sequential stream of
   `(m, l, acc)` updates, one tile/element at a time" -- the one-thread-
   per-row design is a direct, line-for-line translation of that exact
   function into CUDA, not a re-derivation under a different parallel
   structure. This is what makes the kernel's correctness argument traceable
   back through V3's design and Phase 4's independent unit tests, rather
   than needing a fresh proof.
3. **Consistent with this repo's kernel-3 precedent.** V2/V3's kernel 3
   (`weighted_sum_tiled_kernel`) already uses "each thread owns one output
   element and accumulates across a tiled reduction loop" -- the
   one-thread-per-row design is the direct extension of that same idea to
   also own the softmax normalization, not a new parallelization philosophy
   introduced only for V4.

## Concrete constants
- **`kFusedTileDim = 32`**: both the Q-rows-per-block and
  KV-rows-per-streamed-tile dimension. Reuses V2's `kAttnTileDim` /
  V3's `kOnlineSoftmaxAttnTileDim` value, for the identical reasons ADR
  0007 already gives (this GPU's 32-thread warp size; consistency across
  this repo's sibling tiled kernels) -- not re-derived from scratch, since
  nothing about V4's design changes the argument for why 32 is a reasonable
  starting point. Two `kFusedTileDim x D` shared-memory tiles (K and V)
  cost `2 * 32 * 128 * 4 = 32,768` bytes at the largest supported
  `head_dim` (128) -- comfortably inside this GPU's 49,152-byte default
  per-block shared-memory budget (`docs/gpu-model.md`, `../02-kernelforge`),
  the same margin-of-safety reasoning ADR 0007 gives for V2.
- **`kMaxHeadDimFused = 128`**: each thread's private Q-row cache
  (`q_reg[kMaxHeadDimFused]`) and output accumulator (`acc[kMaxHeadDimFused]`)
  must be fixed-size arrays (CUDA device-code array sizes are compile-time
  constants); 128 exactly covers this repo's full test matrix (spec SS1.7:
  "head_dim 32/64/128"). `head_dim > 128` is rejected with a specific,
  documented `TORCH_CHECK` failure in `cuda_fused/bindings.cpp`
  ("unsupported shapes must fail clearly," spec SS1.7) rather than silently
  truncating or reading out of bounds.

## Consequences and accepted tradeoffs (explicitly deferred to Phase 6)
- **Register pressure is real and not yet tuned.** Two `[128]`-element
  float arrays per thread (`q_reg`, `acc`) reserve substantial register (or
  spilled local-memory) capacity per thread regardless of the ACTUAL
  runtime `head_dim` (the compiler cannot generally prove a tighter bound
  from a runtime-valued loop trip count against a fixed-size array) -- this
  is an accepted Phase 5 characteristic, matching ADR 0007's own framing
  for V2's tile size ("a value that is correct and reasonably justified,
  not the tuned optimum"). Phase 6 (roadmap: "Tile-size/block-shape sweep
  ... profile occupancy" -> "Tuned defaults based on benchmark evidence,
  not folklore") is where this gets measured and, if warranted, addressed
  (e.g. a `head_dim`-templated kernel instantiation) -- not attempted here,
  since doing so now would trade Phase 5's readability requirement for a
  performance claim this phase does not need to make (spec: "the project
  succeeds even if the custom kernel does not beat the platform's built-in
  attention").
- **No causal-tile skip beyond the per-row early exit.** The kernel skips
  an entire KV tile's inner compute loop when `causal && tile_start > row`
  for a given thread, and breaks out of a tile's inner loop early once
  `j > row`, but still performs the tile's cooperative shared-memory LOAD
  unconditionally (since other rows in the same block may still need it).
  A more aggressive design could reorder or skip whole tiles at the block
  level when EVERY row in the block has already passed them, but that adds
  complexity for a benefit that depends on the tile/row-tile alignment
  (again, a Phase 6 tuning question under spec D5, not a Phase 5 defect).
- **What this design achieves, verified, not just argued:**
  `cuda_fused/bindings.cpp`'s `attention_fused_forward` allocates only
  `out` (`[B, H, S, D]`) -- there is no `[B, H, S, S]` scores tensor
  anywhere in the Python-facing entry point for this variant, unlike V1/V2/V3.
  `tests/correctness/test_fused_attention.py::test_fused_peak_memory_grows_far_slower_than_materializing_variants`
  measures `torch.cuda.max_memory_allocated()` directly and checks the
  V3-vs-V4 peak-memory ratio widens substantially and monotonically as
  `seq_len` grows -- the literal, measured version of this ADR's design
  claim, not merely the claim itself.

## Revisit
Same posture as ADR 0007: `kFusedTileDim` and the register-pressure
tradeoff above are provisional. D5 (spec: "Shared-memory vs.
register-pressure tile constraints on the available GPU -> resolved
empirically in Phase 6") remains formally unresolved for V4 specifically,
not just V2, until Phase 6's sweep. If that sweep finds different values
measurably better for V4, this ADR is superseded, not edited in place.
