# ADR 0009: V3 Online-Softmax Kernel Design (Combined-Pass, Not Full Fusion)

- **Status:** Accepted
- **Date:** 2026-08-17
- **Decision driver:** Roadmap Phase 4 ("Running max + normalization
  accumulator implemented/tested independently, then integrated" -> exit
  criterion "Tests cover extreme score values and multiple tile
  boundaries"). Once `docs/online-softmax.md`/`src/flashlite/online_softmax.py`
  establish the running-max/running-sum combine identity is correct on CPU
  (tests/math/test_online_softmax.py), a concrete decision is needed for
  how that identity is realized as a CUDA kernel, and how much of V2's
  pipeline it changes.

## Context
Two materially different designs were available for V3:

1. **Combined-pass, materialization unchanged**: keep V2's kernel 1
   (`compute_scores_tiled_kernel`) and kernel 3 (`weighted_sum_tiled_kernel`)
   exactly as they are -- the full `[B, H, S, S]` score/probability matrix
   is still written to and read from global memory -- and replace ONLY
   kernel 2's algorithm: instead of two separate full passes over each row
   (find the max, then sum `exp(x - max)`), fold max-finding and
   sum-accumulation into ONE combined pass using the online update rule,
   via a per-thread private fold followed by a block-level tree-reduction
   combine (both instances of the identical combine identity,
   `docs/online-softmax.md` SS3).
2. **Full fusion**: go straight to computing `Q K^T` scores tile-by-tile
   inside the softmax/output kernel itself, with no materialized scores
   buffer at all -- i.e., build what became V4 directly as "V3."

## Decision
**Design 1.** V3 changes only kernel 2's algorithm (two-pass -> one
combined max+sum pass, plus the always-needed final normalize-and-write
pass -- three total row-length global-memory passes reduced to two).
Kernels 1 and 3 are duplicated, byte-for-byte, from V2
(`attention_online_softmax.cu`'s header explains why duplication, not
sharing, per "variants live side-by-side," spec Part 2). The full
`[B, H, S, S]` matrix is still materialized between kernels.

Three reasons, matching this repo's established "one optimization variable
per variant" discipline (`attention_tiled.cuh`'s own header, ADR 0007's
framing):

1. **The roadmap's own phase boundary requires it.** Phase 4's exit
   criterion is about the running-max/running-sum algorithm's correctness
   ("tests cover extreme score values and multiple tile boundaries"), not
   about memory scaling. Phase 5's exit criterion is explicitly the memory
   claim ("Peak-memory scales as designed"). Building full fusion directly
   as "V3" would collapse two roadmap phases -- each with its own exit
   criterion and its own kind of evidence (Phase 4: numerical correctness
   across tile boundaries; Phase 5: measured peak-memory scaling) -- into
   one kernel, making it impossible to demonstrate the online-softmax
   ALGORITHM is correct independently of also getting the memory-layout
   fusion correct at the same time. Design 1 isolates exactly one variable;
   Design 2 would change two at once (softmax algorithm AND materialization)
   in a single kernel, which is precisely what this project's "one
   optimization variable per variant" convention exists to avoid.
2. **Matches this repo's own forward-reference documentation.**
   `docs/attention-math.md` SS5 (written during Phase 3, before Phase 4
   existed) already states the intended split: "V3 (online softmax, Phase
   4) changes how the softmax normalization is accumulated ... V4 (fused,
   Phase 5) combines V2's tiling and V3's online softmax so the full S x S
   matrix is never materialized at all." Design 1 is what makes that
   sentence literally true; Design 2 would make it false (V3 would already
   have removed materialization, leaving nothing distinct for "V4" to
   combine).
3. **A smaller, more isolated change is easier to verify.** Testing "does
   the online update reduce the same row to the same normalized
   probabilities as the two-pass version" (V3, `test_online_softmax_attention.py`)
   is a strictly narrower correctness claim than "does an entirely new
   single-kernel pipeline, with a new grid/block/shared-memory design AND a
   new softmax algorithm, reproduce the reference" (V4). Doing the former
   first, and separately, is lower-risk and produces a working intermediate
   checkpoint even if V4's larger design change (thread-per-row, KV-tile
   streaming) needs more iteration.

## Kernel-level choices within Design 1
- **`kOnlineSoftmaxBlockSize = 256`**, identical to V1/V2's
  `kSoftmaxBlockSize`. The online algorithm's per-thread "tile" IS that
  thread's strided subset of the row (`j = tid, tid+256, tid+512, ...`);
  reusing the existing block size keeps the reduction's power-of-two
  halving logic (already proven correct in V1/V2) unchanged, and keeps the
  comparison between V2's and V3's kernel-2 latency an apples-to-apples
  "same parallelism, different algorithm" comparison rather than
  conflating a block-size change with the algorithm change. No new tile-
  size ADR is needed for this constant specifically because it is not a
  new decision -- it is a deliberate non-change.
- **`-FLT_MAX` sentinel, not literal `-INFINITY`**, for "no data folded in
  yet," matching V1/V2's `softmax_rows_kernel`'s own existing convention
  (`float local_max = -FLT_MAX;`). `docs/online-softmax.md` SS10 and
  `attention_online_softmax.cu`'s kernel-2 comment both explain this is a
  deliberate, float32-hardware-specific choice (true IEEE `-inf - -inf` is
  `NaN`, not a safe zero) -- NOT an inconsistency with
  `src/flashlite/online_softmax.py`'s use of true `-math.inf` (Python
  float64 arithmetic handles that case safely, so no sentinel workaround is
  needed there).

## Consequences
- V3's benchmark story (`benchmarks/methodology.md`) is specifically about
  kernel 2's global-memory traffic: 2 passes over the row instead of 3, a
  measurable ~33% reduction in that one kernel's reads, while kernels 1/3
  (and hence the majority of total per-call latency at larger shapes, per
  `docs/io-analysis.md` SS7's finding that kernel 2 plus shared overhead
  dominates once tiling has already been applied) are unaffected by
  design -- so V3's end-to-end speedup over V2 is expected to be modest,
  not dramatic, and that is not a defect: the dramatic change is V4's,
  covered by ADR 0011.
- `tests/correctness/test_online_softmax_attention.py`'s
  `test_online_softmax_correct_when_true_max_is_in_a_late_wave` test exists
  specifically because `kOnlineSoftmaxBlockSize=256` means the smaller
  shapes already covered by the shape-matrix test (`seq_len <= 257`) never
  cross more than one 256-element "wave" boundary within a single thread's
  strided loop -- this repo's own established shape sweep is not, by
  itself, sufficient coverage for Design 1's specific reduction structure,
  so a dedicated larger-`seq_len` test was added rather than assuming the
  existing sweep already covered it.
