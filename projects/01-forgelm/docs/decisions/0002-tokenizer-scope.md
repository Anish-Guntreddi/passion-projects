# ADR 0002: Tokenizer scope — minimal byte-level BPE (D1)

**Status:** Accepted (adopts the spec's recommended default)
**Date:** 2026-08-17
**Open decision:** D1 (`01-forgelm-spec.md` §1.9) — *"Tokenizer complexity vs
schedule → default: minimal byte-level BPE, ~500 LOC ceiling."*

## Decision

Implement a from-scratch **byte-level BPE** tokenizer
(`forgelm/tokenizer/bpe.py`, `ByteLevelBPETokenizer`):

- Base vocabulary = the 256 raw byte values. No `<unk>` token is needed —
  every UTF-8 string is representable, so there is zero out-of-vocabulary
  risk for any input text, training or generation time alike.
- Training learns merges via the standard Sennrich et al. (2016) algorithm:
  repeatedly merge the most frequent adjacent symbol pair, over
  pre-tokenized "words" (letters / digits / punctuation-runs /
  whitespace-runs) so merges never cross a word boundary. Ties are broken
  deterministically by pair value, so training is reproducible from the
  same corpus.
- Three reserved special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`) occupy the
  top of the id space and are recognized as literal substrings during
  `encode()`.
- Persisted as JSON (merge list + config only — the full byte-vocab is
  rederived deterministically from the merge list on load).

Implementation landed at **292 lines** in `bpe.py` (measured with `wc -l`,
including docstrings/comments), well inside the 500 LOC ceiling.

## Alternatives considered

- **tiktoken / sentencepiece / HuggingFace `tokenizers`**: rejected outright
  by the engineering quality bar (§1.7: "no copied full model implementation
  from a framework") and by the portfolio-wide rule against wholesale
  copying from mature libraries — using one of these wouldn't demonstrate
  *how* a tokenizer works, which is the point of this project.
- **Word-level / whitespace tokenizer**: simpler, but produces a large
  vocabulary with real OOV risk, and doesn't showcase subword mechanics —
  BPE is the standard technique worth demonstrating for an LLM-from-scratch
  portfolio piece.
- **Full GPT-2/GPT-4-style regex + byte-to-unicode display remapping**: the
  byte-to-unicode remapping (mapping raw bytes to printable Unicode
  characters purely for human-readable vocab files) is a cosmetic detail
  that adds real complexity without changing correctness or the educational
  value of the *algorithm*; skipped. Token ids are the only thing that
  matters downstream (model embeddings), and this tokenizer works directly
  on raw byte ids.

## Consequences

- Training is O(merges × corpus-word-count) per merge (full pair-count
  recompute each iteration) rather than using an incremental
  priority-queue update — intentionally simpler and slower. Acceptable
  because Phase 1's corpora are small (KBs–low hundreds of KB) and
  "correctness before optimization" (agent execution rule 3) explicitly
  applies. Revisit only if a later phase's corpus size makes this the
  bottleneck.
- Because it's byte-level, `decode(encode(text)) == text` exactly for any
  text — critically, this is what makes the Phase 1 exit criterion
  ("training batches... reconstructable") true: `decode(train_ids +
  val_ids) == original_text` always, regardless of corpus content, since
  concatenating the two id lists reproduces the exact original encoded
  sequence.

## Known limitation: interior windows and multi-byte characters

Merges group raw *bytes*, not Unicode characters, so a learned token can
represent a byte span that starts or ends mid multi-byte UTF-8 character
(e.g. a merge of the last ASCII byte before a curly quote with that quote's
first UTF-8 byte). Decoding an arbitrary *interior* slice of ids — one that
doesn't start at position 0 or end at the true end of a sequence produced
by `encode()` — is therefore only guaranteed to succeed for
single-byte-per-character text (ASCII). This is a well-known, accepted
property of byte-level BPE in general (the same reason streaming
detokenizers for GPT-style models buffer partially-decoded output), not a
bug specific to this implementation. Decoding the *full* sequence a split
produced (`decode(train_ids + val_ids)`, not `decode(train_ids) +
decode(val_ids)`) sidesteps it entirely, since position 0 and the true end
are always valid character boundaries. `tests/integration/test_pipeline.py`
verifies exact reconstruction this way on the real (non-ASCII-only)
example corpus; `tests/unit/test_dataset.py` additionally verifies
interior-window decoding on the ASCII-only `tiny_corpus` fixture, where it
is unconditionally safe.
