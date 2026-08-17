# Example corpus

`alice_in_wonderland.txt` is *Alice's Adventures in Wonderland* by Lewis
Carroll (published 1865), downloaded unmodified from Project Gutenberg:

- Source: <https://www.gutenberg.org/files/11/11-0.txt>
- Retrieved: 2026-08-17
- License: public domain in the United States (author died 1898; work
  predates any copyright term that could still apply). Project Gutenberg's
  own header/footer markers (`*** START/END OF THE PROJECT GUTENBERG EBOOK
  11 ***`) are kept intact in the committed file, per Project Gutenberg's
  redistribution terms (<https://www.gutenberg.org/policy/permission.html>).

This satisfies open decision **D2** ("small, legal, reproducible dataset;
document license") for Phase 0/1 development and tests: it is small enough
to commit directly (~150 KB) and train a tokenizer / build a dataset on
without any network access at test time, which keeps CI fully offline and
deterministic. See `docs/decisions/0003-dataset-choice.md` for the full
rationale — including that ADR's "Phase 5 resolution" section: this same
corpus (84,583 tokens once tokenized) turned out sufficient for the Phase
5 scaling experiment's toy-to-small compute budget too, so no larger
corpus was ever downloaded.

`../tests/fixtures/` and `../tests/conftest.py` additionally define a tiny
synthetic string corpus for sub-second unit tests that don't need real
prose.
