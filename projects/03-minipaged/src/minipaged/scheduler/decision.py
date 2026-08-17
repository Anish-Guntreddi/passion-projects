"""SchedulerDecision: one scheduling step's outcome (FR9).

FR9 names ``SchedulerDecision`` explicitly as a public interface to be
defined before implementation: "selected requests, prefill/decode work,
token budget, expected KV growth, preemptions/rejections if implemented."
Every named field is present below: selected requests (``admitted``/
``decoded``), prefill/decode work (``admitted``/``decoded`` again, plus
``prefill_tokens_used``), token budget (``prefill_token_budget``),
expected KV growth (``expected_kv_growth_blocks``), and preemptions/
rejections (``rejected``/``cancelled``/``stalled`` -- no preemption is
implemented in Phase 3, so only the rejection half is populated).
``SchedulerEngine`` (``scheduler.py``) appends one of these to
``self.decisions`` per ``step()`` call -- the "scheduling timeline" that
is the Phase 3 exit criterion's deliverable ("Runtime replays arrivals
and produces scheduling timelines").
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SchedulerDecision:
    """The outcome of exactly one ``SchedulerEngine.step()`` call.

    Request-id tuples are all disjoint by construction *within one tick*
    (a request cannot be, say, both ``rejected`` and ``cancelled`` in the
    same admission attempt) except across ticks, where the same request
    id can legitimately reappear across multiple ``rejected`` entries
    before eventually appearing in ``admitted`` or ``cancelled``.

    ``stalled`` exists for structural parity with FR9's "preemptions/
    rejections if implemented" and because ``SchedulerEngine`` inherits
    Phase 2's ``_can_decode`` stall mechanism unchanged -- but Phase 3's
    admission ledger (see ``docs/decisions/0005-...md``) is specifically
    designed to make this field always empty; see
    ``tests/property/test_scheduler_invariants.py`` for the proof.
    """

    tick: float
    admitted: tuple[str, ...]  # newly admitted (prefilled) this tick
    rejected: tuple[str, ...]  # soft: insufficient ledger capacity or prefill budget, retried later
    cancelled: tuple[str, ...]  # hard: worst-case can never fit total KV capacity or step budget
    decoded: tuple[str, ...]  # running sequences that received a decode step this tick
    stalled: tuple[str, ...]  # decode attempted but paged growth failed (see class docstring)
    completed: tuple[str, ...]  # sequences that finished this tick
    prefill_tokens_used: int  # sum of prompt_len over `admitted` this tick
    prefill_token_budget: int  # the step's configured token budget (D3)
    kv_free_blocks: int  # PagedKVManager.pool.num_free after this tick's admission + decode
    kv_allocated_blocks: int  # PagedKVManager.pool.num_allocated after this tick
    expected_kv_growth_blocks: int  # worst-case future growth still reserved but not yet
    # physically allocated, for every currently admitted/running sequence: the admission
    # ledger's `reserved_tokens` (each sequence's blocks_needed_for(prompt_len +
    # max_new_tokens) watermark, summed) minus `kv_allocated_blocks` (what's actually
    # backed by a PhysicalBlock right now). This is the FR9-named "expected KV growth"
    # field: how many more blocks the scheduler has already committed capacity for, in
    # the worst case, before any of the currently running sequences can complete --
    # exactly the gap that Decision 0005's ledger reserves precisely so it can never be
    # exceeded (see the module docstring's `sum(watermark_i) <= num_blocks` argument).
    # Always >= 0: a sequence's physical usage never exceeds its own ledger watermark.

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)
