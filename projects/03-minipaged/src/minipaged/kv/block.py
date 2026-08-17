"""PhysicalBlock: the unit of physical KV storage in the paged allocator.

Per FR9, ``PhysicalBlock`` is one of the public data structures defined
before implementation. It is deliberately tiny: an id and a reference
count. Everything else -- which sequence(s) point at a block, in what
logical order -- lives one layer up in ``BlockTable`` (``table.py``) and
``BlockPool`` (``pool.py``); a ``PhysicalBlock`` does not know who
references it, only how many referrers there currently are.

``ref_count`` is present starting now (Phase 2), even though Phase 2 has
no sharing yet (every block has exactly one owner, so ``ref_count`` is
always 0 or 1 in practice until Phase 4). FR5 calls out "refcount
lifecycle" as part of the paged KV subsystem's public interface, and
core invariant #1 ("no physical block is simultaneously free and
referenced") is stated in terms of ref_count -- defining the field now,
correctly, is what lets that invariant be a real, literal test starting
Phase 2 instead of waiting for Phase 4's prefix sharing to introduce it
retroactively. See ``docs/decisions/0003-block-size-tokens-per-block.md``
and ``docs/kv-memory.md`` for how this fits into the paged allocator as a
whole.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PhysicalBlock:
    """One fixed-size slot of physical KV storage.

    ``block_id`` is this block's stable index into ``BlockPool``'s
    backing array -- never reused for a different logical identity, only
    recycled (freed and reallocated) by the pool. ``ref_count`` is the
    number of active logical references to this block: 0 means free (on
    the pool's free list, not backing any sequence's data), >=1 means
    allocated. Phase 2: always exactly 1 while allocated (single owner).
    Phase 4 (prefix sharing) is what makes >1 reachable, when multiple
    sequences' block tables point at the same physical block.
    """

    block_id: int
    ref_count: int = 0

    @property
    def is_free(self) -> bool:
        """True iff nothing currently references this block (core invariant #1
        restated as a predicate: this must be exactly the condition under
        which ``block_id`` is allowed to sit on ``BlockPool``'s free list)."""
        return self.ref_count == 0
