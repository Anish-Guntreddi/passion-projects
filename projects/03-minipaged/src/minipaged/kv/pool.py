"""BlockPool: the physical block free-list allocator (FR5, FR9).

A fixed-size array of ``PhysicalBlock``s plus a free list. This is the
*only* place physical blocks are created, allocated, shared, or freed --
``BlockTable`` (per-sequence logical-to-physical mapping, ``table.py``)
and ``PagedKVManager`` (ties allocation into the engine, ``manager.py``)
both go through it rather than touching block state directly, which is
what keeps core invariant #1 ("no physical block is simultaneously free
and referenced") checkable in one place.

Four operations, matching FR5's "allocate/free/retain/release" exactly:

* ``allocate()`` -- take a fresh block off the free list, ref_count=1.
  Raises ``BlockPoolExhausted`` if none are free (Phase 2/3 policy for
  what to do about that -- stall-and-retry, or reject at admission --
  lives one layer up, in ``PagedKVEngine``/``SchedulerEngine``, not here;
  this class only ever reports "yes" or "no", it never blocks or waits).
* ``free(block_id)`` -- the sole-owner fast path: requires ref_count==1
  (this caller must be the block's only referrer), sets it to 0, returns
  it to the free list. This is what ``BlockTable`` calls today, since
  Phase 2 has no sharing yet -- every block genuinely is sole-owned, and
  asserting that is a real invariant check, not a formality.
* ``retain(block_id)`` -- ref_count += 1. Unused until Phase 4 (prefix
  sharing introduces a second referrer), defined and tested now so the
  full FR5 interface exists before Phase 4 needs to extend anything.
* ``release(block_id)`` -- ref_count -= 1; returns the block to the free
  list only once ref_count reaches 0. The general decrement operation,
  correct whether or not the block is currently shared -- this is what
  Phase 4 will switch ``BlockTable`` to using once blocks can have more
  than one owner (``free`` requiring ref_count==1 would then start
  raising for legitimately-shared blocks it shouldn't be called on).

Free-list order is a plain Python list used as a LIFO stack (O(1)
allocate/free), which makes allocation order a deterministic function of
allocate/free call order -- required for core invariant #7 (deterministic
traces reproduce identical allocation history under a fixed seed/config).
"""

from __future__ import annotations

from minipaged.kv.block import PhysicalBlock


class BlockPoolExhausted(RuntimeError):
    """Raised by ``allocate()`` when the pool has no free blocks left.

    Distinct from ``ContiguousMemoryModel``'s ``CapacityError``
    (``minipaged.kv.contiguous``): that one means "this request can never
    fit even in an empty pool" (checked before ever touching the pool).
    This one means "the pool is fully committed *right now*" -- a
    transient condition that a caller may reasonably retry once other
    sequences release blocks. See ``PagedKVEngine._can_decode`` /
    ``EventType.DECODE_STALLED`` for how the engine layer turns this into
    a safe, observable "retry next tick" outcome instead of letting it
    propagate as an unhandled exception.
    """


class BlockPool:
    """Fixed-size pool of ``PhysicalBlock``s with a free-list allocator."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._blocks: list[PhysicalBlock] = [PhysicalBlock(block_id=i) for i in range(num_blocks)]
        # LIFO stack: allocate() pops the highest-index free block, free()/
        # release() push back onto the same end. Deterministic given a
        # deterministic call sequence -- see module docstring.
        self._free: list[int] = list(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - self.num_free

    def _validate_block_id(self, block_id: int) -> None:
        """Bounds-check ``block_id`` against ``[0, num_blocks)``.

        Every public method that takes a caller-supplied ``block_id``
        (``block``/``retain``/``release``/``free``) routes through here
        first. Without this, Python's negative-index wraparound lets an
        out-of-range id (e.g. ``-1``) silently alias a *different* real
        block -- and worse, corrupt the free list, since ``free()``/
        ``release()`` push the literal (invalid) ``block_id`` onto
        ``self._free`` rather than the index that was actually touched, so
        a later ``allocate()`` hands the bogus id back out to a caller
        (``BlockTable``) as if it were a real block. ``allocate()`` itself
        needs no check -- it only ever returns ids it generated. See the
        regression tests in ``tests/unit/test_pool.py`` for the concrete
        corruption scenario this closes."""
        if not (0 <= block_id < self.num_blocks):
            raise ValueError(
                f"block_id {block_id} out of range for pool of {self.num_blocks} blocks "
                f"(valid range: [0, {self.num_blocks}))"
            )

    def block(self, block_id: int) -> PhysicalBlock:
        """Read-only lookup of a block's current state, by id."""
        self._validate_block_id(block_id)
        return self._blocks[block_id]

    def allocate(self) -> int:
        """Take a free block; sets its ref_count to 1 (sole owner). Returns
        its ``block_id``. Raises ``BlockPoolExhausted`` if none are free."""
        if not self._free:
            raise BlockPoolExhausted(f"no free blocks (0/{self.num_blocks} available)")
        block_id = self._free.pop()
        block = self._blocks[block_id]
        assert block.ref_count == 0, (
            f"invariant violated: block {block_id} was on the free list with "
            f"ref_count={block.ref_count} (expected 0)"
        )
        block.ref_count = 1
        return block_id

    def retain(self, block_id: int) -> int:
        """Add a second (or later) logical referrer to an already-allocated
        block -- Phase 4 prefix sharing. Raises if the block is currently
        free (nothing to share). Returns the new ref_count."""
        self._validate_block_id(block_id)
        block = self._blocks[block_id]
        if block.ref_count < 1:
            raise ValueError(f"cannot retain free block {block_id} (ref_count=0)")
        block.ref_count += 1
        return block.ref_count

    def release(self, block_id: int) -> int:
        """Drop one logical reference to ``block_id``. Returns the block to
        the free list once ref_count reaches 0 (and only then). Raises if
        the block is already free (double release). Returns the new
        ref_count."""
        self._validate_block_id(block_id)
        block = self._blocks[block_id]
        if block.ref_count < 1:
            raise ValueError(f"block {block_id} is already free -- double release")
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free.append(block_id)
        return block.ref_count

    def free(self, block_id: int) -> None:
        """Sole-owner teardown: ``block_id`` must currently have
        ref_count==1. Sets it to 0 and returns it to the free list.
        Raises ``ValueError`` if the block is shared (ref_count != 1) --
        use ``release`` for that case instead. See module docstring for
        why both exist."""
        self._validate_block_id(block_id)
        block = self._blocks[block_id]
        if block.ref_count != 1:
            raise ValueError(
                f"free() requires sole ownership (ref_count==1); block {block_id} has "
                f"ref_count={block.ref_count} -- use release() for shared blocks"
            )
        block.ref_count = 0
        self._free.append(block_id)
