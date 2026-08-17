"""Property tests for the paged allocator (``BlockPool`` + ``BlockTable``,
via ``PagedKVManager``), the Phase 2 analogue of
``tests/property/test_kv_memory_invariants.py``. Exercises core
invariants #1 ("no physical block is simultaneously free and
referenced"), #3 ("a sequence's logical block count covers its KV
length"), and #4 ("releasing a sequence eventually returns all unshared
blocks") across arbitrary, possibly-invalid, possibly-conflicting
operation sequences.
"""

from __future__ import annotations

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from minipaged.kv.manager import PagedKVManager

NUM_BLOCKS = 16
BLOCK_SIZE = 4


@dataclasses.dataclass(frozen=True)
class _AdmitOp:
    request_id: str
    prompt_len: int


@dataclasses.dataclass(frozen=True)
class _GrowOp:
    request_id: str
    num_tokens: int


@dataclasses.dataclass(frozen=True)
class _ReleaseOp:
    request_id: str


def _ops_strategy():
    request_ids = st.integers(min_value=0, max_value=5).map(lambda i: f"req-{i}")
    admit = st.builds(
        _AdmitOp,
        request_id=request_ids,
        prompt_len=st.integers(min_value=1, max_value=NUM_BLOCKS * BLOCK_SIZE + 8),
    )
    grow = st.builds(
        _GrowOp,
        request_id=request_ids,
        num_tokens=st.integers(min_value=0, max_value=NUM_BLOCKS * BLOCK_SIZE + 8),
    )
    release = st.builds(_ReleaseOp, request_id=request_ids)
    return st.lists(st.one_of(admit, grow, release), min_size=0, max_size=40)


@given(ops=_ops_strategy())
@settings(max_examples=200)
def test_random_admit_grow_release_sequences_never_violate_paged_invariants(
    ops: list[_AdmitOp | _GrowOp | _ReleaseOp],
) -> None:
    """Whatever sequence of (possibly-invalid, possibly-conflicting) admit/
    grow/release operations is thrown at the manager: the pool never over-
    commits (#1: allocated blocks conserved, free+allocated == total,
    no block ever double-counted), every active request's block table
    always covers its most recently grown length (#3), and draining every
    remaining active request returns all capacity with no leaks (#4)."""
    manager = PagedKVManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    live: dict[str, int] = {}  # request_id -> current_len, for currently-admitted requests

    for op in ops:
        if isinstance(op, _AdmitOp):
            if op.request_id in live:
                continue  # would raise "already admitted" -- not this property's concern
            if manager.admit(op.request_id, op.prompt_len):
                live[op.request_id] = op.prompt_len
        elif isinstance(op, _GrowOp):
            if op.request_id not in live:
                continue  # can't grow a request that was never admitted
            if op.num_tokens < live[op.request_id]:
                continue  # BlockTable.grow_to only ever grows; not this property's concern
            if manager.can_grow_to(op.request_id, op.num_tokens):
                manager.grow(op.request_id, op.num_tokens)
                live[op.request_id] = op.num_tokens
            # else: correctly left unchanged -- exactly what can_grow_to promises
        else:
            if op.request_id in live:
                manager.release(op.request_id)
                del live[op.request_id]

        # Invariants that must hold after every single operation.
        pool = manager.pool
        assert pool.num_free + pool.num_allocated == pool.num_blocks
        assert pool.num_free >= 0
        assert pool.num_allocated == sum(
            manager.blocks_needed_for(length) for length in live.values()
        )
        for request_id, length in live.items():
            assert manager.is_admitted(request_id)
            assert manager.capacity_tokens_for(request_id) >= length  # core invariant #3

    # Draining every remaining live request returns all capacity: no leaks
    # (core invariant #4).
    for request_id in list(live):
        manager.release(request_id)
    assert manager.pool.num_free == NUM_BLOCKS
    assert manager.pool.num_allocated == 0
