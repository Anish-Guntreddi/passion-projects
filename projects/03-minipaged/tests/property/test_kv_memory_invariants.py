"""Property tests for ``ContiguousMemoryModel``, per spec S1.9: "random
allocate/free sequences conserve blocks; no leaks; shared references
safe" (the "shared references" clause is Phase 4 prefix-sharing scope;
"conserve / no leaks" applies to Phase 1's reservation model today).

Core invariant #4 ("releasing a sequence eventually returns all unshared
blocks") and the Phase 1 analogue of invariant #5 ("scheduler never
admits work requiring more KV capacity than policy permits" -> "the model
never lets reserved_tokens exceed capacity_tokens") are both exercised
here across arbitrary, possibly-invalid, possibly-conflicting operation
sequences rather than the hand-picked ones in
``tests/unit/test_contiguous.py``.
"""

from __future__ import annotations

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from minipaged.kv.contiguous import CapacityError, ContiguousMemoryModel

CAPACITY = 64


@dataclasses.dataclass(frozen=True)
class _ReserveOp:
    request_id: str
    size: int
    used: int


@dataclasses.dataclass(frozen=True)
class _ReleaseOp:
    request_id: str


def _ops_strategy():
    request_ids = st.integers(min_value=0, max_value=7).map(lambda i: f"req-{i}")
    reserve = st.builds(
        _ReserveOp,
        request_id=request_ids,
        size=st.integers(min_value=1, max_value=CAPACITY + 16),  # sometimes deliberately too big
        used=st.integers(min_value=0, max_value=CAPACITY + 16),
    ).filter(lambda op: op.used <= op.size)
    release = st.builds(_ReleaseOp, request_id=request_ids)
    return st.lists(st.one_of(reserve, release), min_size=0, max_size=40)


@given(ops=_ops_strategy())
@settings(max_examples=200)
def test_random_reserve_release_sequences_never_violate_capacity_invariants(
    ops: list[_ReserveOp | _ReleaseOp],
) -> None:
    """Whatever sequence of (possibly-invalid, possibly-conflicting)
    reserve/release operations is thrown at the model, it never ends up
    over-committed, ``used`` never exceeds ``reserved``, and every
    successfully-reserved request is releasable for exactly what it
    reserved -- draining every live reservation always returns all
    capacity (no leaks)."""
    model = ContiguousMemoryModel(capacity_tokens=CAPACITY)
    live: dict[str, int] = {}  # request_id -> size reserved, for currently-held requests

    for op in ops:
        if isinstance(op, _ReserveOp):
            if op.request_id in live:
                continue  # would raise "already holds a reservation" -- not this property's concern
            try:
                admitted = model.try_reserve(op.request_id, op.size, op.used)
            except CapacityError:
                assert op.size > CAPACITY
                continue
            if admitted:
                live[op.request_id] = op.size
        else:
            if op.request_id in live:
                freed = model.release(op.request_id)
                assert freed == live.pop(op.request_id)

        # Invariants that must hold after every single operation.
        assert model.reserved_tokens == sum(live.values())
        assert model.reserved_tokens <= CAPACITY
        assert model.free_tokens == CAPACITY - model.reserved_tokens
        assert model.free_tokens >= 0
        assert model.used_tokens <= model.reserved_tokens

    # Draining every remaining live reservation returns all capacity: no leaks.
    for request_id in list(live):
        model.release(request_id)
    assert model.reserved_tokens == 0
    assert model.free_tokens == CAPACITY
    assert model.used_tokens == 0
