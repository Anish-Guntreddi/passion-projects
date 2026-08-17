from __future__ import annotations

import pytest

from minipaged.kv.pool import BlockPool, BlockPoolExhausted


def test_rejects_non_positive_num_blocks():
    with pytest.raises(ValueError, match="num_blocks"):
        BlockPool(num_blocks=0, block_size=16)


def test_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        BlockPool(num_blocks=4, block_size=0)


def test_starts_with_every_block_free():
    pool = BlockPool(num_blocks=4, block_size=16)
    assert pool.num_free == 4
    assert pool.num_allocated == 0
    for block_id in range(4):
        assert pool.block(block_id).is_free


def test_allocate_takes_a_block_off_the_free_list():
    pool = BlockPool(num_blocks=4, block_size=16)
    block_id = pool.allocate()
    assert pool.num_free == 3
    assert pool.num_allocated == 1
    assert pool.block(block_id).ref_count == 1
    assert not pool.block(block_id).is_free


def test_allocate_returns_distinct_ids_until_exhausted():
    pool = BlockPool(num_blocks=3, block_size=16)
    ids = {pool.allocate(), pool.allocate(), pool.allocate()}
    assert ids == {0, 1, 2}
    assert pool.num_free == 0


def test_allocate_raises_block_pool_exhausted_when_empty():
    pool = BlockPool(num_blocks=1, block_size=16)
    pool.allocate()
    with pytest.raises(BlockPoolExhausted, match="0/1"):
        pool.allocate()


def test_allocation_order_is_deterministic():
    """Same call sequence -> same block ids, every time (core invariant #7)."""

    def _allocate_three() -> list[int]:
        pool = BlockPool(num_blocks=5, block_size=16)
        return [pool.allocate() for _ in range(3)]

    assert _allocate_three() == _allocate_three()


def test_free_returns_a_sole_owned_block_to_the_free_list():
    pool = BlockPool(num_blocks=2, block_size=16)
    block_id = pool.allocate()
    pool.free(block_id)
    assert pool.num_free == 2
    assert pool.block(block_id).is_free


def test_free_rejects_a_block_with_more_than_one_referrer():
    pool = BlockPool(num_blocks=2, block_size=16)
    block_id = pool.allocate()
    pool.retain(block_id)  # ref_count now 2
    with pytest.raises(ValueError, match="sole ownership"):
        pool.free(block_id)
    assert pool.block(block_id).ref_count == 2  # unchanged by the rejected call


def test_free_rejects_an_already_free_block():
    pool = BlockPool(num_blocks=2, block_size=16)
    with pytest.raises(ValueError, match="sole ownership"):
        pool.free(0)


def test_retain_increments_ref_count():
    pool = BlockPool(num_blocks=2, block_size=16)
    block_id = pool.allocate()
    new_count = pool.retain(block_id)
    assert new_count == 2
    assert pool.block(block_id).ref_count == 2
    assert not pool.block(block_id).is_free


def test_retain_rejects_a_free_block():
    pool = BlockPool(num_blocks=2, block_size=16)
    with pytest.raises(ValueError, match="free block"):
        pool.retain(0)


def test_release_decrements_and_frees_at_zero():
    pool = BlockPool(num_blocks=2, block_size=16)
    block_id = pool.allocate()
    pool.retain(block_id)  # ref_count 2
    assert pool.release(block_id) == 1
    assert not pool.block(block_id).is_free
    assert pool.num_free == 1
    assert pool.release(block_id) == 0
    assert pool.block(block_id).is_free
    assert pool.num_free == 2


def test_release_rejects_double_release():
    pool = BlockPool(num_blocks=2, block_size=16)
    block_id = pool.allocate()
    pool.release(block_id)
    with pytest.raises(ValueError, match="already free"):
        pool.release(block_id)


# -- block_id bounds validation --------------------------------------------
# Regression coverage: without a bounds check, Python's negative-index
# wraparound lets an out-of-range id silently alias a *different* real
# block and corrupt the free list (a bogus id gets pushed instead of the
# index actually touched, and a later allocate() hands it back out).


@pytest.mark.parametrize("bad_id", [-1, -4, 4, 100])
def test_block_rejects_out_of_range_ids(bad_id):
    pool = BlockPool(num_blocks=4, block_size=16)
    with pytest.raises(ValueError, match="out of range"):
        pool.block(bad_id)


@pytest.mark.parametrize("bad_id", [-1, -4, 4, 100])
def test_retain_rejects_out_of_range_ids(bad_id):
    pool = BlockPool(num_blocks=4, block_size=16)
    with pytest.raises(ValueError, match="out of range"):
        pool.retain(bad_id)


@pytest.mark.parametrize("bad_id", [-1, -4, 4, 100])
def test_release_rejects_out_of_range_ids(bad_id):
    pool = BlockPool(num_blocks=4, block_size=16)
    with pytest.raises(ValueError, match="out of range"):
        pool.release(bad_id)


@pytest.mark.parametrize("bad_id", [-1, -4, 4, 100])
def test_free_rejects_out_of_range_ids(bad_id):
    pool = BlockPool(num_blocks=4, block_size=16)
    with pytest.raises(ValueError, match="out of range"):
        pool.free(bad_id)


def test_free_of_negative_id_does_not_corrupt_the_free_list():
    """Direct reproduction of the reported corruption: pool.free(-1) must
    not silently free the real block at index -1 (num_blocks - 1) while
    pushing the literal value -1 onto the free list."""
    pool = BlockPool(num_blocks=4, block_size=16)
    for _ in range(4):
        pool.allocate()  # fully allocated: real block 3 still has ref_count 1
    assert pool.num_free == 0
    with pytest.raises(ValueError, match="out of range"):
        pool.free(-1)
    assert pool.num_free == 0  # unchanged -- no id, bogus or real, was freed
    assert pool.block(3).ref_count == 1  # real block 3 untouched


def test_no_block_is_simultaneously_free_and_referenced():
    """Core invariant #1, exercised directly against the pool's own bookkeeping."""
    pool = BlockPool(num_blocks=3, block_size=16)
    a = pool.allocate()
    b = pool.allocate()
    pool.free(a)
    for block_id in range(pool.num_blocks):
        block = pool.block(block_id)
        is_on_free_list = block_id not in {b}
        assert block.is_free == is_on_free_list
        if block.is_free:
            assert block.ref_count == 0
        else:
            assert block.ref_count >= 1
