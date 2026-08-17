from __future__ import annotations

import pytest

from minipaged.kv.pool import BlockPool, BlockPoolExhausted
from minipaged.kv.table import BlockTable, blocks_needed_for


def test_blocks_needed_for_rounds_up():
    assert blocks_needed_for(0, block_size=16) == 0
    assert blocks_needed_for(1, block_size=16) == 1
    assert blocks_needed_for(16, block_size=16) == 1
    assert blocks_needed_for(17, block_size=16) == 2
    assert blocks_needed_for(32, block_size=16) == 2


def test_blocks_needed_for_rejects_negative_tokens():
    with pytest.raises(ValueError, match="num_tokens"):
        blocks_needed_for(-1, block_size=16)


def test_new_table_has_no_blocks():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    assert table.num_blocks == 0
    assert table.capacity_tokens == 0
    assert table.covers(0)
    assert not table.covers(1)


def test_grow_to_allocates_exactly_the_blocks_needed():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    grown = table.grow_to(20)  # needs 2 blocks (20 > 16)
    assert grown == 2
    assert table.num_blocks == 2
    assert table.capacity_tokens == 32
    assert table.covers(20)
    assert pool.num_free == 2


def test_grow_to_is_a_no_op_when_already_sufficient():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    table.grow_to(10)
    assert table.grow_to(16) == 0  # still fits in the single block already held
    assert table.num_blocks == 1


def test_grow_to_only_allocates_the_delta():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    table.grow_to(10)  # 1 block
    grown = table.grow_to(20)  # needs 2 total -> allocates 1 more
    assert grown == 1
    assert table.num_blocks == 2


def test_grow_to_rejects_negative_target():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    with pytest.raises(ValueError, match="num_tokens"):
        table.grow_to(-1)


def test_grow_to_raises_and_rolls_back_when_pool_is_exhausted():
    pool = BlockPool(num_blocks=2, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    with pytest.raises(BlockPoolExhausted):
        table.grow_to(48)  # needs 3 blocks, pool only has 2
    assert table.num_blocks == 0  # nothing partially committed
    assert pool.num_free == 2  # every tentatively-allocated block was rolled back


def test_grow_to_rollback_does_not_disturb_other_tables_blocks():
    pool = BlockPool(num_blocks=3, block_size=16)
    other = BlockTable(block_size=16, pool=pool)
    other.grow_to(16)  # takes 1 block, 2 free left
    victim = BlockTable(block_size=16, pool=pool)
    with pytest.raises(BlockPoolExhausted):
        victim.grow_to(48)  # needs 3, only 2 free
    assert victim.num_blocks == 0
    assert other.num_blocks == 1  # untouched
    assert pool.num_free == 2


def test_free_all_returns_every_block_and_clears_the_table():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    table.grow_to(40)  # 3 blocks
    freed = table.free_all()
    assert freed == 3
    assert table.num_blocks == 0
    assert pool.num_free == 4


def test_free_all_on_empty_table_is_a_no_op():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    assert table.free_all() == 0
    assert pool.num_free == 4


def test_physical_block_for_token_maps_within_a_block():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    table.grow_to(20)  # 2 blocks: logical block 0 covers tokens 0-15, block 1 covers 16-31
    first_block, second_block = table.block_ids
    assert table.physical_block_for_token(0) == first_block
    assert table.physical_block_for_token(15) == first_block
    assert table.physical_block_for_token(16) == second_block
    assert table.physical_block_for_token(19) == second_block


def test_physical_block_for_token_rejects_out_of_range_index():
    pool = BlockPool(num_blocks=4, block_size=16)
    table = BlockTable(block_size=16, pool=pool)
    table.grow_to(10)
    with pytest.raises(IndexError):
        table.physical_block_for_token(16)  # only 1 block (0-15) allocated
    with pytest.raises(IndexError):
        table.physical_block_for_token(-1)


def test_a_sequences_logical_block_count_always_covers_its_kv_length():
    """Core invariant #3, driven through several growth steps."""
    pool = BlockPool(num_blocks=8, block_size=4)
    table = BlockTable(block_size=4, pool=pool)
    for current_len in range(1, 15):
        table.grow_to(current_len)
        assert table.covers(current_len)
        assert table.capacity_tokens - current_len < 4  # never over-allocates by a whole block
