from __future__ import annotations

from minipaged.kv.block import PhysicalBlock


def test_default_block_is_free():
    block = PhysicalBlock(block_id=0)
    assert block.ref_count == 0
    assert block.is_free


def test_block_with_positive_ref_count_is_not_free():
    block = PhysicalBlock(block_id=0, ref_count=1)
    assert not block.is_free


def test_block_with_multiple_refs_is_not_free():
    block = PhysicalBlock(block_id=0, ref_count=3)
    assert not block.is_free


def test_block_id_is_preserved():
    block = PhysicalBlock(block_id=7)
    assert block.block_id == 7
