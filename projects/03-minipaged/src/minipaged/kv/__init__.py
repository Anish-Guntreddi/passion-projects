"""KV-memory subsystem.

Phase 1: ``contiguous.py`` -- the naive contiguous-reservation baseline
(``ContiguousMemoryModel``, ``ContiguousKVEngine``) that Phase 2's paged
allocator is measured against.

Phase 2: the paged KV subsystem named in the spec's repository layout --
``block.py`` (``PhysicalBlock``), ``pool.py`` (``BlockPool`` free-list),
``table.py`` (per-sequence ``BlockTable``), and ``manager.py``
(``PagedKVManager`` + ``PagedKVEngine``, ties allocation into the engine,
mirroring ``ContiguousKVEngine``'s hook pattern).

Phase 4 (not yet implemented) extends this package with prefix sharing /
copy-on-write, building on ``BlockPool.retain``/``release`` (already
present, unused until then).
"""

from __future__ import annotations

from minipaged.kv.block import PhysicalBlock
from minipaged.kv.contiguous import (
    CapacityError,
    ContiguousKVEngine,
    ContiguousMemoryMetrics,
    ContiguousMemoryModel,
)
from minipaged.kv.manager import PagedKVEngine, PagedKVManager, PagedKVMetrics
from minipaged.kv.pool import BlockPool, BlockPoolExhausted
from minipaged.kv.table import BlockTable, blocks_needed_for

__all__ = [
    "BlockPool",
    "BlockPoolExhausted",
    "BlockTable",
    "CapacityError",
    "ContiguousKVEngine",
    "ContiguousMemoryMetrics",
    "ContiguousMemoryModel",
    "PagedKVEngine",
    "PagedKVManager",
    "PagedKVMetrics",
    "PhysicalBlock",
    "blocks_needed_for",
]
