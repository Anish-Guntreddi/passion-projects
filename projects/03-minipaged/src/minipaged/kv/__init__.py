"""KV-memory subsystem.

Phase 1 status: ``contiguous.py`` -- the naive contiguous-reservation
baseline (``ContiguousMemoryModel``, ``ContiguousKVEngine``) that Phase 2's
paged allocator will be measured against.

Phase 2 (not yet implemented) adds the paged KV subsystem named in the
spec's repository layout: ``block.py`` (``PhysicalBlock``), ``pool.py``
(``BlockPool`` free-list), ``table.py`` (per-sequence ``BlockTable``), and
``manager.py`` (ties allocation into the engine, mirroring
``ContiguousKVEngine``'s hook pattern).
"""

from __future__ import annotations

from minipaged.kv.contiguous import (
    CapacityError,
    ContiguousKVEngine,
    ContiguousMemoryMetrics,
    ContiguousMemoryModel,
)

__all__ = [
    "CapacityError",
    "ContiguousKVEngine",
    "ContiguousMemoryMetrics",
    "ContiguousMemoryModel",
]
