"""Top-level serving runtime -- boundary placeholder.

Not implemented yet. Once a real model (Phase 5) and scheduler (Phase 3)
exist, this package composes them into the actual serving loop that the
streaming API (Phase 6) drives. Phase 0-1's equivalent composition root is
``minipaged.simulation.engine.SimulationEngine`` /
``minipaged.kv.contiguous.ContiguousKVEngine``, which intentionally do not
live here since they never touch a real model.
"""

from __future__ import annotations
