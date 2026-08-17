"""Deterministic domain simulation: clock, event log, arrival-trace
generator, and the Phase 0 SimulationEngine.

Everything in this package runs with no real model (see
``docs/decisions/0001-simulation-first-architecture.md``); it exists to
make scheduling/lifecycle behavior observable and testable before any
model or GPU is involved.
"""

from __future__ import annotations

from minipaged.simulation.clock import SimClock
from minipaged.simulation.engine import SimulationEngine
from minipaged.simulation.events import Event, EventLog, EventType
from minipaged.simulation.trace import TraceConfig, TraceRequestSpec, generate_trace

__all__ = [
    "Event",
    "EventLog",
    "EventType",
    "SimClock",
    "SimulationEngine",
    "TraceConfig",
    "TraceRequestSpec",
    "generate_trace",
]
