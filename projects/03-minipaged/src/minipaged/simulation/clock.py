"""SimClock: a deterministic, controllable discrete clock.

The point of "deterministic simulation mode" (Phase 0 deliverable) is that
nothing in the simulator ever calls ``time.time()`` -- every notion of
"now" flows through one SimClock instance that only moves forward when
explicitly told to. That is what makes replaying the exact same trace
twice byte-for-byte reproducible (core invariant #7).
"""

from __future__ import annotations


class SimClock:
    """Monotonic simulated clock measured in abstract time units ("ticks").

    A tick has no wall-clock meaning in Phase 0-1 -- it is whatever unit
    the trace generator's arrival times and the engine's per-decode-step
    cost are expressed in. ``advance`` refuses negative deltas so
    ``now()`` is always non-decreasing, which every ordering-dependent
    invariant (event log ordering, admission ordering) relies on.
    """

    def __init__(self, start: float = 0.0) -> None:
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, delta: float) -> float:
        if delta < 0:
            raise ValueError(f"SimClock cannot move backwards (delta={delta})")
        self._now += delta
        return self._now

    def advance_to(self, timestamp: float) -> float:
        """Jump forward to an absolute timestamp; no-op if already there or past it."""
        if timestamp > self._now:
            self._now = timestamp
        return self._now
