from __future__ import annotations

import pytest

from minipaged.simulation.clock import SimClock


def test_starts_at_zero_by_default():
    clock = SimClock()
    assert clock.now() == 0.0


def test_starts_at_custom_value():
    clock = SimClock(start=5.0)
    assert clock.now() == 5.0


def test_rejects_negative_start():
    with pytest.raises(ValueError):
        SimClock(start=-1.0)


def test_advance_moves_forward_and_returns_new_now():
    clock = SimClock()
    result = clock.advance(2.5)
    assert result == 2.5
    assert clock.now() == 2.5
    clock.advance(2.5)
    assert clock.now() == 5.0


def test_advance_rejects_negative_delta():
    clock = SimClock()
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1.0)


def test_advance_zero_is_a_noop_but_allowed():
    clock = SimClock(start=3.0)
    clock.advance(0.0)
    assert clock.now() == 3.0


def test_advance_to_jumps_forward():
    clock = SimClock()
    clock.advance_to(10.0)
    assert clock.now() == 10.0


def test_advance_to_is_noop_if_target_is_in_the_past():
    clock = SimClock(start=10.0)
    clock.advance_to(3.0)
    assert clock.now() == 10.0


def test_advance_to_is_noop_if_target_equals_now():
    clock = SimClock(start=7.0)
    clock.advance_to(7.0)
    assert clock.now() == 7.0


def test_clock_never_decreases_across_mixed_operations():
    clock = SimClock()
    seen = [clock.now()]
    for delta, target in [(1.0, None), (None, 0.5), (2.0, None), (None, 100.0)]:
        if delta is not None:
            clock.advance(delta)
        else:
            clock.advance_to(target)
        seen.append(clock.now())
    assert seen == sorted(seen)
