# ADR-0007: Accounting counters are tick-granularity

**Status:** Accepted; implemented in Phase 2
**Decision:** D7 (spec §1.10)
**Date:** 2026-08-17 (recorded ahead of implementation; see ADR-0003;
revisited after Phase 2 implementation)

## Context

Phase 2 accounting reports process runtime. D7 is ticks vs.
finer-grained counters; spec default: ticks.

## Decision

Adopt the spec default: the existing `ticks` global, already
incremented by the timer interrupt and already exposed via
`sys_uptime()` (`kernel/trap.c`, `kernel/sysproc.c`), is the unit for
Phase 2's runtime/wait-time accounting -- not a finer-grained
(cycle-counter or RTC-based) measurement. Implemented in Phase 2:
`kernel/trap.c`'s `clockintr()` charges one `runtick` per timer
interrupt to whichever process is running on that hart, and
`kernel/proc.c`'s `sleep()` charges `waitticks` in units of the same
global `ticks` counter. Recorded ahead of implementation for the same
reason as ADR-0003/0004.

## Rationale

xv6's timer interrupt rate is the only clock source already wired
through the kernel and already backing the existing `pause()`/
`uptime()` code paths. QEMU's virtual timer tick rate is stable and
simulation-appropriate, but it is not a real wall-clock- or
cycle-accurate signal -- reporting anything finer than ticks would
imply a precision the platform doesn't actually have, which is the
kind of fabricated precision the portfolio's benchmark-honesty rule
(never report a number the hardware/platform didn't actually measure)
warns against.

## Consequences

Phase 2+ runtime/wait-time fields are integer tick counts, matching
the granularity `sys_uptime()` already reports to userspace, so a
Phase 3 `xvtop` display and any Phase 4 scheduler benchmark share one
consistent unit end to end.
