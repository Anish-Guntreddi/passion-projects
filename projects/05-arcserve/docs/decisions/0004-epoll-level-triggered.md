# ADR 0004: Level-triggered epoll (spec decision D2)

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: Decided before Phase 2, per the spec's kickoff prompt ("D2
  before Phase 2").

## Context

Spec §1.9, decision D2: *"Edge- vs level-triggered epoll → default per
brief: simplest correct first — level-triggered; ET as measured experiment
later."*

## Decision

`EpollReactor` (`include/arcserve/reactor/epoll_reactor.hpp` /
`src/reactor/epoll_reactor.cpp`) registers every fd without `EPOLLET`:
plain level-triggered epoll. Every `epoll_ctl(2)` call in this codebase —
the listener, every accepted client, in both `EchoReactorServer` and
`NonblockingHttpServer` — uses this default (level-triggered) behavior.

### Why level-triggered first

- **Adopts the spec's own recommended default** (portfolio-wide rule:
  adopt each spec's recommended default unless there's a specific reason
  not to; there wasn't one here).
- **Simpler, more forgiving read/write-loop discipline.** Edge-triggered
  mode requires every readable/writable notification to be drained to
  EAGAIN *exactly once*, in every code path, or the fd silently stops being
  reported — a single missed drain-to-EAGAIN anywhere is a live,
  hard-to-notice bug (a connection that quietly stops making progress, not
  a crash). Level-triggered means "if data or buffer space is still there,
  epoll_wait will say so again" is true unconditionally, which is what
  makes this reactor's simplify-first design possible:
  `process_readable()`'s loop-until-WouldBlock (in both
  `EchoReactorServer` and `NonblockingHttpServer`) is an optimization here,
  not a correctness requirement — under LT, even a single non-draining
  `read_some()` call is code that will just get the fd re-reported on the
  next `epoll_wait()`, not silently starved.
- **Consistency with `docs/decisions/0001-...`'s "narrower warning set" and
  `0003-...`'s "handwritten but narrow parser" reasoning:** this project
  repeatedly picks the option that stays correct-by-construction and easy
  to reason about over the option that's asymptotically faster but
  requires every call site to get a subtle invariant right.

### Why this doesn't block later performance work

D2's own text is explicit that ET is "a measured experiment later" — this
ADR does not foreclose it. If Phase 8's benchmark study finds LT's extra
`epoll_wait()` wake-ups are a measurable throughput ceiling under high
concurrency, switching specific hot fds (or all of them) to ET is a scoped,
benchmarkable follow-up change, with before/after numbers — not a default
assumed better without evidence (spec rule 6: never claim performance
results without committed data).

## Consequences

- Every reactor-driven server in this codebase can call `read_some()`/
  `try_flush()` once per dispatched event and simply rely on being
  re-notified if that wasn't enough — no per-call-site "did I actually
  drain to EAGAIN?" bookkeeping.
- `EpollReactorTest.LevelTriggeredFiresAgainIfNotDrained`
  (`tests/unit/test_epoll_reactor.cpp`) asserts this behavior directly, so
  a future change that flips to `EPOLLET` without updating every read/write
  loop accordingly fails a test instead of silently regressing.
- `EpollReactor::add()`/`modify()` take a raw interest mask
  (`kReadable`/`kWritable`) with no ET flag exposed at all yet — adding
  edge-triggered support later is an additive API change (e.g. a third
  bit), not a rewrite.
