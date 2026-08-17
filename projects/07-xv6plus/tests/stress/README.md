# tests/stress/

Reserved for Phase 6 (stress/race hardening: concurrent fork/exit,
syscall-heavy, memory-pressure, scheduler stress; lock/invariant
audit). Empty as of Phase 0/1. `tests/syscall/test_trace_fork_inheritance.py`
already exercises `forktest`'s proc-table-filling fork/exit load as
part of the Phase 1 suite, but a dedicated stress category with
repeated/looped runs is Phase 6 scope per the roadmap in the project
spec.
