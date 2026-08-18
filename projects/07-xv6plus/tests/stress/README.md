# tests/stress/

Phase 6 (stress/race hardening: concurrent fork/exit, syscall-heavy,
memory-pressure, scheduler stress; lock/invariant audit). Four test
modules, one per `user/stress*.c` program, each racing genuinely
concurrent workloads across every hart (this project's QEMU session
boots `-smp 3`, `kernel/param.h`) against a kernel path Phases 1-5
only ever exercised sequentially:

| Test module | Drives | Spec roadmap row 6 category |
|---|---|---|
| `test_stress_fork_exit.py` | `user/stressforkexit.c` | concurrent fork/exit |
| `test_stress_syscalls.py` | `user/stresssyscalls.c` | syscall-heavy |
| `test_stress_scheduler_race.py` | `user/stresssched.c` | scheduler stress |
| `test_stress_vm_pressure.py` | `user/stressvm.c` | memory-pressure |

Full design writeup, per-program rationale, and the lock/invariant
audit (spec Phase 6 deliverable): [`docs/stress-testing.md`](../../docs/stress-testing.md).

`_stress_helpers.py` holds this category's shared report-line regexes
(same `_<category>_helpers.py`-not-`_helpers.py` naming reason
`tests/accounting/_acct_helpers.py` and
`tests/scheduler/_sched_helpers.py` use: `scripts/run_tests.py` runs
every discovered test module inside one Python process, and a bare
`_helpers` name would collide across categories).

`tests/syscall/test_trace_fork_inheritance.py` still exercises
`forktest`'s proc-table-filling fork/exit load as part of the Phase 1
suite, but that is a single sequential process filling the table, not
concurrent racing -- this directory's tests are the dedicated,
repeated-run stress category the roadmap calls for.
