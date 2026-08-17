// xv6-plus: tracetest -- direct, deterministic exercise of the
// trace() syscall's enable/disable semantics (FR1, Phase 1).
// Original xv6-plus test program; driven by
// tests/syscall/test_trace_toggle.py.
//
// Unlike xtrace (which enables tracing and then exec()s into a
// different program), this program calls trace() on itself, so it
// can prove that:
//   - a traced syscall produces exactly one "syscall NAME -> RET"
//     line while tracing is enabled;
//   - trace(0) actually turns tracing back off for the same,
//     still-running process (not just "until the next exec").
//
// getpid() is used as the traced call because it is cheap, has no
// side effects, and is easy to correlate 1:1 with a trace line.

#include "kernel/types.h"
#include "kernel/syscall.h"
#include "user/user.h"

int
main(void)
{
  printf("tracetest: enabling\n");
  trace(1 << SYS_getpid);
  getpid();                    // expect one "syscall getpid -> N" line

  printf("tracetest: disabling\n");
  trace(0);
  getpid();                    // expect no trace line for this one

  printf("tracetest: done\n");
  exit(0);
}
