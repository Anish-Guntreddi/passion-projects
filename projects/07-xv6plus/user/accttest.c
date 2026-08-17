// xv6-plus: accttest -- deterministic exercise of the Phase 2
// accounting counters (FR2) via the xvstat(2) syscall (FR3, ADR-0003).
// Original xv6-plus test program; driven by
// tests/accounting/test_syscall_count_monotonic.py and
// tests/accounting/test_waitticks_from_pause.py.
//
// Reports its own struct xv_pstat at three stages, each on its own
// clearly-marked line, so a Python test can find and compare them:
//   1. "start"           -- right after main() begins.
//   2. "after_getpids"   -- after a fixed number of cheap, side-
//                           effect-free getpid() calls, to move
//                           syscall_count forward by a known amount.
//   3. "after_pause"     -- after pause(N), which sleeps on &ticks
//                           (kernel/sysproc.c: sys_pause -> sleep()),
//                           to move waitticks forward by about N.

#include "kernel/types.h"
#include "user/user.h"
#include "user/acct.h"

#define GETPID_LOOPS 5
#define PAUSE_TICKS  5

static void
report(const char *tag)
{
  struct xv_pstat st;

  if(xv_find_self(&st) < 0){
    printf("accttest: %s: self not found\n", tag);
    return;
  }
  printf("accttest: %s: pid=%d state=%d sz=%lu runticks=%lu waitticks=%lu syscalls=%lu\n",
         tag, st.pid, st.state, st.sz, st.runticks, st.waitticks, st.syscalls);
}

int
main(void)
{
  report("start");

  for(int i = 0; i < GETPID_LOOPS; i++)
    getpid();
  report("after_getpids");

  pause(PAUSE_TICKS);
  report("after_pause");

  printf("accttest: done\n");
  exit(0);
}
