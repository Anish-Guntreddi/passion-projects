// xv6-plus: acctreport -- prints the calling process's own struct
// xv_pstat and exits. Original xv6-plus test program. Used standalone
// by nothing on its own; it exists to be exec()'d into by
// acctexectest.c so that tests/accounting/test_exec_preserves_counters.py
// can observe that Phase 2 accounting counters survive exec() (same
// struct proc/pid, only the program image changes -- see
// docs/accounting.md "fork/exec/exit semantics").

#include "kernel/types.h"
#include "user/user.h"
#include "user/acct.h"

int
main(void)
{
  struct xv_pstat st;

  if(xv_find_self(&st) < 0){
    printf("acctreport: self not found\n");
    exit(1);
  }
  printf("acctreport: pid=%d syscalls=%lu\n", st.pid, st.syscalls);
  exit(0);
}
