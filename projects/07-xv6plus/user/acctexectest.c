// xv6-plus: acctexectest -- Phase 2 FR2 test: accounting counters
// survive exec(), unlike fork() where they deliberately reset (see
// acctforktest.c and kernel/proc.c's kfork()). Original xv6-plus test
// program; driven by
// tests/accounting/test_exec_preserves_counters.py.
//
// Runs a fixed, known number of cheap syscalls, then exec()s into
// acctreport, which reports the (same struct proc's) syscall count.
// kexec() (kernel/exec.c) never touches syscall_count/runticks/
// waitticks -- it changes the page table, name, and program counter,
// nothing else on struct proc that this feature cares about -- so
// acctreport's reported count should include everything counted here
// plus the exec() call itself and a little more from acctreport's own
// startup, comfortably above this program's own syscall count alone.

#include "kernel/types.h"
#include "user/user.h"

#define PRE_EXEC_GETPID_LOOPS 7

int
main(void)
{
  for(int i = 0; i < PRE_EXEC_GETPID_LOOPS; i++)
    getpid();

  printf("acctexectest: about to exec\n");

  char *args[] = {"acctreport", 0};
  exec("acctreport", args);

  printf("acctexectest: exec failed\n");
  exit(1);
}
