// xv6-plus: acctforktest -- Phase 2 FR2 test: a child's accounting
// counters start fresh at 0, they are not inherited from the parent's
// already-accumulated totals. Contrast with trace_mask (Phase 1),
// which *is* deliberately inherited across fork() -- see the comments
// on kfork() in kernel/proc.c and docs/accounting.md "fork/exec/exit
// semantics". Original xv6-plus test program; driven by
// tests/accounting/test_fork_fresh_counters.py.
//
// The parent deliberately runs its own syscall_count up before
// forking, so a bug that copied it into the child (instead of
// leaving allocproc()'s zeroed value alone) would be obvious in the
// child's reported count.

#include "kernel/types.h"
#include "user/user.h"
#include "user/acct.h"

#define PARENT_GETPID_LOOPS 20

int
main(void)
{
  struct xv_pstat st;

  for(int i = 0; i < PARENT_GETPID_LOOPS; i++)
    getpid();

  if(xv_find_self(&st) < 0){
    printf("acctforktest: parent: self not found\n");
    exit(1);
  }
  printf("acctforktest: parent pid=%d syscalls=%lu\n", st.pid, st.syscalls);

  int pid = fork();
  if(pid < 0){
    printf("acctforktest: fork failed\n");
    exit(1);
  }

  if(pid == 0){
    if(xv_find_self(&st) < 0){
      printf("acctforktest: child: self not found\n");
      exit(1);
    }
    printf("acctforktest: child pid=%d syscalls=%lu runticks=%lu waitticks=%lu\n",
           st.pid, st.syscalls, st.runticks, st.waitticks);
    exit(0);
  }

  wait(0);
  printf("acctforktest: done\n");
  exit(0);
}
