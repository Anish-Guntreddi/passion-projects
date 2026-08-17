// xv6-plus: vmoobtest -- Phase 5 FR6 invalid-address test program
// (spec Â§1.9 "invalid address/permission cases"). Forks a child that
// dereferences a virtual address far beyond its own process size:
// vmfault()'s `va >= p->sz` check (kernel/vm.c) recognizes this as an
// out-of-bounds fault, counts it into pagefaults_failed, and
// kernel/trap.c's usertrap() kills the process (kexit(-1), so the
// parent's wait() sees xstatus == -1). The PARENT then proves
// invariant #8 (observability/extension code must not destabilize the
// kernel it observes): an ordinary syscall from the parent succeeds
// immediately afterward. See docs/vm-extension.md and
// tests/vm/test_oob_access_killed.py.

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"

int
main(void)
{
  int pid = fork();
  if(pid < 0){
    fprintf(2, "vmoobtest: fork failed\n");
    exit(1);
  }
  if(pid == 0){
    char *brk = sbrk(0); // current process size; doesn't grow it
    // Well beyond p->sz, still comfortably under MAXVA/TRAMPOLINE:
    // guaranteed to be a recognized page fault, not a walk() panic.
    volatile char *bad = (volatile char *)(brk + 0x10000000);
    *bad = 1; // never returns -- the fault kills this process right here
    fprintf(2, "vmoobtest: child survived an out-of-bounds write (unreachable)\n");
    exit(1);
  }

  int xstatus = 0;
  int wpid = wait(&xstatus);
  printf("vmoobtest: child pid=%d wait_pid=%d xstatus=%d\n", pid, wpid, xstatus);

  int me = getpid();
  printf("vmoobtest: parent still alive, getpid=%d\n", me);
  printf("vmoobtest: done\n");
  exit(0);
}
