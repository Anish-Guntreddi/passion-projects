// xv6-plus: vmpfailcount -- Phase 5 FR6 direct pagefaults_failed
// verification. Every other test that drives vmfault()'s failure path
// (test_oob_access_killed.py/vmoobtest.c, test_memory_exhaustion_
// recovery.py/vmexhausttest.c, test_permission_fault_killed.py/
// vmpermtest.c) only ever *infers* the counter incremented from the
// faulting process's exit status (xstatus == -1) -- that process is
// killed by the same fault it triggered, so it never survives to read
// its own pagefaults_failed.
//
// This program instead triggers vmfault()'s failure path through
// xvstat(2) itself: sys_xvstat() (kernel/sysproc.c) calls copyout()
// directly on whatever output address userspace passes, and a failing
// copyout() -> vmfault() there just makes the syscall return -1 -- it
// does NOT kill the calling process (only a genuine instruction-level
// fault reaching usertrap() directly does that; see kernel/trap.c).
// So this same process survives to read its OWN pagefaults_failed
// both before and after, directly confirming the exact increment
// instead of only a side effect of it.

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"
#include "user/acct.h"

int
main(void)
{
  struct xv_pstat before;
  if(xv_find_self(&before) < 0){
    fprintf(2, "vmpfailcount: could not find self (before)\n");
    exit(1);
  }

  // Deliberately out-of-bounds output pointer for xvstat(2) -- same
  // "well beyond p->sz, still comfortably under MAXVA/TRAMPOLINE"
  // offset user/vmoobtest.c uses, guaranteed to be a recognized
  // page fault (va >= p->sz) rather than a walk() panic.
  char *bad = (char *)sbrk(0) + 0x10000000;
  int r = xvstat(0, (struct xv_pstat *)bad);
  printf("vmpfailcount: xvstat(0, bad_ptr) -> %d\n", r);

  struct xv_pstat after;
  if(xv_find_self(&after) < 0){
    fprintf(2, "vmpfailcount: could not find self (after)\n");
    exit(1);
  }

  printf("vmpfailcount: pagefaults_failed before=%lu after=%lu\n",
         before.pagefaults_failed, after.pagefaults_failed);
  printf("vmpfailcount: done\n");
  exit(0);
}
