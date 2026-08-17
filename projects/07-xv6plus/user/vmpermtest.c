// xv6-plus: vmpermtest -- Phase 5 FR6 permission-fault test program
// (spec Â§1.9 "invalid address/permission cases" -- the "permission"
// half, which test_oob_access_killed.py/user/vmoobtest.c never
// exercised: that program only ever drives an out-of-bounds address,
// va >= p->sz). Forks a child that writes through a pointer into its
// OWN text segment: kernel/exec.c's flags2perm() maps a program's
// text segment PTE_R|PTE_X, never PTE_W (ELF program-header flag 0x2
// is unset for text), so this is an in-bounds, already-mapped store
// with insufficient permission -- a genuine permission violation, not
// an out-of-bounds or out-of-memory failure. vmfault() (kernel/vm.c)
// recognizes this via vm_permission_violation(), counts it into
// pagefaults_failed, and kernel/trap.c's usertrap() kills the process
// with its own distinct diagnostic message (not the generic "invalid
// address or out of memory" line -- see docs/vm-extension.md and
// tests/vm/test_permission_fault_killed.py).

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"

int
main(void)
{
  int pid = fork();
  if(pid < 0){
    fprintf(2, "vmpermtest: fork failed\n");
    exit(1);
  }
  if(pid == 0){
    // main's own code lives in this process's text segment -- a
    // real, in-bounds, already-mapped address, just not a writable
    // one.
    volatile char *text = (volatile char *)main;
    *text = 0x90; // never returns -- the permission fault kills this process here
    fprintf(2, "vmpermtest: child survived a write to its own text segment (unreachable)\n");
    exit(1);
  }

  int xstatus = 0;
  int wpid = wait(&xstatus);
  printf("vmpermtest: child pid=%d wait_pid=%d xstatus=%d\n", pid, wpid, xstatus);

  int me = getpid();
  printf("vmpermtest: parent still alive, getpid=%d\n", me);
  printf("vmpermtest: done\n");
  exit(0);
}
