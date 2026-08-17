// xv6-plus: vmforktest -- Phase 5 FR6 fork-interaction test program
// (spec Â§1.9 "fork/exec interactions"). Grows sz via sbrklazy()
// without touching the new memory (so uvmcopy(), kernel/vm.c, has no
// mapped page there yet to copy -- it skips unmapped pages by
// design), then fork()s. The child touches one page in that region:
// it must be serviced by the CHILD's OWN independent vmfault() call,
// not something inherited from the parent, so the child's pagefaults
// counter goes to 1 -- and the PARENT's stays at 0, proving separate
// struct proc means separate counters even though the region's size
// came from a common sbrklazy() call before the fork(). See
// docs/vm-extension.md and tests/vm/test_fork_lazy_region.py.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/riscv.h"
#include "user/user.h"
#include "user/acct.h"

int
main(void)
{
  char *base = sbrklazy(2 * PGSIZE);
  if(base == SBRK_ERROR){
    fprintf(2, "vmforktest: sbrklazy failed\n");
    exit(1);
  }

  int pid = fork();
  if(pid < 0){
    fprintf(2, "vmforktest: fork failed\n");
    exit(1);
  }
  if(pid == 0){
    base[0] = 7; // this child's own first fault in the shared-by-sz region
    struct xv_pstat st;
    if(xv_find_self(&st) < 0){
      fprintf(2, "vmforktest: child could not find self\n");
      exit(1);
    }
    printf("vmforktest: child pid=%d pagefaults=%lu\n", getpid(), st.pagefaults);
    exit(0);
  }

  wait(0);
  struct xv_pstat st;
  if(xv_find_self(&st) < 0){
    fprintf(2, "vmforktest: parent could not find self\n");
    exit(1);
  }
  printf("vmforktest: parent pid=%d pagefaults=%lu\n", getpid(), st.pagefaults);
  printf("vmforktest: done\n");
  exit(0);
}
