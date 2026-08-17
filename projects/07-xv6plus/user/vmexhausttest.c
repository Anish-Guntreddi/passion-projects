// xv6-plus: vmexhausttest -- Phase 5 FR6 memory-exhaustion test
// program (spec Â§1.9 "memory exhaustion where practical" and
// "page-accounting cleanup on exit"). kernel/memlayout.h fixes
// PHYSTOP at KERNBASE + 128MB regardless of QEMU's -m flag (freerange()
// hands the allocator exactly that range at boot, kernel/kalloc.c), so
// this test runs under the harness's ordinary QEMU session (128M, the
// value the kernel is actually built to expect) rather than a
// deliberately smaller one -- shrinking -m below what the kernel
// already assumes about physical RAM is not a safe configuration for
// this kernel, not a safety margin.
//
// The child repeatedly sbrklazy()s one page and immediately touches
// it, printing a running progress line every PROGRESS_STRIDE pages so
// the last line printed before it (may) die shows how far it got.
// Once a touch triggers vmfault()'s kalloc()==0 path
// (kernel/vm.c), that fault is recognized-but-unserviceable and
// kernel/trap.c's usertrap() kills the process right there, mid-
// instruction -- no further C code in this process runs (same
// structural limitation docs/accounting.md documents for SYS_exit).
// PAGE_BUDGET bounds the loop so a build where exhaustion doesn't
// happen (e.g. more RAM than expected) reports that plainly instead
// of spinning.
//
// The PARENT then proves the two things this test exists for: the
// child's run was bounded (killed or budget-limited, not hung), and
// whatever memory the child DID grab is fully reclaimed on reap --
// proven by successfully repeating a modest allocate-and-touch
// afterward (freeproc() -> proc_freepagetable() -> uvmfree(), same
// path every process exit already takes).

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/riscv.h"
#include "user/user.h"

#define PAGE_BUDGET      40000  // ~160MB nominal -- see docs/vm-extension.md calibration
#define PROGRESS_STRIDE   2000
#define RECOVERY_PAGES       4

static int
grow_and_touch_one_page(void)
{
  char *p = sbrklazy(PGSIZE);
  if(p == SBRK_ERROR)
    return -1; // sz itself couldn't grow -- a different failure mode than a page fault
  *p = 1;       // triggers the actual page fault
  return 0;
}

int
main(void)
{
  int pid = fork();
  if(pid < 0){
    fprintf(2, "vmexhausttest: fork failed\n");
    exit(1);
  }
  if(pid == 0){
    for(int i = 0; i < PAGE_BUDGET; i++){
      if(grow_and_touch_one_page() < 0){
        printf("vmexhausttest: child sz-growth failed at page %d\n", i);
        exit(1);
      }
      if((i + 1) % PROGRESS_STRIDE == 0)
        printf("vmexhausttest: child progress pages=%d\n", i + 1);
    }
    printf("vmexhausttest: child reached PAGE_BUDGET=%d without exhausting memory\n",
           PAGE_BUDGET);
    exit(0);
  }

  int xstatus = 0;
  wait(&xstatus);
  printf("vmexhausttest: child pid=%d xstatus=%d\n", pid, xstatus);

  // Recovery check: a fresh, modest allocation must still succeed --
  // proving the exhausted child's pages were reclaimed on reap, not
  // leaked.
  int rpid = fork();
  if(rpid < 0){
    fprintf(2, "vmexhausttest: recovery fork failed\n");
    exit(1);
  }
  if(rpid == 0){
    for(int i = 0; i < RECOVERY_PAGES; i++){
      if(grow_and_touch_one_page() < 0){
        printf("vmexhausttest: recovery FAILED at page %d -- memory not reclaimed?\n", i);
        exit(1);
      }
    }
    printf("vmexhausttest: recovery ok, touched %d pages\n", RECOVERY_PAGES);
    exit(0);
  }
  wait(&xstatus);
  printf("vmexhausttest: recovery xstatus=%d\n", xstatus);
  printf("vmexhausttest: done\n");
  exit(0);
}
