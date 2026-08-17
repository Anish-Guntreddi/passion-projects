// xv6-plus: vmexectest -- Phase 5 FR6 exec-interaction test program
// (spec Â§1.9 "fork/exec interactions"). Grows sz via sbrklazy()
// without touching the new memory (a pending, never-serviced lazy
// region), then exec()s into a different program. kexec()
// (kernel/exec.c) builds the new image in a fresh page table and only
// frees the OLD page table (uvmfree(), which frees whatever pages
// happen to be mapped -- here, none in the pending region) after
// committing -- proving a never-touched lazy region doesn't leak or
// crash anything across exec(). See docs/vm-extension.md and
// tests/vm/test_exec_discards_lazy_region.py.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/riscv.h"
#include "user/user.h"

int
main(void)
{
  char *base = sbrklazy(4 * PGSIZE);
  if(base == SBRK_ERROR){
    fprintf(2, "vmexectest: sbrklazy failed\n");
    exit(1);
  }

  printf("vmexectest: about to exec, never touched the sbrklazy'd region\n");
  exec("echo", (char *[]){"echo", "vmexectest_exec_ok", 0});
  fprintf(2, "vmexectest: exec failed\n");
  exit(1);
}
