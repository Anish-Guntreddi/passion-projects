// xv6-plus: vmfaulttest -- Phase 5 FR6 basic page-fault-telemetry
// test program. sbrklazy()s a handful of pages (growing sz without
// mapping anything), touches each page once, and reports
// pagefaults/pagefaults_failed at four stages -- proving: sbrklazy()
// alone doesn't fault anything in; touching N distinct pages produces
// exactly N counted faults; and touching an already-resident page a
// second time does NOT double-count (vmfault()'s ismapped() check,
// kernel/vm.c). See docs/vm-extension.md and
// tests/vm/test_pagefault_counting.py.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/riscv.h"
#include "user/user.h"
#include "user/acct.h"

#define NPAGES 4

static void
report(const char *tag)
{
  struct xv_pstat st;

  if(xv_find_self(&st) < 0){
    fprintf(2, "vmfaulttest: %s: could not find self\n", tag);
    exit(1);
  }
  printf("vmfaulttest: %s: pid=%d pagefaults=%lu pagefaults_failed=%lu sz=%lu\n",
         tag, getpid(), st.pagefaults, st.pagefaults_failed, st.sz);
}

int
main(void)
{
  report("start");

  char *base = sbrklazy(NPAGES * PGSIZE);
  if(base == SBRK_ERROR){
    fprintf(2, "vmfaulttest: sbrklazy failed\n");
    exit(1);
  }
  report("after_sbrklazy"); // still 0 faults -- growing sz doesn't touch memory

  for(int i = 0; i < NPAGES; i++)
    base[i * PGSIZE] = (char)(i + 1); // one byte per page -> one fault per page

  report("after_touch"); // pagefaults == NPAGES

  base[0] = 99; // re-touch the first page's already-mapped byte
  report("after_retouch"); // unchanged from after_touch

  printf("vmfaulttest: done\n");
  exit(0);
}
