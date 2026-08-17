// xv6-plus: xvstatbounds -- Phase 2 FR3 / invariant #7 test: xvstat(2)
// rejects an out-of-range index and an invalid user pointer instead
// of touching kernel state it shouldn't, and keeps working normally
// afterwards. Original xv6-plus test program; driven by
// tests/accounting/test_xvstat_bounds.py.
//
// Mirrors, for xvstat's user-pointer argument, the same "syscall
// interfaces validate user pointers/arguments" invariant that Phase
// 1's docs/invariants.md discussed for trace() (which has no pointer
// argument to validate, by design). xvstat has both an int (idx,
// bounds-checked by kernel/proc.c's procstat()) and a pointer (addr,
// validated by copyout() itself, same as sys_fstat()'s filestat()) --
// this program exercises both failure modes plus the ordinary-success
// case, in one deterministic transcript.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/pstat.h"
#include "user/user.h"

int
main(void)
{
  struct xv_pstat st;

  int r1 = xvstat(-1, &st);
  printf("xvstatbounds: idx=-1 -> %d\n", r1);

  int r2 = xvstat(NPROC, &st);
  printf("xvstatbounds: idx=NPROC -> %d\n", r2);

  int r3 = xvstat(0, (struct xv_pstat*)0xdeadbeef);
  printf("xvstatbounds: bad addr -> %d\n", r3);

  int r4 = xvstat(0, &st);
  printf("xvstatbounds: idx=0 -> %d\n", r4);

  printf("xvstatbounds: done\n");
  exit(0);
}
