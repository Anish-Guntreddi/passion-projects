// xv6-plus (Phase 2/3): shared helper for the accounting test
// programs (tests/accounting/, tests/xvtop/) and xvtop itself.
//
// xvstat(2) is index-based, not pid-based (see kernel/pstat.h and
// docs/accounting.md) -- exactly the same shape as a real process
// monitor's problem: `ps`/`top` don't get to ask the kernel "show me
// PID 17," they enumerate the process table and filter. This header
// factors that scan into one place instead of repeating it in every
// program that needs "my own stats."
//
// A header, not a separate .o: this Makefile's UPROGS pattern rule
// links exactly one program-specific object file per tool alongside
// $(ULIB) (see the "_%: %.o $(ULIB) ..." rule), so a shared .c file
// would need a Makefile change. A `static inline` header function,
// compiled separately into each including translation unit, is the
// smallest way to share this logic without touching that rule.
#include "kernel/param.h"
#include "kernel/pstat.h"

// Scan the whole proc table for the calling process's own entry.
// Returns 0 and fills *st on success, -1 if (surprisingly) it isn't
// found -- which can only mean idx=0..NPROC-1 was exhausted before
// hitting the caller's own pid, i.e. NELEM/NPROC drifted out of sync
// with the kernel, since a live process always has *some* slot.
static inline int
xv_find_self(struct xv_pstat *st)
{
  int me = getpid();

  for(int i = 0; i < NPROC; i++){
    if(xvstat(i, st) < 0)
      break; // idx out of range; NPROC slots exhausted
    if(st->pid == me)
      return 0;
  }
  return -1;
}
