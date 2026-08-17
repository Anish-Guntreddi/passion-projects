// xv6-plus: tixvalidate -- Phase 4 FR5 argument-validation test
// program. Both new Phase 4 syscalls take a plain int (no user
// pointer to mis-validate, same shape as Phase 1's trace()), but each
// still has real input validation worth proving directly:
//   settickets(n): n < 0 is rejected (-1); n == 0 is accepted (0
//     tickets is a valid, deliberately-low-priority configuration --
//     see the zero-ticket-floor discussion in
//     docs/decisions/0010-lottery-scheduler-design.md); an ordinary
//     positive value is accepted and observably takes effect;
//     n > SCHED_MAX_TICKETS (kernel/sched.h) is rejected (-1) --
//     added during review followup, see
//     docs/decisions/0013-ticket-count-bound.md for why the bound
//     exists (a large-enough RUNNABLE ticket sum would otherwise
//     truncate through the lottery draw's uint32 PRNG modulo).
//   schedpolicy(p): a value outside {SCHED_RR, SCHED_LOTTERY} is
//     rejected (-1) and leaves the current policy untouched; a valid
//     value is accepted and returns the *previous* policy.
// See tests/scheduler/test_settickets_validation.py and
// tests/scheduler/test_schedpolicy_validation.py.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/sched.h"
#include "user/user.h"
#include "user/acct.h"

int
main(void)
{
  int r;

  r = settickets(-1);
  printf("tixvalidate: settickets(-1) -> %d\n", r);

  r = settickets(0);
  printf("tixvalidate: settickets(0) -> %d\n", r);

  r = settickets(SCHED_MAX_TICKETS + 1);
  printf("tixvalidate: settickets(SCHED_MAX_TICKETS+1) -> %d\n", r);

  r = settickets(SCHED_MAX_TICKETS);
  printf("tixvalidate: settickets(SCHED_MAX_TICKETS) -> %d\n", r);

  r = settickets(5);
  printf("tixvalidate: settickets(5) -> %d\n", r);

  struct xv_pstat st;
  if(xv_find_self(&st) < 0){
    fprintf(2, "tixvalidate: could not find self\n");
    exit(1);
  }
  printf("tixvalidate: tickets_after=%lu\n", st.tickets);

  // 99 is deliberately outside {SCHED_RR, SCHED_LOTTERY} (0, 1).
  r = schedpolicy(99);
  printf("tixvalidate: schedpolicy(99) -> %d\n", r);

  r = schedpolicy(SCHED_LOTTERY);
  printf("tixvalidate: schedpolicy(SCHED_LOTTERY) -> %d\n", r);

  r = schedpolicy(SCHED_RR);
  printf("tixvalidate: schedpolicy(SCHED_RR) -> %d\n", r);

  printf("tixvalidate: done\n");
  exit(0);
}
