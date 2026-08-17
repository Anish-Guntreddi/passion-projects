// xv6-plus: tixtest -- Phase 4 lifecycle test program (FR5). Proves
// tickets (kernel/proc.h) follow the same inherited-across-fork
// discipline Phase 1 already established for trace_mask, and that a
// freshly-allocated proc-table slot -- whether brand new or reused --
// always reports SCHED_DEFAULT_TICKETS (kernel/sched.h), never a
// previous occupant's configured value. See docs/scheduler.md and
// tests/scheduler/test_tickets_lifecycle.py.
//
// Usage: tixtest [tickets]
//   No argument: report this process's own tickets/selections at
//     "start" and exit. Two separate `tixtest` runs in the same test
//     both landing on "start" tickets == SCHED_DEFAULT_TICKETS is
//     exactly the invariant-#5 proof: whichever proc-table slot the
//     second run reuses, it must not show the first run's leftover
//     configuration if the first run set one (see the `tickets`
//     argument case below).
//   tickets given: settickets(tickets), then fork() -- both parent
//     and child report their own tickets/selections. The child's
//     tickets must equal the parent's post-settickets() value
//     (inheritance); selections stay near 0 for both (fresh, not
//     inherited).

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"
#include "user/acct.h"

static void
report(const char *tag)
{
  struct xv_pstat st;

  if(xv_find_self(&st) < 0){
    fprintf(2, "tixtest: %s: could not find self via xvstat\n", tag);
    exit(1);
  }
  printf("tixtest: %s: pid=%d tickets=%lu selections=%lu\n",
         tag, getpid(), st.tickets, st.selections);
}

int
main(int argc, char *argv[])
{
  report("start");

  if(argc < 2){
    printf("tixtest: done\n");
    exit(0);
  }

  int t = atoi(argv[1]);
  if(settickets(t) < 0){
    fprintf(2, "tixtest: settickets(%d) rejected\n", t);
    exit(1);
  }

  int pid = fork();
  if(pid < 0){
    fprintf(2, "tixtest: fork failed\n");
    exit(1);
  }
  if(pid == 0){
    report("child");
    exit(0);
  }

  wait(0);
  report("parent");
  printf("tixtest: done\n");
  exit(0);
}
