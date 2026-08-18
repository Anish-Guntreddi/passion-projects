// xv6-plus: stressforkexit -- Phase 6 stress/race hardening (spec
// Â§1.9/Â§Part 3 roadmap row 6: "concurrent fork/exit ... stress").
// Original xv6-plus test program.
//
// Two-part workload, both exercising invariants #1 (accounting never
// corrupts process lifecycle state), #3 (telemetry init/fork/exec/exit
// correctness), and #5 (zombie/free slots never leak) -- but under
// real concurrent load across every hart (this project's QEMU session
// boots -smp 3, kernel/param.h), which the Phase 1-5 lifecycle tests
// never did: tests/syscall/test_trace_fork_inheritance.py and
// tests/accounting/test_slot_reuse_reset.py each exercise one
// fork/exit cycle (or forktest's proc-table-filling load) at a time,
// not many concurrently-RUNNABLE fork/exit cycles racing allocproc()/
// freeproc()/wait_lock/pid_lock against each other simultaneously.
//
// Part 1 (ROUNDS rounds of CHILDREN_PER_ROUND concurrent children):
// each round forks CHILDREN_PER_ROUND children essentially back to
// back -- they become RUNNABLE together and genuinely race each other
// for allocpid()/allocproc() slots and CPU time across all three
// harts -- each child does a handful of cheap syscalls (exercising
// syscall_count under concurrent p->lock churn, one lock per process
// so no cross-child contention by construction -- see
// docs/decisions/0009-accounting-counter-locking.md) then exits with a
// distinct, checkable status. The parent wait()s for all
// CHILDREN_PER_ROUND, verifying every expected status arrives exactly
// once -- a corrupted pid_lock/wait_lock/proc-table race under this
// load would show up here as a missing, duplicated, or wrong-status
// reap, not just a hang.
//
// Part 2 (table-fill proof): after every round is fully reaped (so
// only this process, sh, and init hold proc-table slots), fork()
// until it fails, forcing every remaining slot to be filled
// concurrently and held open (each filled child blocks in pause(),
// the same "sit around until killed" idiom user/forphan.c and
// user/dorphan.c already use for a deliberately long-lived process).
// The number of successful forks must exactly equal NPROC minus the
// 3 slots already held (init, sh, this process) -- an exact, provable
// bound against invariant #5 (no leaked UNUSED/ZOMBIE-should-be-free
// slot from Part 1's churn silently eating into this budget). Every
// filled child is then kkill()ed and reaped, proving the fill itself
// doesn't leak either.

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"
#include "user/acct.h"

#define ROUNDS              5
#define CHILDREN_PER_ROUND  8

// This process + init + sh hold the other 3 slots at fill time (see
// module comment) -- NPROC (kernel/param.h) minus those 3.
#define EXPECTED_FILL (NPROC - 3)

static void
run_round(int r)
{
  int pids[CHILDREN_PER_ROUND];
  int seen[CHILDREN_PER_ROUND];

  for(int i = 0; i < CHILDREN_PER_ROUND; i++)
    seen[i] = 0;

  for(int i = 0; i < CHILDREN_PER_ROUND; i++){
    int pid = fork();
    if(pid < 0){
      printf("stressforkexit: round %d fork %d failed\n", r, i);
      exit(1);
    }
    if(pid == 0){
      // Cheap syscall churn (getpid + one self-lookup via xvstat,
      // which itself loops xvstat() calls -- see user/acct.h) so this
      // child's own syscall_count moves under real concurrent p->lock
      // traffic from its siblings' identical work on the other harts.
      struct xv_pstat st;
      for(int k = 0; k < 20; k++)
        getpid();
      xv_find_self(&st);
      exit(1000 + r * CHILDREN_PER_ROUND + i);
    }
    pids[i] = pid;
  }

  for(int i = 0; i < CHILDREN_PER_ROUND; i++){
    int xstatus = -1;
    int wpid = wait(&xstatus);
    // Find which slot this reap corresponds to by expected status,
    // not by assuming wait() drains pids[] in fork order (it need
    // not -- children finish in whatever order the scheduler picks).
    int found = -1;
    for(int i2 = 0; i2 < CHILDREN_PER_ROUND; i2++){
      if(pids[i2] == wpid && xstatus == 1000 + r * CHILDREN_PER_ROUND + i2){
        found = i2;
        break;
      }
    }
    if(found < 0 || seen[found]){
      printf("stressforkexit: round %d reap MISMATCH wpid=%d xstatus=%d\n",
             r, wpid, xstatus);
      exit(1);
    }
    seen[found] = 1;
  }

  for(int i = 0; i < CHILDREN_PER_ROUND; i++){
    if(!seen[i]){
      printf("stressforkexit: round %d missing child %d\n", r, i);
      exit(1);
    }
  }

  printf("stressforkexit: round %d children=%d ok\n", r, CHILDREN_PER_ROUND);
}

int
main(void)
{
  for(int r = 0; r < ROUNDS; r++)
    run_round(r);
  printf("stressforkexit: rounds done\n");

  // Part 2: fill every remaining proc-table slot concurrently and
  // hold each one open.
  int filled[NPROC];
  int nfilled = 0;

  while(nfilled < NPROC){
    int pid = fork();
    if(pid < 0)
      break; // table full -- this is the expected, not an error, exit
    if(pid == 0){
      for(;;) pause(1000); // sit around until killed, like forphan.c/dorphan.c
    }
    filled[nfilled++] = pid;
  }

  printf("stressforkexit: fill count=%d\n", nfilled);

  for(int i = 0; i < nfilled; i++)
    kill(filled[i]);
  for(int i = 0; i < nfilled; i++)
    wait(0);

  printf("stressforkexit: done\n");
  exit(0);
}
