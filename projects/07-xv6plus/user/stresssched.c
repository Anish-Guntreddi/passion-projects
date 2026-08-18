// xv6-plus: stresssched -- Phase 6 stress/race hardening (spec
// roadmap row 6: "scheduler stress"). Original xv6-plus test program.
//
// user/schedbench.c (Phase 4) sets the scheduling policy exactly once,
// sequentially, from the quiescent parent before any child becomes
// RUNNABLE, and each child calls settickets() exactly once before its
// CPU loop -- there is never a policy or ticket-count change *while*
// a competing workload is actively RUNNABLE across multiple harts.
// This program adds that: N_SPINNERS children spin (schedbench's
// do_work() pattern) for a fixed wall-clock competition window while
// a separate CHAOS child concurrently -- on another hart -- calls
// schedpolicy()/settickets() rapidly and repeatedly, racing
// schedule_lottery()'s live two-pass draw (kernel/proc.c) and
// sched_lock against the spinners' own scheduling decisions in real
// time.
//
// This directly stresses invariant #4 ("the scheduler always
// eventually selects eligible work"): every spinner must still
// accumulate real runticks/selections despite the policy switching
// out from under it mid-run (schedule_roundrobin() and
// schedule_lottery() must each remain individually correct and the
// dispatch between them, kernel/proc.c's scheduler() loop reading
// sched_policy once per iteration, must never wedge). It also
// exercises invariant #2 (sched_lock, kernel/proc.c, is a leaf lock,
// never nested with any p->lock -- ADR-0010; nothing here should be
// able to create a new lock-order edge even under maximum realistic
// churn) and invariant #8 (adversarial, rapid policy churn from a
// live competing process must not destabilize the kernel).

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/sched.h"
#include "user/user.h"
#include "user/acct.h"

#define N_SPINNERS    5
#define TARGET_TICKS 20
#define BURST_ITERS 200000
// Safety cap on chaos_worker()'s loop, not an expected/guaranteed
// count: each iteration blocks on pause(1) (>= 1 tick, kernel/sysproc.c)
// plus real scheduling latency while N_SPINNERS=5 competitors are
// actively RUNNABLE, so the TARGET_TICKS=20 wall-clock condition below
// binds first in practice -- real captured runs on this project's dev
// host measured 9-11 toggles, well under this cap (see
// tests/stress/test_stress_scheduler_race.py, which asserts a >= 3
// liveness floor on the real reported count rather than assuming this
// constant is reached).
#define CHAOS_TOGGLES 40

// One ticket count per spinner -- a deliberate spread, same style as
// user/schedbench.c's captured runs (docs/scheduler.md), though this
// program is a liveness stress test, not a fairness measurement: no
// assertion here relies on the exact ratios below.
static int spinner_tickets[N_SPINNERS] = {5, 10, 20, 40, 80};

static void
do_work(int bench_start)
{
  volatile long acc = 0;

  while(uptime() - bench_start < TARGET_TICKS){
    for(long i = 0; i < BURST_ITERS; i++)
      acc += (i ^ (i << 1)) - (i >> 2);
  }
}

static void
spinner(int idx, int bench_start, int wfd)
{
  if(settickets(spinner_tickets[idx]) < 0){
    fprintf(wfd, "stresssched: spinner %d settickets rejected\n", idx);
    exit(1);
  }

  do_work(bench_start);

  struct xv_pstat st;
  if(xv_find_self(&st) < 0){
    fprintf(wfd, "stresssched: spinner %d could not find self\n", idx);
    exit(1);
  }

  fprintf(wfd, "stresssched: report idx=%d pid=%d tickets=%d runticks=%lu "
          "selections=%lu\n",
          idx, getpid(), spinner_tickets[idx], st.runticks, st.selections);
  exit(0);
}

// Concurrently, on its own hart, hammer the two scheduler-experiment
// control syscalls while the spinners above are actively RUNNABLE --
// the actual "chaos" this program is named for. A short pause()
// between toggles (rather than a tight loop) lets this process
// genuinely interleave with the spinners across real scheduling
// decisions instead of just winning every draw itself. Reports
// through its own pipe (wfd), same reason as the spinners above and
// user/schedbench.c: printf() has no per-line atomicity on this
// pinned revision, and this process's report must not interleave with
// the parent's own console writes as it drains the spinners' pipes.
static void
chaos_worker(int bench_start, int wfd)
{
  int policy = SCHED_LOTTERY;
  int toggles = 0;

  for(; toggles < CHAOS_TOGGLES && uptime() - bench_start < TARGET_TICKS; toggles++){
    policy = (policy == SCHED_LOTTERY) ? SCHED_RR : SCHED_LOTTERY;
    if(schedpolicy(policy) < 0){
      fprintf(wfd, "stresssched: chaos schedpolicy(%d) rejected\n", policy);
      exit(1);
    }
    settickets((toggles * 7) % (SCHED_MAX_TICKETS > 1000 ? 1000 : SCHED_MAX_TICKETS));
    pause(1);
  }

  fprintf(wfd, "stresssched: chaos done toggles=%d\n", toggles);
  exit(0);
}

static void
drain_pipe_to_console(int rfd)
{
  char buf[256];
  int n;

  while((n = read(rfd, buf, sizeof(buf))) > 0)
    write(1, buf, n);
  close(rfd);
}

int
main(void)
{
  if(schedpolicy(SCHED_LOTTERY) < 0){
    fprintf(2, "stresssched: initial schedpolicy(SCHED_LOTTERY) rejected\n");
    exit(1);
  }

  int start = uptime();
  int rfd[N_SPINNERS];

  for(int i = 0; i < N_SPINNERS; i++){
    int p[2];
    if(pipe(p) < 0){
      fprintf(2, "stresssched: pipe failed at spinner %d\n", i);
      exit(1);
    }
    int pid = fork();
    if(pid < 0){
      fprintf(2, "stresssched: fork failed at spinner %d\n", i);
      exit(1);
    }
    if(pid == 0){
      close(p[0]);
      spinner(i, start, p[1]);
      // not reached
    }
    close(p[1]);
    rfd[i] = p[0];
  }

  int chaos_p[2];
  if(pipe(chaos_p) < 0){
    fprintf(2, "stresssched: pipe failed for chaos worker\n");
    exit(1);
  }
  int chaos_pid = fork();
  if(chaos_pid < 0){
    fprintf(2, "stresssched: fork failed for chaos worker\n");
    exit(1);
  }
  if(chaos_pid == 0){
    close(chaos_p[0]);
    chaos_worker(start, chaos_p[1]);
    // not reached
  }
  close(chaos_p[1]);

  for(int i = 0; i < N_SPINNERS; i++)
    drain_pipe_to_console(rfd[i]);
  drain_pipe_to_console(chaos_p[0]);

  for(int i = 0; i < N_SPINNERS; i++)
    wait(0);
  wait(0); // chaos worker

  // Restore the baseline policy for whatever runs next in this QEMU
  // session, same hygiene as user/schedbench.c's own cleanup.
  schedpolicy(SCHED_RR);

  printf("stresssched: done\n");
  exit(0);
}
