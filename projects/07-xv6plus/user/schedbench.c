// xv6-plus: schedbench -- Phase 4 scheduler-experiment benchmark
// driver (FR5, spec Â§1.9). Sets the requested global scheduling
// policy, forks one child per ticket count given on the command
// line, has each child do an identical fixed amount of pure-CPU
// busy-work (no blocking syscalls inside the loop, so the runticks
// each child accumulates measure real CPU competition, not I/O wait),
// and prints one machine-parseable report line per child. Consumed by
// tests/scheduler/*.py and captured into benchmarks/raw/ for
// docs/scheduler.md and benchmarks/methodology.md.
//
// Usage: schedbench policy tickets...
//   policy  -- SCHED_RR (0) or SCHED_LOTTERY (1), see kernel/sched.h
//   tickets -- one nonnegative ticket count per child to fork (one
//              child per argument, in argument order)
//
// Example: schedbench 1 60 20 10 5 5 -- five children under lottery
// scheduling with a 12:1 ticket spread between the busiest and
// quietest child.
//
// Why each child gets its own private pipe instead of just calling
// printf() straight to the console: up to NCPU children genuinely run
// concurrently (this project's QEMU session boots with -smp 3), and
// this xv6 pinned revision's printf()/fprintf() (user/printf.c) write
// one byte per write(2) syscall with no per-*line* atomicity -- two
// processes both mid-printf() at once produce a byte-interleaved,
// unparseable mess on a shared fd (verified empirically while
// calibrating this program; see docs/scheduler.md). Giving each child
// its own pipe makes that impossible (only that one child ever writes
// to it), and only the single-threaded parent ever writes to the real
// console, after collecting each child's report in turn.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/sched.h"
#include "user/user.h"
#include "user/acct.h"

// Each child spins until TARGET_TICKS timer ticks (kernel/trap.c's
// clockintr() fires roughly every tenth of a second) have passed
// since the whole benchmark's own start, checking elapsed wall time
// only once every BURST_ITERS arithmetic ops (an uptime() syscall
// every iteration would itself dominate the measurement). This is a
// wall-clock-bounded workload deliberately, not a fixed-iteration-
// count one: this project's own calibration found a fixed iteration
// count picked against one QEMU/host's throughput could be multiple
// orders of magnitude too fast or slow elsewhere, while "run until N
// wall ticks pass" gives every run the same *competition window* to
// measure fairness within, independent of raw emulated throughput.
// The fairness signal this produces is in how much of that fixed
// window each child's runticks/selections capture (docs/scheduler.md),
// not in "elapsed" (which converges to roughly TARGET_TICKS for every
// child, including a starved one, by construction).
#define TARGET_TICKS 30
#define BURST_ITERS  200000

// The real ceiling for a direct (non-shell) caller is NOT MAXARG-2
// (30): the parent keeps one pipe read-fd open per still-uncollected
// child (see the module comment above -- the write end is closed
// immediately, but the read end is retained until that child's report
// is drained), on top of the 3 fds a process already holds for
// stdin/stdout/stderr, all bounded by NOFILE (kernel/param.h, 16).
// Forking child i (0-indexed) needs 2 free fd slots for its pipe()
// call while i earlier read-ends are still retained, i.e. it needs
// 3 + i + 2 <= NOFILE -- so the largest i that can succeed is
// NOFILE - 6, i.e. NOFILE - 4 total children with NOFILE=16: 12, not
// 30. (In practice this project's own interactive/scripted shell
// caps real usage far below even that, at 7 ticket values --
// user/sh.c's MAXARGS=10, see "Shell argument limit" in
// docs/scheduler.md -- but a future direct caller should see the true
// bound here, not an aspirational one.)
#define MAX_CHILDREN (NOFILE - 4)

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
drain_pipe_to_console(int rfd)
{
  char buf[256];
  int n;

  while((n = read(rfd, buf, sizeof(buf))) > 0)
    write(1, buf, n);
  close(rfd);
}

int
main(int argc, char *argv[])
{
  if(argc < 3){
    fprintf(2, "Usage: %s policy tickets...\n", argv[0]);
    exit(1);
  }

  int policy = atoi(argv[1]);
  if(schedpolicy(policy) < 0){
    fprintf(2, "schedbench: schedpolicy(%d) rejected\n", policy);
    exit(1);
  }

  int n = argc - 2;
  if(n > MAX_CHILDREN){
    fprintf(2, "schedbench: too many children (%d > %d)\n", n, MAX_CHILDREN);
    exit(1);
  }
  int rfd[MAX_CHILDREN];
  int start = uptime();

  for(int i = 0; i < n; i++){
    int tickets = atoi(argv[2 + i]);
    int p[2];
    if(pipe(p) < 0){
      fprintf(2, "schedbench: pipe failed at child %d\n", i);
      exit(1);
    }

    int pid = fork();
    if(pid < 0){
      fprintf(2, "schedbench: fork failed at child %d\n", i);
      exit(1);
    }
    if(pid == 0){
      // child: writer only on its own pipe.
      close(p[0]);

      if(settickets(tickets) < 0){
        fprintf(2, "schedbench: settickets(%d) rejected\n", tickets);
        exit(1);
      }
      do_work(start);
      int end = uptime();

      struct xv_pstat st;
      if(xv_find_self(&st) < 0){
        fprintf(2, "schedbench: child could not find its own xvstat entry\n");
        exit(1);
      }
      fprintf(p[1],
              "schedbench: report pid=%d tickets=%d start=%d end=%d elapsed=%d "
              "runticks=%lu waitticks=%lu selections=%lu\n",
              getpid(), tickets, start, end, end - start,
              st.runticks, st.waitticks, st.selections);
      close(p[1]);
      exit(0);
    }

    // parent: reader only; release the write end immediately so this
    // process's own fd table doesn't accumulate one extra descriptor
    // per still-running child (NOFILE, kernel/param.h).
    close(p[1]);
    rfd[i] = p[0];
  }

  // Collect each child's report in argument order. read() on child i's
  // pipe blocks until that specific child exits (closing its own
  // write end) regardless of what order the other children finish in
  // -- their own output just waits harmlessly in their own pipe
  // buffers (well under PIPESIZE for one short report line) until
  // this loop gets to them.
  for(int i = 0; i < n; i++)
    drain_pipe_to_console(rfd[i]);

  for(int i = 0; i < n; i++)
    wait(0);

  // Restore the baseline policy so a later command in the same QEMU
  // session (including a later test in the same run) doesn't silently
  // keep running under lottery scheduling -- sched_policy is a global,
  // not scoped to this benchmark. See docs/scheduler.md "policy is
  // global, not per-benchmark" and
  // tests/scheduler/test_schedpolicy_validation.py.
  schedpolicy(SCHED_RR);

  printf("schedbench: done\n");
  exit(0);
}
