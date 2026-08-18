// xv6-plus: stresssyscalls -- Phase 6 stress/race hardening (spec
// roadmap row 6: "syscall-heavy ... stress"). Original xv6-plus test
// program.
//
// Forks N_WORKERS children that simultaneously hammer the syscall
// dispatch path (kernel/syscall.c: syscall()) with a tight getpid()
// loop -- genuinely concurrent across every hart (-smp 3), unlike
// tests/accounting/test_syscall_count_monotonic.py's single sequential
// process. Each worker snapshots its own syscall_count via xvstat(2)
// before and after its loop (user/acct.h's xv_find_self(), the same
// before/after idiom user/accttest.c already uses, not an exact-count
// match -- xv_find_self()'s own scan cost varies with this worker's
// proc-table index, so ">= N more" is the right, non-fragile
// assertion, same reasoning tests/accounting/test_syscall_count_monotonic.py
// already documents) and reports over its own private pipe
// (user/schedbench.c's pattern: this pinned xv6 revision's printf()
// writes one byte per write(2) with no per-line atomicity, so N_WORKERS
// processes printing straight to the shared console at once would
// interleave into an unparseable mess -- verified empirically during
// user/schedbench.c's own calibration, see docs/scheduler.md).
//
// Exercises invariant #1 (syscall_count must never corrupt or lose
// updates under real concurrent p->lock traffic -- one lock per
// process, so this is about the dispatch path and scheduler churn
// around it, not lock contention on a single lock) and invariant #8
// (heavy syscall volume across all harts at once must not destabilize
// the kernel -- checked by a plain regression command afterward).

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"
#include "user/acct.h"

#define N_WORKERS             4
#define SYSCALLS_PER_WORKER 6000

static void
worker(int idx, int wfd)
{
  struct xv_pstat before, after;

  if(xv_find_self(&before) < 0){
    fprintf(wfd, "stresssyscalls: worker %d could not find self (before)\n", idx);
    exit(1);
  }

  for(int i = 0; i < SYSCALLS_PER_WORKER; i++)
    getpid();

  if(xv_find_self(&after) < 0){
    fprintf(wfd, "stresssyscalls: worker %d could not find self (after)\n", idx);
    exit(1);
  }

  fprintf(wfd, "stresssyscalls: report idx=%d pid=%d before=%lu after=%lu\n",
          idx, getpid(), before.syscalls, after.syscalls);
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
  int rfd[N_WORKERS];

  for(int i = 0; i < N_WORKERS; i++){
    int p[2];
    if(pipe(p) < 0){
      fprintf(2, "stresssyscalls: pipe failed at worker %d\n", i);
      exit(1);
    }
    int pid = fork();
    if(pid < 0){
      fprintf(2, "stresssyscalls: fork failed at worker %d\n", i);
      exit(1);
    }
    if(pid == 0){
      close(p[0]);
      worker(i, p[1]);
      // not reached
    }
    close(p[1]);
    rfd[i] = p[0];
  }

  for(int i = 0; i < N_WORKERS; i++)
    drain_pipe_to_console(rfd[i]);
  for(int i = 0; i < N_WORKERS; i++)
    wait(0);

  printf("stresssyscalls: done\n");
  exit(0);
}
