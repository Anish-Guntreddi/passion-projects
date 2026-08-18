// xv6-plus: stressvm -- Phase 6 stress/race hardening (spec roadmap
// row 6: "memory-pressure ... stress"). Original xv6-plus test
// program.
//
// Phase 5's own VM tests (tests/vm/) each drive exactly one process
// through vmfault()'s lazy-allocation path at a time. This program
// instead runs N_WORKERS processes calling vmfault() -- which itself
// calls kalloc()/mappages() (kernel/vm.c, kernel/kalloc.c, both
// upstream, unmodified) -- *simultaneously* across every hart (-smp
// 3), the first genuinely concurrent exercise of that fault-in path
// and of kalloc()'s own freelist lock in this project. Two parts:
//
// Part 1 (ROUNDS clean grow/touch/shrink cycles, all N_WORKERS
// concurrent): each worker sbrklazy()s PAGES_PER_ROUND pages, touches
// every one (forcing PAGES_PER_ROUND real vmfault() calls racing its
// siblings' identical calls on the other harts), then sbrk()s back
// down by the same amount (eager free -- kernel/sysproc.c's sys_sbrk()
// always takes the eager growproc() path for n < 0, regardless of the
// SBRK_LAZY flag passed to sbrklazy() for growth). Reports its own
// pagefaults/pagefaults_failed via xvstat(2) before/after the whole
// loop (before/after idiom, same reasoning as user/stresssyscalls.c
// and user/accttest.c) over its own private pipe -- a real,
// empirically-hit bug in an earlier version of this file printed
// straight to the console from all N_WORKERS concurrent clean workers
// and produced a byte-interleaved, unparseable mess (this pinned
// revision's printf()/fprintf() write one byte per write(2) syscall
// with no per-line atomicity -- the exact hazard user/schedbench.c's
// module comment already documents and works around the same way).
// Well within the ~128MB physical budget even with all N_WORKERS
// concurrent (N_WORKERS * PAGES_PER_ROUND * 4KB is a few MB), so
// pagefaults_failed must stay at 0 here -- any nonzero value would
// mean concurrent kalloc() pressure is spuriously failing allocations
// that should succeed.
//
// Part 2 (concurrent real exhaustion): N_WORKERS fresh children all
// race to lazily grow+touch pages up to a large PAGE_BUDGET
// (user/vmexhausttest.c's own budget and rationale: comfortably above
// the ~128MB actually available, kernel/memlayout.h's PHYSTOP,
// docs/vm-extension.md "memory-exhaustion test configuration") at the
// *same time* -- unlike vmexhausttest.c's single child, this shares
// the one physical free-page pool across N_WORKERS simultaneous
// consumers, directly exercising kalloc()'s freelist lock under real
// multi-hart contention as it runs out. Each worker that hits real
// exhaustion is killed cleanly via the same recognized-but-
// unserviceable vmfault() path Phase 5 already tests (xstatus == -1),
// not a kernel panic -- proving invariant #6 (VM changes preserve
// isolation; failure is contained to the faulting process) and
// invariant #2 (no lock-order hazard: kalloc()'s lock is a leaf lock,
// untouched by this project) hold under real concurrent pressure, not
// just the single-process case Phase 5 covered. A final small
// allocate-and-touch after every worker is reaped proves full
// reclamation (invariant #3/#5), the same recovery check
// vmexhausttest.c already uses.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/riscv.h"
#include "user/user.h"
#include "user/acct.h"

#define N_WORKERS         3   // == NCPU (this project's QEMU -smp 3): guarantees
                               // true one-per-hart concurrency, not just interleaving.
#define ROUNDS            4
#define PAGES_PER_ROUND 300
#define PAGE_BUDGET   40000   // same budget/rationale as user/vmexhausttest.c
#define RECOVERY_PAGES    4

static int
grow_and_touch_one_page(void)
{
  char *p = sbrklazy(PGSIZE);
  if(p == SBRK_ERROR)
    return -1;
  *p = 1;
  return 0;
}

static void
clean_worker(int idx, int wfd)
{
  struct xv_pstat before, after;

  if(xv_find_self(&before) < 0){
    fprintf(wfd, "stressvm: clean worker %d could not find self (before)\n", idx);
    exit(1);
  }

  for(int r = 0; r < ROUNDS; r++){
    for(int i = 0; i < PAGES_PER_ROUND; i++){
      if(grow_and_touch_one_page() < 0){
        fprintf(wfd, "stressvm: clean worker %d round %d page %d unexpectedly failed\n",
                idx, r, i);
        exit(1);
      }
    }
    sbrk(-(PAGES_PER_ROUND * PGSIZE));
  }

  if(xv_find_self(&after) < 0){
    fprintf(wfd, "stressvm: clean worker %d could not find self (after)\n", idx);
    exit(1);
  }

  fprintf(wfd, "stressvm: clean idx=%d pid=%d pagefaults_before=%lu pagefaults_after=%lu "
          "failed_before=%lu failed_after=%lu\n",
          idx, getpid(), before.pagefaults, after.pagefaults,
          before.pagefaults_failed, after.pagefaults_failed);
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

// Reports through its own pipe too, same reason as clean_worker()
// above: if memory is somehow never actually exhausted before
// PAGE_BUDGET (unexpected in this project's fixed 128MB build, but
// not structurally impossible), multiple concurrent exhaust workers
// reaching that message at once would hit the identical
// console-interleaving hazard. In the expected case (real exhaustion),
// this process is killed by the kernel's own recognized-but-
// unserviceable page-fault path before it ever reaches a printf call
// here at all -- the pipe is a defensive consistency measure, not
// something the normal run relies on draining.
static void
exhaust_worker(int idx, int wfd)
{
  for(int i = 0; i < PAGE_BUDGET; i++){
    if(grow_and_touch_one_page() < 0){
      fprintf(wfd, "stressvm: exhaust worker %d sz-growth failed at page %d\n", idx, i);
      exit(1);
    }
  }
  fprintf(wfd, "stressvm: exhaust worker %d reached PAGE_BUDGET=%d without exhausting memory\n",
          idx, PAGE_BUDGET);
  exit(0);
}

int
main(void)
{
  // Part 1: clean concurrent grow/touch/shrink cycles. Each worker
  // reports over its own private pipe (see clean_worker()'s comment
  // above) so N_WORKERS concurrent finishers can never interleave
  // their console output.
  int clean_rfd[N_WORKERS];
  for(int i = 0; i < N_WORKERS; i++){
    int p[2];
    if(pipe(p) < 0){
      fprintf(2, "stressvm: pipe failed at clean worker %d\n", i);
      exit(1);
    }
    int pid = fork();
    if(pid < 0){
      fprintf(2, "stressvm: clean fork failed at worker %d\n", i);
      exit(1);
    }
    if(pid == 0){
      close(p[0]);
      clean_worker(i, p[1]);
      // not reached
    }
    close(p[1]);
    clean_rfd[i] = p[0];
  }
  for(int i = 0; i < N_WORKERS; i++)
    drain_pipe_to_console(clean_rfd[i]);
  for(int i = 0; i < N_WORKERS; i++){
    int xstatus = -1;
    wait(&xstatus);
    if(xstatus != 0){
      printf("stressvm: clean phase FAILED, a worker exited %d\n", xstatus);
      exit(1);
    }
  }
  printf("stressvm: clean phase done\n");

  // Part 2: concurrent real exhaustion, same shared physical pool.
  // Each worker gets a pipe too (see exhaust_worker()'s comment) --
  // in the expected case every worker is killed by the kernel before
  // writing anything, so draining it below just observes an immediate
  // EOF (every writer fd closed), harmlessly.
  int exhaust_rfd[N_WORKERS];
  for(int i = 0; i < N_WORKERS; i++){
    int p[2];
    if(pipe(p) < 0){
      fprintf(2, "stressvm: pipe failed at exhaust worker %d\n", i);
      exit(1);
    }
    int pid = fork();
    if(pid < 0){
      fprintf(2, "stressvm: exhaust fork failed at worker %d\n", i);
      exit(1);
    }
    if(pid == 0){
      close(p[0]);
      exhaust_worker(i, p[1]);
      // not reached
    }
    close(p[1]);
    exhaust_rfd[i] = p[0];
  }
  for(int i = 0; i < N_WORKERS; i++)
    drain_pipe_to_console(exhaust_rfd[i]);

  int killed_count = 0;
  for(int i = 0; i < N_WORKERS; i++){
    int xstatus = 0;
    int wpid = wait(&xstatus);
    printf("stressvm: exhaust reap wpid=%d xstatus=%d\n", wpid, xstatus);
    if(xstatus == -1)
      killed_count++;
  }
  printf("stressvm: exhaust phase killed=%d/%d\n", killed_count, N_WORKERS);

  // Recovery check: a fresh, modest allocation must still succeed,
  // proving every exhausted worker's pages were fully reclaimed.
  int rpid = fork();
  if(rpid < 0){
    fprintf(2, "stressvm: recovery fork failed\n");
    exit(1);
  }
  if(rpid == 0){
    for(int i = 0; i < RECOVERY_PAGES; i++){
      if(grow_and_touch_one_page() < 0){
        printf("stressvm: recovery FAILED at page %d -- memory not reclaimed?\n", i);
        exit(1);
      }
    }
    printf("stressvm: recovery ok, touched %d pages\n", RECOVERY_PAGES);
    exit(0);
  }
  int rstatus = 0;
  wait(&rstatus);
  printf("stressvm: recovery xstatus=%d\n", rstatus);

  printf("stressvm: done\n");
  exit(0);
}
