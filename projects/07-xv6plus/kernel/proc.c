#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include "pstat.h"
#include "sched.h"

struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

// xv6-plus (Phase 4, FR5, ADR-0010): scheduler-experiment globals.
// `sched_policy` selects which of schedule_roundrobin()/
// schedule_lottery() every hart's scheduler() loop calls each time
// around; `rng_state` is the lottery draw's PRNG state. Both are
// protected by `sched_lock`, a new, deliberately leaf lock: it is
// never held while any p->lock is held, and nothing reachable while
// holding it ever acquires a p->lock, so it cannot introduce a cycle
// with the existing wait_lock -> p->lock order (see
// docs/decisions/0010-lottery-scheduler-design.md).
struct spinlock sched_lock;
int sched_policy = SCHED_RR;          // default: unchanged upstream behavior
static uint32 rng_state;              // xorshift32 state; see lottery_rand()

// Fixed, documented seed (not derived from `ticks`/boot entropy):
// ADR-0004 requires the lottery draw to be reproducible so a measured
// fairness result can be reproduced exactly, not just "randomized but
// plausible."
#define SCHED_RNG_SEED 2463534242u

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table.
void
procinit(void)
{
  struct proc *p;
  
  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  // xv6-plus (Phase 4, FR5, ADR-0010): scheduler-experiment lock and
  // PRNG seed, initialized once at boot on hart 0 (procinit() itself
  // only ever runs there -- see kernel/main.c).
  initlock(&sched_lock, "sched");
  rng_state = SCHED_RNG_SEED;
  for(p = proc; p < &proc[NPROC]; p++) {
      initlock(&p->lock, "proc");
      p->state = UNUSED;
      p->kstack = KSTACK((int) (p - proc));
  }
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int
cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu*
mycpu(void)
{
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc*
myproc(void)
{
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int
allocpid()
{
  int pid;
  
  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;

  // xv6-plus: fresh telemetry state for a freshly-allocated slot.
  // (freeproc() also clears this on free; set here too so the
  // invariant "telemetry is initialized at allocation" holds even if
  // that ever changes. See docs/invariants.md #3.)
  p->trace_mask = 0;
  // xv6-plus (Phase 2, FR2): fresh accounting counters for a
  // freshly-allocated slot. See the comment on these fields in
  // kernel/proc.h and docs/accounting.md.
  p->runticks = 0;
  p->waitticks = 0;
  p->syscall_count = 0;
  // xv6-plus (Phase 4, FR5): fresh scheduler-experiment state for a
  // freshly-allocated slot. tickets defaults to SCHED_DEFAULT_TICKETS
  // (kernel/sched.h), not 0 -- see the field comment in kernel/proc.h
  // for why a never-configured process still needs a normal share.
  p->tickets = SCHED_DEFAULT_TICKETS;
  p->sched_selections = 0;
  // xv6-plus (Phase 5, FR6): fresh page-fault telemetry for a
  // freshly-allocated slot. See kernel/proc.h and docs/vm-extension.md.
  p->pagefaults = 0;
  p->pagefaults_failed = 0;

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
  // xv6-plus: never let a reused proc-table slot inherit the previous
  // occupant's tracing state. See docs/invariants.md #5.
  p->trace_mask = 0;
  // xv6-plus (Phase 2, FR2): same rule for the accounting counters --
  // a reused slot must never show a previous occupant's runtime,
  // wait time, or syscall count. See docs/invariants.md #5 and
  // tests/accounting/test_slot_reuse_reset.py.
  p->runticks = 0;
  p->waitticks = 0;
  p->syscall_count = 0;
  // xv6-plus (Phase 4, FR5): same rule for the scheduler-experiment
  // fields -- a reused slot must never keep a previous occupant's
  // ticket count or selection history. See
  // tests/scheduler/test_tickets_lifecycle.py.
  p->tickets = 0;
  p->sched_selections = 0;
  // xv6-plus (Phase 5, FR6): same rule for page-fault telemetry.
  p->pagefaults = 0;
  p->pagefaults_failed = 0;
}

// Create a user page table for a given process, with no user memory,
// but with trampoline and trapframe pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

// Free a process's page table, and free the
// physical memory it refers to.
void
proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}

// Set up first user process.
void
userinit(void)
{
  struct proc *p;

  p = allocproc();
  initproc = p;
  
  p->cwd = namei("/");

  p->state = RUNNABLE;

  release(&p->lock);
}

// Shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint64 sz;
  struct proc *p = myproc();

  sz = p->sz;
  if(n > 0){
    if((sz = uvmalloc(p->pagetable, sz, sz + n, PTE_W)) == 0) {
      return -1;
    }
  } else if(n < 0){
    sz = uvmdealloc(p->pagetable, sz, sz + n);
  }
  p->sz = sz;
  return 0;
}

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int
kfork(void)
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // Allocate process.
  if((np = allocproc()) == 0){
    return -1;
  }

  // Copy user memory from parent to child.
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  // xv6-plus: tracing is inherited across fork() so that a traced
  // process's children are traced too, until they call trace() again
  // (e.g. trace(0) to opt out). See docs/tracing.md.
  np->trace_mask = p->trace_mask;

  // xv6-plus (Phase 4, FR5): tickets are inherited across fork(),
  // same rule and same reason as trace_mask just above -- a forked
  // workload keeps its parent's configured scheduling share until it
  // calls settickets() itself. No separate acquire needed: allocproc()
  // already returned with np->lock held (see its own doc comment),
  // and that same critical section is still open here (it isn't
  // released until `release(&np->lock)` below) -- this write already
  // has the same p->lock protection settickets()'s own write-side
  // locking gives the field (ADR-0010), it just doesn't need a new
  // acquire to get it.
  np->tickets = p->tickets;
  // np->sched_selections is already 0 from allocproc() above and
  // stays that way here -- deliberately NOT inherited, same rule as
  // the Phase 2 counters below (a child's own selection history
  // starts at birth, not copied from the parent's).

  // xv6-plus (Phase 2, FR2): deliberately NOT inherited. np's
  // accounting counters are already 0 from allocproc() above and
  // stay that way here -- a child's runtime/wait/syscall totals
  // describe its own execution from birth, not a copy of however
  // much the parent had already accumulated. This is the opposite
  // choice from trace_mask just above; see docs/accounting.md
  // "fork/exec/exit semantics" for the rationale and
  // tests/accounting/test_fork_fresh_counters.py for the test.

  pid = np->pid;

  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  release(&np->lock);

  return pid;
}

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void
reparent(struct proc *p)
{
  struct proc *pp;

  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp->parent == p){
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void
kexit(int status)
{
  struct proc *p = myproc();

  if(p == initproc)
    panic("init exiting");

  // Close all open files.
  for(int fd = 0; fd < NOFILE; fd++){
    if(p->ofile[fd]){
      struct file *f = p->ofile[fd];
      fileclose(f);
      p->ofile[fd] = 0;
    }
  }

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;

  release(&wait_lock);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
kwait(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    // Scan through table looking for exited children.
    havekids = 0;
    for(pp = proc; pp < &proc[NPROC]; pp++){
      if(pp->parent == p){
        // make sure the child isn't still in exit() or swtch().
        acquire(&pp->lock);

        havekids = 1;
        if(pp->state == ZOMBIE){
          // Found one.
          pid = pp->pid;
          if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate,
                                  sizeof(pp->xstate)) < 0) {
            release(&pp->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(pp);
          release(&pp->lock);
          release(&wait_lock);
          return pid;
        }
        release(&pp->lock);
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || killed(p)){
      release(&wait_lock);
      return -1;
    }
    
    // Wait for a child to exit.
    sleep(p, &wait_lock);  //DOC: wait-sleep
  }
}

// xv6-plus (Phase 4, FR5, ADR-0010): switch proc p into RUNNING on
// cpu c and run it until it swtch()es back here. Caller must hold
// p->lock and have already confirmed p->state == RUNNABLE; this
// helper flips the state, counts the selection, runs the process, and
// leaves the lock held on return (matching upstream's existing
// contract at every call site below -- the caller releases it).
// Factored out of the two policies below purely to keep
// sched_selections++ in exactly one place instead of two copies that
// could drift.
static void
run_selected(struct cpu *c, struct proc *p)
{
  p->state = RUNNING;
  p->sched_selections++;
  c->proc = p;
  swtch(&c->context, &p->context);
  c->proc = 0;
}

// Baseline scheduling policy: unmodified upstream xv6 round-robin,
// verbatim except for the run_selected() refactor above (which adds
// the sched_selections++ instrumentation but changes no control
// flow). One call scans the whole proc table once, in table order,
// and runs every RUNNABLE process it finds for one quantum before
// moving on. Returns 1 if at least one process ran, 0 if the table
// held no RUNNABLE process this pass.
static int
schedule_roundrobin(struct cpu *c)
{
  struct proc *p;
  int found = 0;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == RUNNABLE) {
      // Switch to chosen process.  It is the process's job
      // to release its lock and then reacquire it
      // before jumping back to us.
      run_selected(c, p);
      found = 1;
    }
    release(&p->lock);
  }
  return found;
}

// xv6-plus (Phase 4, FR5, ADR-0004/ADR-0010): xorshift32 PRNG, used
// only by the lottery draw below. Not cryptographic -- a small, fast,
// fixed-seed generator is exactly what a reproducible fairness
// benchmark needs (ADR-0004). Caller must hold sched_lock.
static uint32
lottery_rand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

// Sum p->tickets over every currently-RUNNABLE process, reading each
// one under that process's own p->lock (ADR-0006 case (b): tickets is
// cross-process-read here). NPROC is small (64), so one full-table
// scan per draw is cheap.
static uint64
runnable_ticket_total(void)
{
  struct proc *p;
  uint64 total = 0;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == RUNNABLE)
      total += p->tickets;
    release(&p->lock);
  }
  return total;
}

// Deterministic fallback: run the first RUNNABLE process found, in
// table order (i.e. behave like one round-robin step). Used when a
// weighted draw has nothing to draw from (every RUNNABLE process
// currently has 0 tickets -- see the zero-ticket-floor discussion in
// docs/decisions/0010-lottery-scheduler-design.md) and as the
// last-resort exit from schedule_lottery()'s bounded retry loop
// below. This is what keeps invariant #4 ("the scheduler always
// eventually selects eligible work") true even when the ticket-based
// draw can't make a choice.
static int
pick_first_runnable(struct cpu *c)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == RUNNABLE) {
      run_selected(c, p);
      release(&p->lock);
      return 1;
    }
    release(&p->lock);
  }
  return 0;
}

// One weighted draw against the RUNNABLE set, given its already-
// computed ticket total: acquire sched_lock just long enough to draw
// a winning ticket number, then walk the table again summing tickets
// in the same order until the running sum passes the winner. Returns
// 1 and has run the winner if the walk found it, 0 if it didn't (the
// RUNNABLE set or a ticket count changed between the caller's total
// and this walk -- a race, not an error; see schedule_lottery()).
static int
draw_and_run(struct cpu *c, uint64 total)
{
  struct proc *p;
  uint64 winner, cum = 0;

  acquire(&sched_lock);
  winner = lottery_rand() % total;
  release(&sched_lock);

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == RUNNABLE) {
      cum += p->tickets;
      if(cum > winner) {
        run_selected(c, p);
        release(&p->lock);
        return 1;
      }
    }
    release(&p->lock);
  }
  return 0;
}

// xv6-plus (Phase 4, FR5, ADR-0004/ADR-0010): lottery scheduling.
// Each call performs exactly one weighted draw and runs exactly one
// process (unlike schedule_roundrobin(), which runs every RUNNABLE
// process once per call) -- the standard shape for lottery scheduling
// papers, and a natural fit here since scheduler() already calls
// whichever policy function once per hart per re-entry.
//
// The two-pass design (sum tickets, then separately draw and walk
// again) cannot hold one lock across both passes without either
// acquiring every proc's lock at once (a new, much bigger lock-order
// surface) or introducing a single new giant lock that would
// serialize the whole table against every other p->lock use -- both
// rejected as disproportionate for a teaching kernel. Instead the two
// passes are allowed to race against a concurrent state change (a
// process exiting or blocking between them), bounded by a small
// retry loop: SCHED_LOTTERY_MAX_ATTEMPTS reflects how many times that
// specific race can plausibly recur back-to-back, not a tuned
// performance constant. See docs/decisions/0010-lottery-scheduler-design.md.
#define SCHED_LOTTERY_MAX_ATTEMPTS 4

static int
schedule_lottery(struct cpu *c)
{
  for(int attempt = 0; attempt < SCHED_LOTTERY_MAX_ATTEMPTS; attempt++) {
    uint64 total = runnable_ticket_total();
    if(total == 0)
      // No RUNNABLE process has a nonzero ticket count -- either
      // there is no RUNNABLE process at all, or every RUNNABLE
      // process currently has 0 tickets (e.g. a deliberately-starved
      // test workload). Either way there is nothing to weight-draw
      // from; fall back to picking the first RUNNABLE slot, exactly
      // like round-robin would.
      return pick_first_runnable(c);
    if(draw_and_run(c, total))
      return 1;
    // The draw missed (a race between the two passes); retry against
    // current state rather than reporting found=0 to scheduler()'s
    // wfi check below, which would incorrectly suspend this hart even
    // though RUNNABLE work still exists.
  }
  // Exhausted the retry budget (would require a fresh race on every
  // single attempt) -- fall back to a deterministic pick so
  // invariant #4 holds unconditionally, not just in the common case.
  return pick_first_runnable(c);
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run (per the current sched_policy).
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void)
{
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;){
    // The most recent process to run may have had interrupts
    // turned off; enable them to avoid a deadlock if all
    // processes are waiting. Then turn them back off
    // to avoid a possible race between an interrupt
    // and wfi.
    intr_on();
    intr_off();

    // xv6-plus (Phase 4, FR5): sched_policy is read here without
    // sched_lock -- a plain, naturally-aligned int read/write, same
    // "best-effort, not safety-critical" reasoning ADR-0009 already
    // applies to `ticks`: a policy switch (schedpolicy(2)) becoming
    // visible to this hart one iteration later than another is
    // acceptable for a scheduling *policy* knob (invariant #8 asks
    // that observability/extension code not destabilize the kernel,
    // not that a config change be instantaneous across all harts).
    int found = (sched_policy == SCHED_LOTTERY)
                  ? schedule_lottery(c)
                  : schedule_roundrobin(c);

    if(found == 0) {
      // nothing to run; stop running on this core until an interrupt.
      asm volatile("wfi");
    }
  }
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched RUNNING");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  sched();
  release(&p->lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void
forkret(void)
{
  extern char userret[];
  static int first = 1;
  struct proc *p = myproc();

  // Still holding p->lock from scheduler.
  release(&p->lock);

  if (first) {
    // File system initialization must be run in the context of a
    // regular process (e.g., because it calls sleep), and thus cannot
    // be run from main().
    fsinit(ROOTDEV);

    first = 0;
    // ensure other cores see first=0.
    __sync_synchronize();

    // We can invoke kexec() now that file system is initialized.
    // Put the return value (argc) of kexec into a0.
    p->trapframe->a0 = kexec("/init", (char *[]){ "/init", 0 });
    if (p->trapframe->a0 == -1) {
      panic("exec");
    }
  }

  // return to user space, mimicing usertrap()'s return.
  prepare_return();
  uint64 satp = MAKE_SATP(p->pagetable);
  uint64 trampoline_userret = TRAMPOLINE + (userret - trampoline);
  ((void (*)(uint64))trampoline_userret)(satp);
}

// Sleep on channel chan, releasing condition lock lk.
// Re-acquires lk when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();

  // xv6-plus (Phase 2, FR2): snapshot the tick count *before* taking
  // p->lock, and deliberately without tickslock. See
  // docs/decisions/0009-accounting-counter-locking.md: taking
  // tickslock while p->lock is held (as it is for the rest of this
  // function) would run p->lock-then-tickslock, the reverse of the
  // tickslock-then-p->lock order clockintr() already establishes
  // (it calls wakeup(&ticks) -- which acquires p->lock -- while
  // holding tickslock), and could deadlock against it. `ticks` is a
  // plain aligned word; RISC-V guarantees an unlocked load of it can
  // only ever observe a fully-formed old or new value, never a torn
  // one, which is all a best-effort accounting counter needs
  // (invariant #8) -- unlike sys_uptime()'s locked read, this one is
  // not claiming a precise, race-free snapshot.
  uint sleep_start = ticks;

  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock);  //DOC: sleeplock1
  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // xv6-plus (Phase 2, FR2): by the time sched() returns here, this
  // process has already been rescheduled and run again (that's what
  // "returned" means), so (ticks - sleep_start) is how long it was
  // blocked. Still holding p->lock (unchanged from upstream), so
  // this mutation is protected exactly like any other cross-process-
  // readable Phase 2 counter (see kernel/proc.h and ADR-0009).
  p->waitticks += (ticks - sleep_start);

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}

// Wake up all processes sleeping on channel chan.
// Caller should hold the condition lock.
void
wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
      }
      release(&p->lock);
    }
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int
kkill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

void
setkilled(struct proc *p)
{
  acquire(&p->lock);
  p->killed = 1;
  release(&p->lock);
}

int
killed(struct proc *p)
{
  int k;
  
  acquire(&p->lock);
  k = p->killed;
  release(&p->lock);
  return k;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int
either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if(user_dst){
    return copyout(p->pagetable, dst, src, len);
  } else {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int
either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if(user_src){
    return copyin(p->pagetable, dst, src, len);
  } else {
    memmove(dst, (char*)src, len);
    return 0;
  }
}

// xv6-plus (Phase 2, FR3): fill *out with a point-in-time snapshot of
// proc[idx]'s telemetry, for the xvstat(2) syscall (kernel/sysproc.c).
// Returns 0 on success (idx was in range), -1 if idx is outside
// [0, NPROC) -- the sentinel a caller enumerating idx = 0, 1, 2, ...
// (see user/xvtop.c, tests/accounting/, tests/xvtop/) uses to know it
// has reached the end of the table. A slot that is currently UNUSED
// (or was reaped since the caller last polled) is not an error: *out
// is still filled in, with pid == 0 and state == XV_UNUSED, and the
// caller decides whether to display it (docs/invariants.md #5: xvtop
// itself filters these out).
//
// Locking (docs/decisions/0009-accounting-counter-locking.md): every
// field is read while holding proc[idx]'s own p->lock. pid and state
// get a *hard* consistency guarantee from this, because every writer
// of those two fields already holds p->lock too (documented at the
// top of kernel/proc.h, unmodified upstream). sz and name do not:
// their existing writers (growproc(), kexec()) intentionally follow
// the lock-free "private to the process" convention and were not
// changed for this feature, so a concurrent grow/exec on the target
// can still be observed mid-update here. On RISC-V a naturally-
// aligned 64-bit load/store is a single atomic instruction, so this
// can never produce a torn/garbage sz value, only a momentarily stale
// one -- acceptable for a monitoring display and explicitly not a
// safety property this syscall relies on (invariant #8). runticks,
// waitticks, and syscall_count are new Phase 2 counters whose every
// writer (sleep(), syscall(), clockintr()) was deliberately written
// to also take p->lock, specifically so this function can give them
// the same hard guarantee as pid/state -- see the ADR for why Phase
// 1's trace_mask could skip locking entirely and Phase 2 cannot.
int
procstat(int idx, struct xv_pstat *out)
{
  if(idx < 0 || idx >= NPROC)
    return -1;

  struct proc *p = &proc[idx];

  acquire(&p->lock);
  out->pid = p->pid;
  out->state = p->state; // enum procstate values line up with XV_* (kernel/pstat.h)
  out->sz = p->sz;
  out->runticks = p->runticks;
  out->waitticks = p->waitticks;
  out->syscalls = p->syscall_count;
  safestrcpy(out->name, p->name, sizeof(out->name));
  // xv6-plus (Phase 4, FR5): tickets/selections. tickets is written
  // under p->lock by sched_settickets() below, so it gets the same
  // hard guarantee as runticks/waitticks/syscall_count; selections is
  // written under p->lock by run_selected() (this same file), same
  // guarantee.
  out->tickets = p->tickets;
  out->selections = p->sched_selections;
  // xv6-plus (Phase 5, FR6): pagefaults/pagefaults_failed. Read in
  // this same critical section but do NOT carry that hard guarantee
  // -- vmfault() (kernel/vm.c) deliberately writes them without
  // p->lock (docs/decisions/0012-pagefault-telemetry-locking.md), so
  // this is the same best-effort staleness class as sz/name above.
  out->pagefaults = p->pagefaults;
  out->pagefaults_failed = p->pagefaults_failed;
  release(&p->lock);

  return 0;
}

// xv6-plus (Phase 4, FR5, ADR-0010): set myproc()'s own ticket count.
// Rejects n < 0 (negative tickets have no meaning for a weighted
// draw) and returns -1 without changing anything; n == 0 is
// deliberately allowed (a process can be configured to only ever run
// via the zero-ticket floor -- see schedule_lottery()'s
// pick_first_runnable() fallback -- which is exactly how
// tests/scheduler/test_lottery_zero_ticket_floor.py exercises
// invariant #4's starvation bound). Also rejects n > SCHED_MAX_TICKETS
// (kernel/sched.h) -- without this bound, a large enough sum of
// RUNNABLE tickets would truncate through draw_and_run()'s uint32 PRNG
// modulo and make some processes structurally unreachable by any draw;
// see kernel/sched.h's comment on SCHED_MAX_TICKETS and
// docs/decisions/0013-ticket-count-bound.md. Takes p->lock on write
// because, unlike trace_mask, tickets is read cross-process by the
// lottery scheduler as a real scheduling input (ADR-0006 case (b));
// see the field comment in kernel/proc.h.
int
sched_settickets(int n)
{
  struct proc *p = myproc();

  if(n < 0 || n > SCHED_MAX_TICKETS)
    return -1;

  acquire(&p->lock);
  p->tickets = n;
  release(&p->lock);
  return 0;
}

// xv6-plus (Phase 4, FR5, ADR-0010): set the global scheduling policy
// for every hart's scheduler() loop. Rejects anything other than
// SCHED_RR/SCHED_LOTTERY (kernel/sched.h) and leaves sched_policy
// unchanged in that case -- deliberately, so a bad argument can never
// leave the scheduler dispatch in scheduler() looking at a value
// neither schedule_roundrobin() nor schedule_lottery() was written
// for (invariant #4). No process-identity check: this is a teaching
// kernel (spec Â§1.6 non-goal: production-grade security), and every
// other control syscall in this codebase (trace(), settickets() above)
// is equally unrestricted. Returns the *previous* policy on success,
// so a benchmark harness can restore it afterwards (see
// user/schedbench.c), or -1 if policy was invalid.
int
sched_setpolicy(int policy)
{
  int prev;

  if(policy != SCHED_RR && policy != SCHED_LOTTERY)
    return -1;

  acquire(&sched_lock);
  prev = sched_policy;
  sched_policy = policy;
  release(&sched_lock);
  return prev;
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}
