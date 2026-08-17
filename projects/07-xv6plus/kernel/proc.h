// Saved registers for kernel context switches.
struct context {
  uint64 ra;
  uint64 sp;

  // callee-saved
  uint64 s0;
  uint64 s1;
  uint64 s2;
  uint64 s3;
  uint64 s4;
  uint64 s5;
  uint64 s6;
  uint64 s7;
  uint64 s8;
  uint64 s9;
  uint64 s10;
  uint64 s11;
};

// Per-CPU state.
struct cpu {
  struct proc *proc;          // The process running on this cpu, or null.
  struct context context;     // swtch() here to enter scheduler().
  int noff;                   // Depth of push_off() nesting.
  int intena;                 // Were interrupts enabled before push_off()?
};

extern struct cpu cpus[NCPU];

// per-process data for the trap handling code in trampoline.S.
// sits in a page by itself just under the trampoline page in the
// user page table. not specially mapped in the kernel page table.
// uservec in trampoline.S saves user registers in the trapframe,
// then initializes registers from the trapframe's
// kernel_sp, kernel_hartid, kernel_satp, and jumps to kernel_trap.
// usertrapret() and userret in trampoline.S set up
// the trapframe's kernel_*, restore user registers from the
// trapframe, switch to the user page table, and enter user space.
// the trapframe includes callee-saved user registers like s0-s11 because the
// return-to-user path via usertrapret() doesn't return through
// the entire kernel call stack.
struct trapframe {
  /*   0 */ uint64 kernel_satp;   // kernel page table
  /*   8 */ uint64 kernel_sp;     // top of process's kernel stack
  /*  16 */ uint64 kernel_trap;   // usertrap()
  /*  24 */ uint64 epc;           // saved user program counter
  /*  32 */ uint64 kernel_hartid; // saved kernel tp
  /*  40 */ uint64 ra;
  /*  48 */ uint64 sp;
  /*  56 */ uint64 gp;
  /*  64 */ uint64 tp;
  /*  72 */ uint64 t0;
  /*  80 */ uint64 t1;
  /*  88 */ uint64 t2;
  /*  96 */ uint64 s0;
  /* 104 */ uint64 s1;
  /* 112 */ uint64 a0;
  /* 120 */ uint64 a1;
  /* 128 */ uint64 a2;
  /* 136 */ uint64 a3;
  /* 144 */ uint64 a4;
  /* 152 */ uint64 a5;
  /* 160 */ uint64 a6;
  /* 168 */ uint64 a7;
  /* 176 */ uint64 s2;
  /* 184 */ uint64 s3;
  /* 192 */ uint64 s4;
  /* 200 */ uint64 s5;
  /* 208 */ uint64 s6;
  /* 216 */ uint64 s7;
  /* 224 */ uint64 s8;
  /* 232 */ uint64 s9;
  /* 240 */ uint64 s10;
  /* 248 */ uint64 s11;
  /* 256 */ uint64 t3;
  /* 264 */ uint64 t4;
  /* 272 */ uint64 t5;
  /* 280 */ uint64 t6;
};

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

// Per-process state
struct proc {
  struct spinlock lock;

  // p->lock must be held when using these:
  enum procstate state;        // Process state
  void *chan;                  // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  int xstate;                  // Exit status to be returned to parent's wait
  int pid;                     // Process ID

  // wait_lock must be held when using this:
  struct proc *parent;         // Parent process

  // these are private to the process, so p->lock need not be held.
  uint64 kstack;               // Virtual address of kernel stack
  uint64 sz;                   // Size of process memory (bytes)
  pagetable_t pagetable;       // User page table
  struct trapframe *trapframe; // data page for trampoline.S
  struct context context;      // swtch() here to run process
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)

  // xv6-plus: syscall tracing (Phase 1, FR1). Bit i set => syscall
  // number i is traced for this process. Mutated only by the owning
  // process itself (via the trace() syscall, on its own struct proc)
  // and read only from that same process's own syscall-dispatch path,
  // so it follows the same "private to the process" convention as the
  // fields above and needs no additional locking. It is copied to
  // children on fork() so tracing is inherited, and reset to 0 in
  // freeproc() so a reused proc-table slot never inherits a stale
  // tracing state from a previous occupant (docs/invariants.md #3, #5).
  int trace_mask;

  // xv6-plus: per-process accounting (Phase 2, FR2/FR3). Deliberately
  // NOT grouped with trace_mask above under the "owner mutates, no
  // lock" rule: unlike trace_mask (read only by the owning process's
  // own syscall-dispatch code), these three counters are read
  // cross-process by the xvstat(2) syscall (kernel/sysproc.c) walking
  // the whole proc table, so every writer explicitly holds p->lock
  // around each mutation -- see
  // docs/decisions/0009-accounting-counter-locking.md for the full
  // reasoning (including a lock-order hazard that rules out reusing
  // tickslock for this) and docs/accounting.md for the design
  // writeup. Initialized to 0 in allocproc() and reset to 0 again in
  // freeproc(), exactly like trace_mask. UNLIKE trace_mask: NOT
  // copied parent->child in kfork() (each process's own counters
  // start at 0 -- see the comment at that call site) and, like
  // trace_mask, NOT touched by kexec() (same struct proc/pid across
  // exec, so accounting is a per-pid-lifetime total, not a
  // per-program-image one).
  uint64 runticks;      // ticks this process has spent RUNNING (ADR-0007)
  uint64 waitticks;     // ticks this process has spent SLEEPING (blocked)
  uint64 syscall_count; // total syscalls made since process creation

  // xv6-plus: scheduler experiment (Phase 4, FR5, ADR-0004, ADR-0010).
  //
  // tickets: this process's lottery-ticket count (>=0). Mutated by the
  // owning process itself via settickets() (kernel/sysproc.c) -- BUT
  // unlike trace_mask, it is also read cross-process by the lottery
  // scheduler (kernel/proc.c: schedule_lottery(), which may run on any
  // hart) as a genuine scheduling input, not just a monitoring
  // display. So settickets() takes p->lock on write, matching
  // ADR-0006 case (b) rather than case (a), even though the writer is
  // always the owning process itself -- see
  // docs/decisions/0010-lottery-scheduler-design.md. Defaults to
  // SCHED_DEFAULT_TICKETS at allocation (kernel/sched.h); inherited
  // across fork() (like trace_mask, unlike the Phase 2 accounting
  // counters just above) so a forked workload keeps its parent's
  // configured share; reset to 0 on free.
  int tickets;
  // sched_selections: number of times the scheduler (under either
  // policy) has chosen this process to run. Mutated only by
  // scheduler() itself, and only ever while already holding this
  // exact process's p->lock (the same critical section that flips
  // p->state to RUNNING) -- so, like runticks, no additional lock is
  // needed. Not inherited across fork() (like the Phase 2 counters:
  // a child's own selection history is its own, not a copy of the
  // parent's). Reset to 0 on allocation and free.
  uint64 sched_selections;

  // xv6-plus: VM extension -- page-fault telemetry (Phase 5, FR6,
  // ADR-0011/ADR-0012). Counts vmfault() (kernel/vm.c) outcomes for
  // this process: pagefaults for a successful on-demand page-in
  // (lazy-sbrk'd memory touched for the first time), pagefaults_failed
  // for a recognized-but-unserviceable fault (address at/beyond p->sz,
  // physical memory exhausted, or a permission violation on an
  // already-mapped page -- see vm_permission_violation(),
  // kernel/vm.c). Mutated only by vmfault() acting on
  // myproc() -- i.e. only ever by the one hart currently running this
  // exact process -- so, like trace_mask, this follows the lock-free
  // "owner-only mutation" convention (ADR-0006 case (a)), even though,
  // like sz/name (ADR-0009), it is read cross-process by xvstat(2)
  // without a hard torn-read guarantee on the write side. That gap is
  // deliberate, not an oversight: see
  // docs/decisions/0012-pagefault-telemetry-locking.md for the
  // specific lock-order hazard (kwait()'s copyout() running while
  // holding the reaped child's p->lock) that rules out taking p->lock
  // inside vmfault(). Reset to 0 on allocation and free; NOT inherited
  // across fork() (like the Phase 2 counters -- a child's own fault
  // history is its own).
  uint64 pagefaults;
  uint64 pagefaults_failed;
};
