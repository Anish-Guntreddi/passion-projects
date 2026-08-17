// xv6-plus (Phase 2, FR3): shared kernel/user struct for the
// xvstat(2) statistics interface (ADR-0003: a dedicated syscall, not
// a procfs-like VFS). One `struct xv_pstat` is a point-in-time
// snapshot of one proc-table slot, as filled in by `procstat()`
// (kernel/proc.c) and returned by the xvstat(2) syscall
// (kernel/sysproc.c: sys_xvstat). See docs/accounting.md for the
// full design and docs/decisions/0009-accounting-counter-locking.md
// for exactly which fields carry a hard consistency guarantee and
// which are deliberately best-effort.
//
// Included directly by both kernel/*.c (kernel/proc.c,
// kernel/sysproc.c) and user/*.c (user/xvtop.c and the Phase 2 test
// programs under tests/accounting/), the same way user code already
// includes kernel/stat.h to get `struct stat` -- see any user/*.c
// file's #include block for the precedent.

#define XV_NAMESZ 16

// Mirrors `enum procstate` in kernel/proc.h (UNUSED=0 .. ZOMBIE=5).
// User code cannot include proc.h directly (it pulls in kernel-only
// types such as pagetable_t and struct trapframe), so the numeric
// values are duplicated here rather than shared via a common header.
// They are kept in sync by inspection: proc.h's enum is unmodified
// upstream xv6 (pinned at ADR-0001) and its declaration order is not
// itself an xv6-plus extension, so it is not expected to drift.
#define XV_UNUSED    0
#define XV_USED      1
#define XV_SLEEPING  2
#define XV_RUNNABLE  3
#define XV_RUNNING   4
#define XV_ZOMBIE    5

struct xv_pstat {
  int    pid;              // 0 if this proc-table slot is not in use
  int    state;             // one of the XV_* constants above
  uint64 sz;                // process memory size, in bytes
  uint64 runticks;          // ticks accounted as RUNNING (ADR-0007: tick granularity)
  uint64 waitticks;         // ticks accounted as SLEEPING (blocked/waiting)
  uint64 syscalls;          // total syscalls made since process creation
  char   name[XV_NAMESZ];   // process name (kernel/proc.h: p->name)

  // xv6-plus (Phase 4, FR5): lottery-scheduler fields. tickets mirrors
  // p->tickets (kernel/proc.h); selections mirrors p->sched_selections
  // -- see docs/scheduler.md and docs/decisions/0010-lottery-scheduler-design.md.
  uint64 tickets;
  uint64 selections;

  // xv6-plus (Phase 5, FR6): page-fault telemetry fields, mirroring
  // p->pagefaults / p->pagefaults_failed (kernel/proc.h) -- see
  // docs/vm-extension.md and
  // docs/decisions/0012-pagefault-telemetry-locking.md for why these
  // two, like sz/name above, are best-effort (not hard-guaranteed
  // torn-free) reads.
  uint64 pagefaults;
  uint64 pagefaults_failed;
};
