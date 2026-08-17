// xv6-plus (Phase 4, FR5, ADR-0004/ADR-0010): scheduler policy
// constants and defaults. Included directly by both kernel/*.c
// (kernel/proc.c, kernel/sysproc.c) and user/*.c (user/schedbench.c
// and the Phase 4 test programs under tests/scheduler/) -- the same
// convention kernel/vm.h already uses for SBRK_EAGER/SBRK_LAZY, and
// ulib.c/user programs already include it directly for that reason.

// schedpolicy(2) argument / kernel global `sched_policy` values.
#define SCHED_RR      0   // baseline round-robin (upstream xv6 behavior; default)
#define SCHED_LOTTERY 1   // xv6-plus: lottery scheduling (ADR-0004)

// Ticket count assigned to every process at allocproc() time, before
// any settickets(2) call. Deliberately nonzero: this is what keeps a
// brand-new process (which has never called settickets()) eligible
// for a fair lottery share by default, rather than needing every
// program to opt in just to run normally under SCHED_LOTTERY.
#define SCHED_DEFAULT_TICKETS 10

// Upper bound accepted by settickets(2) for a single process's ticket
// count (see kernel/proc.c: sched_settickets()). This exists to keep
// runnable_ticket_total()'s uint64 sum (kernel/proc.c) well clear of
// draw_and_run()'s `winner = lottery_rand() % total`, where
// lottery_rand() returns a plain uint32 (xorshift32): if the summed
// ticket total across every RUNNABLE process ever exceeded UINT32_MAX
// (~4.29 billion), `winner` could never land past that point, making
// any process whose cumulative ticket range starts beyond it
// permanently unreachable by any draw -- a real correctness bug, not
// a theoretical one, since settickets(2) is a public, unrestricted
// syscall (docs/decisions/0013-ticket-count-bound.md). With
// NPROC (kernel/param.h) capped at 64, NPROC * SCHED_MAX_TICKETS is
// still over 600x below UINT32_MAX even if every single process slot
// were simultaneously maxed out, while still leaving orders of
// magnitude more dynamic range between ticket tiers than any
// benchmark workload in this project actually uses (docs/scheduler.md:
// the widest captured spread is 12x, 60 vs. 5 tickets).
#define SCHED_MAX_TICKETS 100000
