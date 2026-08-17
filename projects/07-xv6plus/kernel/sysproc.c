#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"
#include "vm.h"
#include "pstat.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  kexit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return kfork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return kwait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int t;
  int n;

  argint(0, &n);
  argint(1, &t);
  addr = myproc()->sz;

  if(t == SBRK_EAGER || n < 0) {
    if(growproc(n) < 0) {
      return -1;
    }
  } else {
    // Lazily allocate memory for this process: increase its memory
    // size but don't allocate memory. If the processes uses the
    // memory, vmfault() will allocate it.
    if(addr + n < addr)
      return -1;
    myproc()->sz += n;
  }
  return addr;
}

uint64
sys_pause(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  if(n < 0)
    n = 0;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kkill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// xv6-plus: FR1 syscall tracing control (Phase 1).
// Set the calling process's trace mask: bit i set means syscall
// number i is traced (printed by syscall() in syscall.c). mask == 0
// disables tracing. Only ever touches myproc()'s own trace_mask, so
// no lock beyond what already protects "private to the process"
// fields is required (see kernel/proc.h). The mask is inherited by
// children on fork() and survives exec(), so a parent can enable
// tracing and then exec() into the program it wants traced (see
// user/xtrace.c).
uint64
sys_trace(void)
{
  int mask;

  argint(0, &mask);
  myproc()->trace_mask = mask;
  return 0;
}

// xv6-plus: FR2/FR3 per-process accounting stats interface (Phase 2,
// ADR-0003). Fetches a point-in-time snapshot of proc-table slot idx
// into the user struct xv_pstat at addr. idx is a plain int (no user
// pointer to validate); addr is validated by copyout() itself, same
// as sys_fstat()'s filestat() (kernel/file.c) -- see invariant #7 in
// docs/invariants.md. All the actual field-reading logic lives in the
// small, independently-reasoned-about kernel helper procstat()
// (kernel/proc.c), keeping this syscall wrapper itself trivial per
// the handoff brief's "syscalls kept small" rule.
//
// Returns 0 and fills *addr on success, -1 if idx is out of
// [0, NPROC) range or the copyout itself fails (e.g. addr not
// mapped/writable in the caller's page table).
// xv6-plus: FR5 scheduler-experiment control (Phase 4). Thin wrapper
// per the handoff brief's "syscalls kept small" rule -- all the
// validation and locking lives in sched_settickets() (kernel/proc.c).
uint64
sys_settickets(void)
{
  int n;

  argint(0, &n);
  return sched_settickets(n);
}

// xv6-plus: FR5 scheduler-experiment control (Phase 4). Thin wrapper;
// see sched_setpolicy() (kernel/proc.c) for validation/locking.
uint64
sys_schedpolicy(void)
{
  int policy;

  argint(0, &policy);
  return sched_setpolicy(policy);
}

uint64
sys_xvstat(void)
{
  int idx;
  uint64 addr;
  struct xv_pstat st;

  argint(0, &idx);
  argaddr(1, &addr);

  if(procstat(idx, &st) < 0)
    return -1;

  if(copyout(myproc()->pagetable, addr, (char *)&st, sizeof(st)) < 0)
    return -1;

  return 0;
}
