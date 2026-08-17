// xv6-plus: xvtop -- userspace monitor over the Phase 2 accounting
// stats interface (FR4, Phase 3). Original xv6-plus code, not
// upstream xv6.
//
// Usage: xvtop [count [period]]
//   count  -- number of refreshes to print (default 1)
//   period -- ticks to pause() between refreshes (default 10 -- this
//             kernel's timer fires roughly every tenth of a second,
//             see kernel/trap.c's clockintr(), so 10 ticks is
//             roughly one second)
//
// Each refresh enumerates every in-use proc-table slot via xvstat(2)
// (idx = 0..NPROC-1, stopping early once xvstat reports idx is out of
// range), skips UNUSED and ZOMBIE slots (docs/invariants.md #5: a
// freed slot or an exited-but-unreaped slot must never look like an
// active process), sorts what's left by
// runticks descending -- the "sorting ... if manageable" half of FR4,
// implemented as a plain insertion sort since NPROC (64) is small
// enough that anything fancier would be optimizing before it's
// needed -- and prints one line per process: pid, state, memory size,
// runticks, waitticks, syscalls, name. See docs/xvtop.md for a
// captured transcript and the design writeup.

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"
#include "user/acct.h"

#define DEFAULT_COUNT  1
#define DEFAULT_PERIOD 10

static const char *
statename(int state)
{
  switch(state){
  case XV_UNUSED:   return "unused";
  case XV_USED:     return "used";
  case XV_SLEEPING: return "sleep";
  case XV_RUNNABLE: return "runble";
  case XV_RUNNING:  return "run";
  case XV_ZOMBIE:   return "zombie";
  default:          return "???";
  }
}

// Collect every in-use slot into rows[0..NPROC). Returns the count
// actually filled in.
static int
collect(struct xv_pstat *rows)
{
  int n = 0;
  struct xv_pstat st;

  for(int i = 0; i < NPROC; i++){
    if(xvstat(i, &st) < 0)
      break; // idx out of range: no more proc-table slots to check
    if(st.state == XV_UNUSED || st.state == XV_ZOMBIE)
      continue; // invariant #5: a free or exited-but-unreaped slot is
                // not active telemetry, even though ZOMBIE is a real
                // (non-UNUSED) state -- see tests/xvtop/test_xvtop_zombie_filtered.py
    rows[n++] = st;
  }
  return n;
}

// Sort rows[0..n) by runticks descending (busiest process first).
static void
sort_by_runticks(struct xv_pstat *rows, int n)
{
  for(int i = 1; i < n; i++){
    struct xv_pstat key = rows[i];
    int j = i - 1;
    while(j >= 0 && rows[j].runticks < key.runticks){
      rows[j + 1] = rows[j];
      j--;
    }
    rows[j + 1] = key;
  }
}

static void
print_snapshot(void)
{
  // static, not a stack local: NPROC * sizeof(struct xv_pstat) is a
  // few KB, and this xv6 build's user stack is a single page
  // (kernel/param.h: USERSTACK 1) -- the exact overflow risk that
  // motivated kernel/proc.c's procstat() to fill one caller-supplied
  // struct at a time instead of copying out a whole array. Static
  // storage puts this in the program's data/bss image (grown by
  // exec()'s normal uvmalloc, not the stack) instead.
  static struct xv_pstat rows[NPROC];
  int n = collect(rows);

  sort_by_runticks(rows, n);

  printf("PID\tSTATE\tSZ\tRUNTICKS\tWAITTICKS\tSYSCALLS\tNAME\n");
  for(int i = 0; i < n; i++){
    struct xv_pstat *r = &rows[i];
    printf("%d\t%s\t%lu\t%lu\t%lu\t%lu\t%s\n",
           r->pid, statename(r->state), r->sz, r->runticks, r->waitticks,
           r->syscalls, r->name);
  }
  printf("-- %d active process(es) --\n", n);
}

int
main(int argc, char *argv[])
{
  int count = DEFAULT_COUNT;
  int period = DEFAULT_PERIOD;

  if(argc > 1)
    count = atoi(argv[1]);
  if(argc > 2)
    period = atoi(argv[2]);
  if(count < 1)
    count = 1;

  for(int i = 0; i < count; i++){
    if(i > 0)
      pause(period);
    printf("=== xvtop refresh %d/%d ===\n", i + 1, count);
    print_snapshot();
  }

  exit(0);
}
