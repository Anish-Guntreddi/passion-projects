// xv6-plus: xvtopzombie -- deterministic support program for
// tests/xvtop/test_xvtop_zombie_filtered.py (invariant #5: "Zombie/
// free process slots never remain visible as active telemetry").
// Original xv6-plus test program, Phase 3.
//
// Forks a child that exits immediately and is deliberately never
// wait()ed on, so it sits in the proc table as a genuine ZOMBIE (not
// UNUSED) until this process itself exits and init reaps it. This
// process then polls xvstat(2) directly for that exact pid until it
// observes XV_ZOMBIE, so the exec() below is guaranteed to run only
// once the zombie definitely exists -- no race against the scheduler
// or a fixed pause(). Only then does it exec() into xvtop: kexec()
// (kernel/exec.c) replaces this process's program image but not its
// pid or its children, so the zombie child is still there, still
// parented to this same pid, for xvtop's very first refresh to see.
//
// Prints the zombie child's pid before exec'ing, so the test can
// confirm that exact pid never appears as a row in xvtop's output.

#include "kernel/types.h"
#include "kernel/param.h"
#include "kernel/pstat.h"
#include "user/user.h"

#define MAX_POLL_TICKS 50

int
main(void)
{
  int pid = fork();
  if(pid < 0){
    printf("xvtopzombie: fork failed\n");
    exit(1);
  }

  if(pid == 0)
    exit(0); // child: exit immediately, never reaped by this process

  struct xv_pstat st;
  int seen_zombie = 0;
  for(int t = 0; t < MAX_POLL_TICKS && !seen_zombie; t++){
    for(int i = 0; i < NPROC; i++){
      if(xvstat(i, &st) < 0)
        break; // idx out of range: no more proc-table slots to check
      if(st.pid == pid && st.state == XV_ZOMBIE){
        seen_zombie = 1;
        break;
      }
    }
    if(!seen_zombie)
      pause(1);
  }

  if(!seen_zombie){
    printf("xvtopzombie: child pid=%d never reached zombie state\n", pid);
    exit(1);
  }

  printf("xvtopzombie: child pid=%d is zombie, execing into xvtop\n", pid);

  char *args[] = {"xvtop", "1", 0};
  exec("xvtop", args);
  printf("xvtopzombie: exec failed\n");
  exit(1);
}
