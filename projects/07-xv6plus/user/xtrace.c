// xv6-plus: xtrace -- userspace control tool for the syscall tracing
// framework (FR1, Phase 1). Original xv6-plus code, not upstream xv6.
//
// Usage: xtrace mask command [args...]
//
// Enables tracing on the calling (xtrace) process for the syscalls
// whose bit is set in mask, then exec()s into command. Because the
// kernel's trace_mask lives on the struct proc and survives exec()
// (kernel/exec.c never touches it) and is inherited across fork()
// (kernel/proc.c:kfork), this means:
//   - command runs with tracing enabled from its very first syscall.
//   - any children command forks are traced too, unless they call
//     trace(0) themselves.
//
// mask is a bitmask of syscall numbers from kernel/syscall.h, e.g.
// (1 << SYS_read) | (1 << SYS_write) to trace read and write.

#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int mask;

  if(argc < 3){
    fprintf(2, "Usage: %s mask command [args...]\n", argv[0]);
    exit(1);
  }

  mask = atoi(argv[1]);
  if(trace(mask) < 0){
    fprintf(2, "%s: trace failed\n", argv[0]);
    exit(1);
  }

  // argv[2..argc-1] is already NUL-pointer-terminated by the loader
  // (kernel/exec.c pushes a 0 after the last argument), so it can be
  // passed straight through as the new program's argv.
  exec(argv[2], &argv[2]);
  fprintf(2, "%s: exec %s failed\n", argv[0], argv[2]);
  exit(1);
}
