// System call numbers
#define SYS_fork    1
#define SYS_exit    2
#define SYS_wait    3
#define SYS_pipe    4
#define SYS_read    5
#define SYS_kill    6
#define SYS_exec    7
#define SYS_fstat   8
#define SYS_chdir   9
#define SYS_dup    10
#define SYS_getpid 11
#define SYS_sbrk   12
#define SYS_pause  13
#define SYS_uptime 14
#define SYS_open   15
#define SYS_write  16
#define SYS_mknod  17
#define SYS_unlink 18
#define SYS_link   19
#define SYS_mkdir  20
#define SYS_close  21
// xv6-plus: FR1 syscall tracing control (Phase 1). Not part of
// upstream xv6; see docs/upstream-delta.md.
#define SYS_trace  22
// xv6-plus: FR2/FR3 per-process accounting stats interface (Phase 2).
// Not part of upstream xv6; see docs/upstream-delta.md.
#define SYS_xvstat 23
