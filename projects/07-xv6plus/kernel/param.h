#define NPROC        64  // maximum number of processes
#define NCPU          8  // maximum number of CPUs
#define NOFILE       16  // open files per process
#define NFILE       100  // open files per system
#define NINODE       50  // maximum number of active i-nodes
#define NDEV         10  // maximum major device number
#define ROOTDEV       1  // device number of file system root disk
#define MAXARG       32  // max exec arguments
#define MAXOPBLOCKS  10  // max # of blocks any FS op writes
#define LOGBLOCKS    (MAXOPBLOCKS*3)  // max data blocks in on-disk log
#define NBUF         (MAXOPBLOCKS*3)  // size of disk block cache
// xv6-plus (Phase 6, FR7): bumped from upstream's 2000. The pinned
// upstream default only has enough free blocks (post-UPROGS) for the
// program count Phases 0-5 shipped with; adding Phase 6's four
// user/stress*.c binaries (~35-45KB/~35-45 blocks each, all baked
// into fs.img via UPROGS the same as every other user program) left
// so few blocks free that usertests' own writebig test (which needs
// MAXFILE == NDIRECT+NINDIRECT == 268 free blocks for one file alone,
// kernel/fs.h) failed with "balloc: out of blocks" -- a real, caught
// regression, not a hypothetical: usertests went from this project's
// established ALL TESTS PASSED baseline (docs/upstream-delta.md
// Phase 5) to failing at writebig the first time it was re-run after
// Phase 6's UPROGS additions. 20000 (10x, ~20MB disk image, trivial
// host-disk cost) restores comfortable headroom for both the current
// UPROGS set and ordinary usertests scratch-file usage, with room to
// spare for Phase 8. Every mkfs.c size formula (nbitmap, nmeta,
// nblocks) is already computed from this constant, so no other file
// needed a matching change. See docs/upstream-delta.md's Phase 6
// entry for the before/after usertests re-verification, and
// benchmarks/raw/usertests/ for the three raw transcripts (baseline,
// pre-fix regression repro, post-fix pass) that back it.
#define FSSIZE       20000  // size of file system in blocks
#define MAXPATH      128   // maximum file path name
#define USERSTACK    1     // user stack pages

