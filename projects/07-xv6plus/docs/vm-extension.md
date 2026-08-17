# VM extension: page-fault telemetry (FR6, Phase 5)

## What it is, and what it isn't (honesty layer, spec §1.2)

**Not original:** the underlying lazy-allocation fault-in mechanism.
This pinned upstream revision (`mit-pdos/xv6-riscv @ xv6-riscv-rev5`,
ADR-0001) already ships `sys_sbrk()`'s eager/lazy split
(`kernel/vm.h`: `SBRK_EAGER`/`SBRK_LAZY`) and `vmfault()`
(`kernel/vm.c`), wired into `kernel/trap.c`'s `usertrap()` and into
`copyout()`/`copyin()`. `sbrklazy()` growing a process's `sz` without
mapping anything, and a first touch allocating a fresh zeroed page on
demand, are upstream (A) behavior, unmodified by this project.

**Original (C):** everything this project adds *on top of* that
mechanism -- see ADR-0011 for why page-fault telemetry (one of FR6's
own listed options) was chosen specifically because that fault-in
seam already existed, and `docs/upstream-delta.md`'s Phase 5 entry for
the exact file-by-file split.

## What was added

- **Per-process counters** (`kernel/proc.h`): `uint64 pagefaults`
  (successful on-demand page-ins) and `uint64 pagefaults_failed`
  (recognized-but-unserviceable faults: out-of-bounds address, or
  `kalloc()`/`mappages()` failure). Incremented inside `vmfault()`
  itself (`kernel/vm.c`), so every call site (`usertrap()`,
  `copyout()`, `copyin()`) is covered uniformly.
- **`kernel/pstat.h`**: `struct xv_pstat` gains
  `pagefaults`/`pagefaults_failed`, filled by `procstat()` -- exposed
  through the existing `xvstat(2)` interface (ADR-0003). **No new
  syscall** was needed for this feature; see ADR-0012 for why (and for
  the counters' locking, deliberately lock-free -- see below).
- **`kernel/trap.c` diagnostic split**: upstream's combined
  `(scause == 13 || scause == 15) && vmfault(...) != 0` condition is
  split into its own branch. The *decision* (continue on success, kill
  on failure) is byte-for-byte unchanged; a recognized-but-failed
  page fault now prints its own diagnostic instead of falling into the
  same generic "unexpected scause" message a truly-unrecognized trap
  type gets. As of a review-followup fix (below), that diagnostic is
  itself split in two: `"...(invalid address or out of memory)..."`
  for an out-of-bounds address or a `kalloc()`/`mappages()` failure,
  and `"...(permission violation: <read|write> access to a page mapped
  without sufficient permission)..."` for a genuine permission
  violation on an already-mapped page. This is the message every test
  below actually greps the console for.
- **Five dedicated test programs** (`user/vm*.c`) and **five dedicated
  test files** (`tests/vm/`), as originally built in Phase 5 -- covering
  every VM benchmark-plan bullet from spec §1.9 for the first time in
  this project (Phase 3's own invariants table marked invariant #6
  "Not applicable: Phases 1-3 make no VM changes" -- meaning this
  already-shipping fault path had zero dedicated tests before Phase
  5). Two more of each were added during review followup (below), for
  seven total.
- **Review followup (two gaps found by external review, both fixed):**
  - **Permission-fault classification**: `ismapped()` only ever
    checked `PTE_V`, not permission bits, so a genuine permission
    violation (e.g. a process writing to its own read-only text
    segment) fell into `vmfault()`'s "already mapped" branch --
    neither counted into `pagefaults_failed` nor distinguished from a
    benign concurrent-fault race, and misdiagnosed by
    `kernel/trap.c` as "invalid address or out of memory." A new
    helper, `vm_permission_violation(pagetable, va, read)`
    (`kernel/vm.c`), checks the actual permission bit (`PTE_R` for a
    load, `PTE_W` for a store) the faulting access needs against
    what's mapped; `vmfault()` uses it to count only the genuine
    violation case, and `kernel/trap.c` uses it again to print the
    correct message. New test: `user/vmpermtest.c` /
    `tests/vm/test_permission_fault_killed.py`.
  - **`pagefaults_failed` was never directly verified**: every
    pre-existing failure-path test only *inferred* the counter
    incremented from a killed process's exit status (`xstatus == -1`)
    -- none ever read the counter's own value, since the faulting
    process there is killed by the very fault it triggered and never
    survives to check it. New test: `user/vmpfailcount.c` /
    `tests/vm/test_pagefault_failed_counted.py`, which triggers
    `vmfault()`'s failure path through a deliberately out-of-bounds
    `xvstat(2)` output pointer instead -- a failing `copyout()` inside
    `sys_xvstat()` just returns -1 to the syscall without killing the
    caller, so that same process reads its own `pagefaults_failed`
    both before and after and the test asserts the exact `+1`.

## Locking: deliberately lock-free (ADR-0012)

`vmfault()` always mutates `myproc()`'s own counters -- only the one
hart currently running that exact process ever touches them at this
code point, so there is no concurrent-writer race to protect against.
Taking `p->lock` inside `vmfault()` was considered and **rejected**:
`kernel/proc.c`'s `kwait()` calls `copyout()` on the *waiting parent's*
page table while still holding the *reaped child's* `p->lock` --
if that `copyout()` triggered a lazy fault on the parent's own memory,
and `vmfault()` then acquired the parent's own `p->lock`, that would
introduce a lock-order edge (child's lock, then parent's lock) that
does not exist anywhere else in this codebase, and that ADR-0009's own
reasoning explicitly assumes never happens. See ADR-0012 for the full
trace. The counters follow the exact same "owner-only mutation, no
lock" convention as Phase 1's `trace_mask` and the exact same
"best-effort, not hard-guaranteed" cross-process-read staleness class
ADR-0009 already accepted for `sz`/`name`.

## Lifecycle: init/fork/exec/exit semantics

- **`allocproc()`/`freeproc()`**: both counters zeroed on allocation
  and again on free (invariant #5), same pattern as every earlier
  phase.
- **`kfork()`**: **not** inherited (like the Phase 2 accounting
  counters) -- a child's own fault history is its own, not a copy of
  the parent's. Directly tested: `tests/vm/test_fork_lazy_region.py`
  proves a child touching a region the parent `sbrklazy()`'d (but
  never touched) triggers the *child's own* independent `vmfault()`
  call (`pagefaults == 1` for the child) while the parent's own
  counter stays at 0 -- `uvmcopy()` (`kernel/vm.c`) already skips
  copying unmapped pages, so there was nothing shared to inherit in
  the first place; the counters just confirm it.
- **`kexec()`**: `kexec()` builds the new image in a fresh page table
  and only frees the *old* one (`uvmfree()`, whatever happened to be
  mapped there -- nothing, for a never-touched lazy region) after
  committing. `tests/vm/test_exec_discards_lazy_region.py` proves this
  doesn't crash or leak for a process holding a pending, untouched
  lazy region at `exec()` time.

## Correctness coverage (spec §1.9 VM benchmark plan, bullet by bullet)

| Spec §1.9 bullet | Test |
|---|---|
| Invalid address/permission cases | `test_oob_access_killed.py` (invalid address: `va >= p->sz`); `test_permission_fault_killed.py` (permission: an in-bounds, already-mapped page accessed without sufficient permission -- added during review followup, since the "permission" half of this bullet was previously claimed-covered but not actually exercised) |
| Fork/exec interactions | `test_fork_lazy_region.py`, `test_exec_discards_lazy_region.py` |
| Mapping/allocation boundaries | `test_pagefault_counting.py` (sbrklazy() alone maps nothing; first touch maps exactly one page) |
| Repeated map/unmap or fault behavior | `test_pagefault_counting.py`'s re-touch case (no double-count) |
| Memory exhaustion where practical | `test_memory_exhaustion_recovery.py` |
| Page-accounting cleanup on exit | `test_memory_exhaustion_recovery.py`'s recovery-allocation check |
| (Telemetry correctness, not a spec §1.9 bullet directly, but implied by "measure only reliably instrumentable quantities") | `test_pagefault_failed_counted.py` -- direct verification that `pagefaults_failed` actually increments (added during review followup; every other failure-path test above only infers this from a killed process's exit status) |

## Memory-exhaustion test configuration

`kernel/memlayout.h` fixes `PHYSTOP` at `KERNBASE + 128MB`
**regardless of QEMU's `-m` flag** -- `kernel/kalloc.c`'s `kinit()`
hands `freerange()` exactly that compile-time range at boot. Shrinking
`-m` below 128M for this specific kernel image is not a safety margin,
it is an unsupported configuration (physical addresses the kernel
believes are free RAM would not actually be backed by any memory QEMU
provided). `tests/vm/test_memory_exhaustion_recovery.py` therefore
runs under the harness's ordinary 128M/`-smp 3` QEMU session, not a
deliberately smaller one, and `user/vmexhausttest.c`'s `PAGE_BUDGET`
(40000 pages, ~160MB nominal) is sized to comfortably exceed the
~128MB actually available so genuine exhaustion is reached well before
the budget runs out.

## Captured real run

From this repository's own build, `qemu-system-riscv64` 8.2 (not
fabricated -- real console transcripts):

```
$ vmfaulttest
vmfaulttest: start: pid=3 pagefaults=0 pagefaults_failed=0 sz=16384
vmfaulttest: after_sbrklazy: pid=3 pagefaults=0 pagefaults_failed=0 sz=32768
vmfaulttest: after_touch: pid=3 pagefaults=4 pagefaults_failed=0 sz=32768
vmfaulttest: after_retouch: pid=3 pagefaults=4 pagefaults_failed=0 sz=32768
vmfaulttest: done

$ vmoobtest
usertrap(): page fault (scause=0xf) could not be serviced (invalid address or out of memory) pid=5 va=0x10004000
vmoobtest: child pid=5 wait_pid=5 xstatus=-1
vmoobtest: parent still alive, getpid=4
vmoobtest: done

$ vmforktest
vmforktest: child pid=7 pagefaults=1
vmforktest: parent pid=6 pagefaults=0
vmforktest: done

$ vmexectest
vmexectest: about to exec, never touched the sbrklazy'd region
vmexectest_exec_ok

$ vmexhausttest
vmexhausttest: child progress pages=2000
...
vmexhausttest: child progress pages=32000
usertrap(): page fault (scause=0xf) could not be serviced (invalid address or out of memory) pid=11 va=0x7ed1000
vmexhausttest: child pid=11 xstatus=-1
vmexhausttest: recovery ok, touched 4 pages
vmexhausttest: recovery xstatus=0
vmexhausttest: done
```

Two more, added during review followup, real transcripts from this
same build:

```
$ vmpermtest
usertrap(): page fault (scause=0xf) could not be serviced (permission violation: write access to a page mapped without sufficient permission) pid=4 va=0x0
vmpermtest: child pid=4 wait_pid=4 xstatus=-1
vmpermtest: parent still alive, getpid=3
vmpermtest: done

$ vmpfailcount
vmpfailcount: xvstat(0, bad_ptr) -> -1
vmpfailcount: pagefaults_failed before=0 after=1
vmpfailcount: done
```

`vmpermtest`'s message is the new, distinct permission-violation
diagnostic -- not `vmoobtest`'s "invalid address or out of memory"
line, because a write to the process's own text segment is neither
out-of-bounds nor an allocation failure. (`va=0x0`: this build links
user text starting at virtual address 0, so `main`'s own address,
cast to a pointer, is a low address -- still a real, in-bounds,
already-mapped page, just not a writable one.) `vmpfailcount` shows
the direct `pagefaults_failed` read: `before=0` (a freshly-`fork()`ed
process has touched nothing yet), `after=1` (exactly one deliberately-
failing `xvstat(2)` call happened).

`vmexhausttest` reached exhaustion between pages 32000 and 34000
(~128-136MB nominal), well under its `PAGE_BUDGET=40000`, in about
1.3s wall time, and the kernel remained fully responsive to an
ordinary command immediately afterward (`echo` regression check, both
here and after `vmoobtest`'s out-of-bounds kill).

## Test coverage

See `tests/vm/`. Run via `scripts/run-tests.sh`. Also re-verified
against upstream's own `usertests` regression suite (`ALL TESTS
PASSED`, 223.5s -- re-run against this exact tree during review
followup; timing varies a little run-to-run with host/QEMU load) --
notably, `usertests`' own `lazytests` group deliberately exhausts
memory via the same `sys_sbrk()`/`vmfault()` path this feature
instruments, and its console output during that run now shows this
project's new, more specific diagnostic message (`"page fault
(scause=...) could not be serviced (invalid address or out of
memory)..."`) in place of upstream's generic `"unexpected scause"`
line, with `usertests` still reporting `OK` for every one of those
cases -- direct, real-world confirmation that the `kernel/trap.c`
diagnostic split changed no control-flow decision, only the message.

**A real regression was found and fixed by this exact re-run, during
review followup.** `usertests`' `MAXVAplus` case deliberately faults
on a virtual address `>= MAXVA` without growing `p->sz` -- this hits
`vmfault()`'s `va >= p->sz` branch, which (both before and after this
followup) returns 0 without ever calling `walk()`. The first version
of `vm_permission_violation()` (added by this same followup, see
above) did not carry the same `va >= MAXVA` guard `walkaddr()` already
uses before calling `walk()` -- `walk()` itself panics unconditionally
on `va >= MAXVA` -- so `kernel/trap.c`'s new call to it, given the raw
faulting address directly, panicked the kernel instead of cleanly
killing the process. Fixed by adding the same guard
`vm_permission_violation()`'s sibling `walkaddr()` already has; the
full `usertests` suite (including `MAXVAplus`) now passes cleanly, as
the transcript above shows. This is exactly why this project treats a
full `usertests` re-run as required verification for any `vmfault()`/
`kernel/trap.c` change, not optional.
