# xv6-plus — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** xv6-plus — Extended Teaching Kernel and Systems Observatory
**Portfolio position:** 07 of 09 · Track D (OS) · independent of the other projects
**Source of truth:** "07 - xv6-plus - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
xv6-plus is an operating-systems project built on MIT's xv6 RISC-V teaching kernel. The goal is not merely completing standard labs; it is a **coherent extension layer** demonstrating process accounting, virtual memory, syscall tracing, scheduling, observability, and kernel debugging in a small codebase the author can fully explain.

### 1.2 Project philosophy (load-bearing honesty requirement)
The repo must clearly distinguish: **(A)** upstream xv6 code, **(B)** completed educational lab modifications, **(C)** original xv6-plus extensions. Standard course work is never marketed as original engineering; the README lists original features explicitly and `docs/upstream-delta.md` documents the delta.

### 1.3 Functional requirements — MVP original features
- **FR1** Syscall tracing framework with per-process enable/filter controls, capturing syscall number/name/return data within safe scope, plus a userspace control tool (`xtrace`).
- **FR2** Per-process accounting: runtime ticks, sleep/wait time where feasible, memory size/pages, syscall counts; lifecycle-correct init/reset on allocation and correct handling on fork/exec/exit.
- **FR3** Kernel/user statistics interface via one or more syscalls (procfs-like virtual interface is stretch — simpler syscall is safer for MVP).
- **FR4** `xvtop` userspace monitor displaying process state, CPU/runtime, memory, and syscall/resource counters; refresh, sorting/filtering if manageable.
- **FR5** One scheduler extension/experiment (e.g., priority or lottery scheduling) with policy selection, deterministic/synthetic workloads, and documented fairness/starvation behavior.
- **FR6** One substantive VM extension chosen after prerequisite labs: mmap subset, lazy allocation, COW-related instrumentation, accessed-page stats, or page-fault telemetry.
- **FR7** Stress/regression test suite (scheduler, VM, syscall, stress directories).

### 1.4 Baseline prerequisites (separate track, not portfolio milestones)
Local build/knowledge path for: xv6 build/QEMU/debug workflow; process table and scheduler; syscalls/traps; RISC-V page tables; locking basics; relevant filesystem/device concepts. The plan identifies which standard course labs are prerequisites but keeps them separate from original milestones.

### 1.5 Core invariants (each guarded by tests/review)
1. Kernel accounting never corrupts process lifecycle state.
2. Locks are never acquired in inconsistent order.
3. Process telemetry is initialized at allocation and handled correctly on fork/exec/exit.
4. The scheduler always eventually selects eligible work per documented policy assumptions.
5. Zombie/free process slots never remain visible as active telemetry.
6. VM changes preserve user/kernel isolation and mapping permissions.
7. Syscall interfaces validate user pointers/arguments.
8. Observability code must not destabilize the kernel it observes.

### 1.6 Non-goals
Turning xv6 into Linux; full POSIX compatibility; production-grade security; unrelated features for commit count; large device-driver work unless deliberately chosen as stretch; copying complete lab solutions without understanding.

### 1.7 Deliverable artifacts (website/resume)
user→syscall→trap→kernel→return diagram; process-state/scheduler diagram; page-table/VM diagram for one implemented feature; xvtop screenshot/GIF; before/after scheduling or memory experiment graph. Resume narrative filled from evidence: *"Extended the xv6 RISC-V teaching kernel with process/resource telemetry, syscall tracing, virtual-memory features and a custom scheduler/diagnostic interface; validated kernel changes with stress tests and built an xvtop-style userspace monitor over new kernel instrumentation."*

### 1.8 Debugging / quality bar
gdb/QEMU workflow for kernel bugs (not printf-only); assertions/panics only for actual invariant violations; no sleeping while holding inappropriate spinlocks; lock order reviewed; syscalls kept small with logic in testable kernel helpers; RISC-V/xv6 assumptions documented.

### 1.9 Benchmark plans
Scheduler: workload programs for CPU-bound, short interactive/yield/sleep, mixed CPU/IO-like, and unequal priorities/tickets; measure only reliably instrumentable quantities — runtime ticks, wait time, completion time, selection counts, fairness distribution.
VM (per chosen extension): invalid address/permission cases; fork/exec interactions; mapping/allocation boundaries; repeated map/unmap or fault behavior; memory exhaustion where practical; page-accounting cleanup on exit.

### 1.10 Open decisions (recommended defaults)
- **D1** Upstream/course revision to pin → *human decision; record in Phase 0.*
- **D2** Which standard labs are already incorporated → *human decision; documented in upstream-delta.md.*
- **D3** procfs-like VFS vs stats syscall → *default per brief: stats syscall for MVP; procfs stretch.*
- **D4** Scheduler policy choice + starvation implications → *default: lottery scheduling (clean fairness story); ADR.*
- **D5** VM extension choice → *decided after prerequisite labs; preferred options: mmap subset or lazy allocation/page-fault telemetry.*
- **D6** Telemetry synchronization overhead approach → *default: per-process counters updated under existing proc lock; ADR if new locks needed.*
- **D7** Ticks vs finer-grained counters → *default: ticks.*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Kernel base | MIT xv6 (RISC-V), pinned revision (D1) | Brief requirement |
| Language | C (kernel + userspace tools) | xv6 convention |
| Toolchain | riscv64 GNU toolchain + QEMU | xv6 standard workflow |
| Debugging | gdb + QEMU | Quality bar §1.8 |
| Test running | `scripts/run-tests.sh` orchestrating QEMU test runs | Reproducibility |
| Benchmark analysis | Python (`scripts/benchmark.py`) + plots | Brief structure |
| VCS discipline | Original-project branch structure over pinned upstream | Honesty requirement §1.2 |
| CI | Build + boot smoke test in QEMU where practical | Reproducibility |

### Repository structure
```
xv6-plus/
  kernel/                 # xv6 + modifications
  user/                   # xvtop.c, xtrace.c, benchmark utilities
  tests/{scheduler,vm,syscall,stress}/
  scripts/run-tests.sh  scripts/benchmark.py
  docs/upstream-delta.md  docs/kernel-architecture.md  docs/tracing.md
  docs/scheduler.md  docs/vm-extension.md  docs/invariants.md  docs/decisions/
  benchmarks/{raw,plots}/  benchmarks/methodology.md
```

---

## Part 3 — Roadmap

Two clearly separated tracks: **Track P** (prerequisite lab readiness — course labs, not portfolio milestones) and **Track O** (original xv6-plus implementation, below).

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Reproducible base** | Pin upstream/course revision, document toolchain/QEMU, branch structure, baseline smoke tests | Clean xv6 boots; test script runs |
| **1 — Syscall tracing foundation** | Tracing syscall/flags, syscall number/name/return capture, userspace control tool | Per-process tracing works without breaking normal execution |
| **2 — Process accounting** | Runtime/memory/syscall counters; lifecycle init/cleanup; stable stats structure returned to userspace | Tests verify fork/exec/exit behavior and counter monotonicity where appropriate |
| **3 — xvtop** | Userspace monitor over the stats interface; refresh, sort/filter if manageable | Tool visibly reports active processes and resource data |
| **4 — Scheduler experiment** | One policy beyond baseline; policy selection; deterministic/synthetic workloads | Benchmark compares fairness/turnaround/response vs baseline scheduler |
| **5 — VM extension** | One bounded VM feature (per D5) | Dedicated VM tests + architecture note |
| **6 — Stress/race hardening** | Concurrent fork/exit, syscall-heavy, memory-pressure, scheduler stress; lock/invariant audit | Repeated stress suite passes |
| **7 — Kernel observability report** | Use tracing/xvtop to explain an end-to-end workload (creation, syscalls, scheduling, memory) | Report committed |
| **8 — Portfolio hardening** | Upstream-delta doc, diagrams, demo recording, benchmark plots, README, release tag | Fresh-clone build/boot verified |

### Stretch goals (post-MVP only)
procfs-like virtual filesystem; COW instrumentation/dashboard; networking statistics; simple resource limits; per-process syscall latency samples; multilevel feedback queue; huge-page experiment if the tree supports it cleanly; BPF-like tracing concept only after the core project.

### Definition of Done
Original features clearly distinguished from course work; xvtop functional; tracing/accounting survives process-lifecycle tests; scheduler experiment has a measurable comparison; VM extension has dedicated correctness tests; stress suite repeatable; docs explain traps/scheduling/VM modification paths; demo/website artifacts exist.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Read the relevant xv6 subsystem before changing it.
2. Do not copy unseen lab solutions as implementation shortcuts.
3. Keep original contributions auditable (branch/commit discipline per §1.2).
4. Do not merge a kernel change without a failure/stress test.
5. Prefer one coherent OS story over many unrelated features.

### Per-kernel-task requirements (brief mandate)
Every kernel task specifies: the kernel invariant(s) at stake; lock/lifetime considerations; any user/kernel ABI change; a stress/regression test; and a rollback/debug strategy.

### Kickoff prompt
> Read `07-xv6plus-spec.md` in full. Produce a two-track engineering plan: Track P lists prerequisite xv6 lab readiness (kept separate from portfolio milestones), and Track O implements original features in strict order: tracing → accounting API → xvtop → scheduler → VM feature → stress hardening → documentation. For every kernel task include the invariant(s), lock/lifetime considerations, ABI changes, a stress/regression test, and a rollback/debug strategy. Decisions D1, D2 and D5 need my input; use the §1.10 defaults for D3, D4, D6, D7 with ADRs. Then implement Phase 0 only (pin revision, toolchain docs, branch structure, boot smoke test) and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review (focus: lock order, lifecycle invariants)  →  /benchmark (Phase 4)  →  /ship
```
Skip `/qa` and `/cso`. Use `/freeze kernel/` selectively when working on userspace tools to protect kernel state.
