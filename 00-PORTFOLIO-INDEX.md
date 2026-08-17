# Anish 0-to-100 Engineering Portfolio — Master Index & Build Order

Source: Google Drive → "Anish 0-to-100 Engineering Curricula" → "Fable Project Planning Briefs" (9 briefs, read in full on 2026-08-17).

Each project has a single spec file in this folder containing three parts — **PRD**, **Tech Stack Plan**, and **Roadmap** — plus a **Claude Code Handoff** section with execution rules and a suggested kickoff prompt. Hand any one file to Claude Code and it has everything it needs to plan and build that repo.

## The nine projects

| # | Project | One-liner | Primary language | Spec file |
|---|---------|-----------|------------------|-----------|
| 01 | **ForgeLM** | From-scratch decoder-only Transformer training stack | Python / PyTorch | `01-forgelm-spec.md` |
| 02 | **KernelForge** | CUDA kernel optimization laboratory with profiler-backed optimization ladders | C++20 / CUDA | `02-kernelforge-spec.md` |
| 03 | **MiniPaged** | Educational LLM serving runtime: paged KV cache + continuous batching | Python (C++ later) | `03-minipaged-spec.md` |
| 04 | **FlashLite** | Naïve → tiled → online-softmax → fused IO-aware attention kernel | CUDA C++ / Python | `04-flashlite-spec.md` |
| 05 | **ArcServe** | High-performance event-driven C++ network server (epoll, backpressure) | C++20/23 | `05-arcserve-spec.md` |
| 06 | **PebbleDB** | Miniature C++ LSM storage engine: WAL, SSTables, compaction, crash recovery | C++20/23 | `06-pebbledb-spec.md` |
| 07 | **xv6-plus** | Extended xv6 RISC-V teaching kernel: tracing, accounting, xvtop, scheduler + VM extension | C (kernel) | `07-xv6plus-spec.md` |
| 08 | **ReleaseGuard** | SLO-aware progressive-delivery and automated rollback controller | Go (recommended) | `08-releaseguard-spec.md` |
| 09 | **Helios** | Capstone: production-oriented LLM inference platform integrating the others | Python + K8s | `09-helios-spec.md` |

## Recommended build order and dependencies

The briefs are explicit on one hard constraint: **Helios is the capstone and must not be started first.** It consumes stable interfaces from MiniPaged, FlashLite, KernelForge, and ReleaseGuard. Everything else is independently buildable, but there are soft learning dependencies worth honoring:

```
Track A (ML systems):    ForgeLM ──► MiniPaged ──► FlashLite ─┐
Track B (GPU perf):      KernelForge ─────────► FlashLite ────┤
Track C (C++/Linux):     ArcServe ──► PebbleDB                ├──► Helios
Track D (OS):            xv6-plus (independent)               │
Track E (SRE/platform):  ReleaseGuard ────────────────────────┘
```

- **ForgeLM before MiniPaged** — MiniPaged assumes fluency with the prefill/decode mechanics ForgeLM teaches.
- **KernelForge before FlashLite** — FlashLite's fused attention kernel builds directly on the CUDA memory-hierarchy skills KernelForge develops. FlashLite explicitly does *not* live in the KernelForge repo.
- **ArcServe before PebbleDB** is a soft ordering (both are C++/Linux; ArcServe's tooling discipline transfers). They can be swapped.
- **ReleaseGuard anytime** — it's self-contained and later plugs into Helios Phase 8.
- **Helios last**, and even then its early phases (mock backend, one mature runtime) don't require the other repos; integrations are Phases 9–10.

## Cross-cutting rules (apply to every repo)

These recur in every brief and should be treated as portfolio-wide law:

1. **Never fabricate benchmark or performance numbers.** Every resume/README claim must trace to committed raw benchmark artifacts with recorded hardware, versions, seeds, warmup and repetition counts.
2. **Correctness before optimization.** Reference implementations and tests land before any optimized variant.
3. **Tests ship in the same change as functionality.** No untested merges.
4. **ADRs before irreversible architectural choices.** Each repo has a `docs/decisions/` folder.
5. **Keep code interview-explainable.** No copying whole implementations from mature libraries; educational components stay hand-built.
6. **Local reproducibility over cloud dependency.** A fresh clone must build, test, and demo without paid services.
7. **Small, narrowly scoped, reviewable commits.**

## How to hand a project to Claude Code

1. Drop the spec file (e.g. `01-forgelm-spec.md`) into the new repo root (or paste it into the session).
2. Use the kickoff prompt in that spec's "Claude Code Handoff" section.
3. If you're using gstack, each spec includes a suggested command sequence (`/office-hours` → `/autoplan` → implement per phase → `/review` → `/ship`).
4. Enforce phase exit criteria as hard gates — Claude Code should not proceed past a failed acceptance criterion.
