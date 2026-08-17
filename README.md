# Passion Projects — 0-to-100 Engineering Portfolio Specs

Nine project specs (PRD · Tech Stack Plan · Roadmap · Claude Code handoff) for a systems/ML engineering portfolio. Start with [`00-PORTFOLIO-INDEX.md`](00-PORTFOLIO-INDEX.md) for the project list, build order, dependency graph, and portfolio-wide rules.

| # | Project | One-liner |
|---|---------|-----------|
| 01 | ForgeLM | From-scratch decoder-only Transformer training stack (PyTorch) |
| 02 | KernelForge | CUDA kernel optimization laboratory |
| 03 | MiniPaged | Educational LLM serving runtime: paged KV cache + continuous batching |
| 04 | FlashLite | Naïve → tiled → online-softmax → fused IO-aware attention kernel |
| 05 | ArcServe | High-performance event-driven C++ network server (epoll) |
| 06 | PebbleDB | Miniature C++ LSM storage engine |
| 07 | xv6-plus | Extended xv6 RISC-V teaching kernel |
| 08 | ReleaseGuard | SLO-aware progressive-delivery / rollback controller (Go, K8s) |
| 09 | Helios | Capstone: production-oriented LLM inference platform |

Each project gets its own repo when built; this repo holds the specs. To kick one off, drop its spec file into the new repo and use the kickoff prompt in that spec's "Claude Code Handoff" section.

**Platform note:** KernelForge (02) and FlashLite (04) require an NVIDIA GPU with CUDA; ArcServe (05) is Linux-only (epoll). Build these on a Linux/NVIDIA machine.
