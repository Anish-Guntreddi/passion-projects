# ArcServe — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** ArcServe — High-Performance C++ Network Server
**Portfolio position:** 05 of 09 · Track C (C++/Linux systems) · soft prerequisite for PebbleDB
**Source of truth:** "05 - ArcServe - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
ArcServe is a C++/Linux systems project demonstrating event-driven networking, explicit resource management, concurrency, backpressure, and evidence-based performance engineering. It begins as a small nonblocking TCP server and evolves into a benchmarked HTTP/1.1-subset (preferred, if scope stays controlled) or custom framed-protocol server.

### 1.2 Functional requirements (MVP)
- **FR1** Nonblocking listening/client sockets; epoll-based event loop.
- **FR2** Connection object/state machine: read/write buffers, parser state, timeout timestamps, peer metadata, lifecycle state.
- **FR3** Incremental parser for a small protocol handling fragmented input; if HTTP, only a documented method/header/body subset with keep-alive; unsupported/oversized messages rejected predictably.
- **FR4** Bounded request/work queues; configurable worker pool for CPU-bound handler work (fixed worker count, bounded queue, safe stop semantics, metrics); simple synchronous handler first.
- **FR5** Backpressure policy: high-water marks and policies for slow readers, full worker queue, and overload.
- **FR6** Output path: EPOLLOUT management, write queues, correct partial-write and slow-client behavior.
- **FR7** Timeouts (connection idle), graceful shutdown (coordinator + reactor wakeup mechanism).
- **FR8** Observability: structured logging; metrics counters — connections accepted/active/closed, requests, bytes, parse errors, queue depth, rejected work, latency histograms where feasible.
- **FR9** Unit/integration/load tests; benchmark harness + methodology.

### 1.3 Core design principles (enforced in review)
One thread never blocks the reactor on ordinary socket I/O; connection state transitions are explicit and testable; partial reads/writes are normal, not errors; queues are bounded; backpressure is observable; resource lifetimes use RAII; every file descriptor has an obvious owner; the server fails gracefully under overload instead of growing memory without bound.

### 1.4 Non-goals
Full RFC-complete HTTP; TLS in first MVP; HTTP/2/3; replacing nginx/Envoy; cross-platform networking; large application framework; distributed service mesh.

### 1.5 Deliverable artifacts (website/resume)
Request-lifecycle/event-loop diagram; throughput-vs-concurrency graph; P50/P95/P99 latency graph; flame graph or profiler summary; event-loop-vs-worker-pool architecture visualization. Resume narrative filled from evidence: *"Built a high-performance event-driven C++ network server using nonblocking Linux sockets, epoll, bounded worker queues and explicit backpressure; benchmarked throughput, tail latency, CPU utilization, context switches and allocation behavior under concurrent load."*

### 1.6 Test strategy
Unit: fd RAII, buffers, parser, state transitions, bounded queue.
Integration: fragmented requests; pipelined/keep-alive behavior if supported; large-request limits; disconnect mid-request; slow client; many simultaneous clients.
Concurrency: worker shutdown, queue saturation, TSan-compatible tests.
Failure: EMFILE/resource-exhaustion simulation where feasible, malformed protocol, client resets, timeouts, shutdown while active.
Parser fuzz/property cases where practical.

### 1.7 Performance experiments (required sweeps)
Concurrency 1 / 10 / 100 / 1k+ as hardware permits; small vs moderate payloads; worker count; queue capacity; keep-alive on/off if HTTP.
Measure: requests/sec; P50/P95/P99 latency; CPU%; RSS; context switches; syscalls where useful; queue depth/rejections; allocations/request if instrumented.

### 1.8 Acceptance criteria (MVP complete when)
Partial reads/writes handled correctly; no fd/memory leaks under sanitizer/stress runs; bounded memory/queue behavior under overload; graceful shutdown leaves no worker/connection lifecycle bugs; benchmark harness reproducible; ≥1 optimization is profiler-backed; architecture explainable without Boost.Asio hiding the event model.

### 1.9 Open decisions (recommended defaults)
- **D1** HTTP subset vs custom framed protocol → *default: HTTP/1.1 subset (GET/POST, small header set, Content-Length bodies), ADR the exact scope in `docs/protocol-scope.md`.*
- **D2** Edge- vs level-triggered epoll → *default per brief: simplest correct first — level-triggered; ET as measured experiment later.*
- **D3** Buffer representation → *default: contiguous grow-capped byte buffers; ADR before Phase 4.*
- **D4** Timer structure → *default: min-heap of deadlines.*
- **D5** Worker affinity/count → *default: hardware_concurrency-derived, configurable; affinity is stretch.*
- **D6** Handwritten vs library parser → *default per brief: handwritten educational parser within narrow scope.*
- **D7** io_uring → *only after epoll baseline is correct and measured (stretch).*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Language / platform | C++20/23 on Linux | Brief requirement |
| Build | CMake + Ninja, warnings-as-errors | Brief requirement |
| Tests | GoogleTest or Catch2 | Brief requirement |
| Static analysis | clang-tidy | Brief requirement |
| Sanitizers | ASan, UBSan, TSan where applicable | Brief requirement |
| Profiling | perf, flame graphs, strace, optional eBPF tooling | Brief requirement |
| Load generation | wrk/wrk2 (HTTP) or custom client (framed) | Choose per D1 |
| CI | Linux: build + tests + static analysis | Brief requirement |

### Repository structure
```
arcserve/
  CMakeLists.txt
  src/{net,reactor,protocol,server,concurrency,buffers,observability,app}/
  include/arcserve/
  tests/{unit,integration,stress}/
  benchmarks/{scenarios,raw,plots}/  benchmarks/methodology.md
  docs/architecture.md  docs/protocol-scope.md  docs/concurrency.md  docs/overload.md  docs/decisions/
  scripts/
```
High-level flow: acceptor → epoll reactor → connection registry → incremental parser → request object → optional bounded work queue → worker pool/handler → response buffer → EPOLLOUT → client. Cross-cutting: timer/timeout management, logging/metrics, shutdown coordinator, memory/buffer strategy.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Build/tooling foundation** | CMake, warnings-as-errors, formatting, sanitizers, tests, CI, RAII fd wrapper | Clean build/test on Linux |
| **1 — Blocking correctness server** | Minimal blocking echo/framed server validating protocol/parser semantics | Integration client tests pass |
| **2 — Nonblocking epoll reactor** | Reactor + connection registry; echo under many connections | No blocking I/O in reactor; stress test passes |
| **3 — Incremental protocol parser** | HTTP subset or framed protocol; partial-read tests, malformed input, limits | Fuzz/property cases where practical; end-to-end requests succeed |
| **4 — Output buffering & partial writes** | EPOLLOUT management, write queues, slow-client behavior | Large responses survive constrained clients without corruption |
| **5 — Worker pool** | Bounded queue + configurable workers; direct-reactor vs worker-dispatch comparison for a CPU-heavy handler | Shutdown and saturation tests pass |
| **6 — Timeouts/backpressure/shutdown** | Idle timeout, overload policy, high-water mark, clean termination | Overload test shows bounded memory/queue behavior |
| **7 — Observability & profiling** | Counters, structured logs, perf/flame-graph investigation, allocation/context-switch metrics | ≥1 profiler-backed optimization landed |
| **8 — Benchmark study** | Sweeps per §1.7 with full metric capture | Committed raw results + plots |
| **9 — Portfolio hardening** | README, diagrams, benchmark report, website assets | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
io_uring backend; TLS via a mature crypto library; zero-copy sendfile; custom slab/arena for connection buffers; lock-free queue experiment; HTTP parser performance comparison; CPU pinning/NUMA experiment; Prometheus exporter.

### Definition of Done
Fresh clone builds on Linux; automated tests pass; server survives stress + sanitizer runs; overload behavior is bounded; benchmark plots exist; README shows a measured performance story with documented hardware/methodology.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Every task includes concurrency/lifetime invariants and acceptance commands.
2. Do not jump to io_uring or custom allocators until the epoll baseline is correct and measured.
3. Correctness before performance; sanitizers stay green.
4. Bounded queues everywhere — an unbounded queue is a review-blocking defect.
5. Tests ship with functionality; TSan-relevant code gets TSan-compatible tests.
6. Never claim performance results without committed raw benchmark data.

### Kickoff prompt
> Read `05-arcserve-spec.md` in full. Produce an engineering plan with epics in strict dependency order: tooling → blocking reference → epoll reactor → parser → output path → workers → backpressure/timeouts → profiling → benchmark → docs. Every task must state its concurrency/lifetime invariants and an acceptance command. ADR decisions D1–D6 before their phases (D1 and D6 before Phase 1; D2 before Phase 2; D3 before Phase 4). io_uring and custom allocators are out of scope until the epoll baseline is measured. Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (Phases 7–8)  →  /ship
```
Skip `/qa` (no browser surface). Consider `/cso` once, at Phase 3, focused on parser robustness against malformed input.
