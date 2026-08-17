# MiniPaged — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** MiniPaged — Educational LLM Serving Runtime with Paged KV Cache and Continuous Batching
**Portfolio position:** 03 of 09 · Track A (ML systems) · after ForgeLM · later plugs into Helios as an optional backend
**Source of truth:** "03 - MiniPaged - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
MiniPaged is a compact, understandable LLM inference runtime demonstrating the systems ideas behind modern high-throughput serving: request scheduling, prefill/decode separation, paged KV-cache allocation, continuous batching, admission control, and streaming token output. It deliberately does **not** clone vLLM — it is a smaller system whose data structures and scheduling decisions can be explained in a technical interview.

### 1.2 Problem statement
Autoregressive LLM serving has unusual memory and scheduling behavior: prompt processing is compute-heavy, decoding is iterative, sequence lengths differ, and KV cache grows per sequence. Naïve per-request contiguous allocation wastes memory; static batching underutilizes compute. MiniPaged makes those constraints **observable** through a small runtime.

### 1.3 Functional requirements (MVP)
- **FR1** Model adapter around one small Hugging Face/PyTorch causal LM (the runtime, not the model, is the educational component).
- **FR2** Request object: request_id, prompt token IDs, max_new_tokens, sampling config, arrival timestamp, state. Explicit request lifecycle states; tokenizer boundary.
- **FR3** Waiting/running/completed queues (or equivalent state structures).
- **FR4** Prefill step + iterative decode step; continuous batching scheduler (token-budget or sequence-budget); admission control when KV capacity is exhausted.
- **FR5** Paged KV subsystem: fixed-size KV-block abstraction; per-sequence logical block table; physical block pool with free-list, allocate/free/retain/release and capacity metrics; refcount lifecycle; deterministic memory accounting.
- **FR6** Prefix block sharing (and copy-on-write if mutation requires it) — advanced-MVP feature, Phase 4.
- **FR7** Streaming response path via CLI or minimal HTTP server; cancellation semantics if practical.
- **FR8** Benchmark/replay harness for synthetic request traces; deterministic simulation mode with a controllable clock.
- **FR9** Key data structures (public interfaces defined before implementation): Request, SequenceState, PhysicalBlock, BlockPool, BlockTable, SchedulerDecision (selected requests, prefill/decode work, token budget, expected KV growth, preemptions/rejections if implemented).

### 1.4 Core invariants (each becomes an explicit test)
1. No physical block is simultaneously free and referenced.
2. ref_count equals the number of active logical references where sharing is enabled.
3. A sequence's logical block count covers its KV length.
4. Releasing a sequence eventually returns all unshared blocks.
5. Scheduler never admits work requiring more KV capacity than policy permits.
6. Completed/cancelled requests cannot remain scheduled.
7. Deterministic simulated traces reproduce identical allocation history under fixed seed/config.

### 1.5 Non-goals
Full vLLM API compatibility; custom CUDA attention kernels in MVP; multi-node distributed inference; every sampling algorithm; production security/multi-tenancy; arbitrary model architectures; replacing mature runtimes.

### 1.6 Implementation strategy (load-bearing)
Build in layers. The first implementation **simulates KV allocation independent of actual model tensors**. Only after allocator and scheduler invariants are well-tested is a real model execution path connected. If true external KV placement is too invasive in the first version, clearly separate the educational block manager/scheduler from the framework's physical tensor implementation and **document the limitation candidly**.

### 1.7 Deliverable artifacts (website/resume)
Request-lifecycle architecture diagram; KV logical/physical block visualization; scheduler timeline diagram/animation; fragmentation/utilization graph (contiguous vs paged); TTFT/throughput/concurrency benchmark plots; optional interactive simulator (highly valuable). Resume narrative filled from evidence: *"Designed a miniature LLM serving runtime with paged KV-cache allocation, logical-to-physical block tables, continuous batching, prefix sharing and request scheduling; benchmarked memory utilization, fragmentation, TTFT and throughput under variable-length workloads."*

### 1.8 Benchmark workloads & metrics
Reproducible synthetic traces: uniform short prompts/decodes; mixed short/long; burst arrivals; shared-prefix; memory-pressure; decode-heavy vs prefill-heavy mixes.
Metrics: TTFT; inter-token latency / time-per-output-token; requests/sec and tokens/sec; queue delay; active sequences; KV physical capacity/used tokens/reserved capacity/utilization; allocation failures/preemptions; fragmentation/waste for the contiguous baseline.

### 1.9 Test plan
Unit: allocator ops, refcounts, block-table growth, edge capacities, state transitions. Property/invariant: random allocate/free sequences conserve blocks; no leaks; shared references safe. Scheduler: capacity never exceeded; fairness policy documented; deterministic decision tests. Integration: trace replay; cancellation; real-model smoke test; concurrent streaming. Failure: KV exhaustion, malformed request, model failure, cancellation, shutdown.

### 1.10 Open decisions (recommended defaults)
- **D1** Block size (tokens per block; relation to layers/heads) → *default: 16 tokens/block; ADR.*
- **D2** How faithfully physical KV tensors follow block tables in MVP → *default: simulated pages first; document gap per §1.6.*
- **D3** Scheduler policy + token budget → *default: FCFS with per-step token budget; ADR.*
- **D4** Prefill prioritization vs decode latency → *default: decode-first with bounded prefill chunk per step.*
- **D5** Fairness policy → *default: FCFS + starvation note in docs.*
- **D6** Prefix cache key/hash semantics → *default: exact token-prefix hash.*
- **D7** Preemption: omit/reject/swap/recompute → *default: reject at admission for MVP; preemption is stretch.*
- **D8** Target model/hardware → *human decision (e.g., a <1B open model that fits available hardware).*
Document all simplifications explicitly — never hide them.

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Control plane / scheduler | Python first | Educational clarity per brief; optional C++ extension only post-MVP |
| Model execution | PyTorch (+ one small HF causal LM) | Brief requirement |
| Server | Minimal HTTP/SSE (e.g. FastAPI/uvicorn) or CLI streaming | FR7; keep tiny |
| Testing | pytest + hypothesis (property tests for invariants) | §1.9 property tests |
| Packaging | pyproject.toml | Standard |
| Lint/type | Ruff + Pyright/mypy | Portfolio quality bar |
| Benchmarks | Custom trace replay harness; JSON raw results; matplotlib plots | FR8 |
| CI | GitHub Actions: lint/type/unit/property tests; model smoke test optional | Standard |

### Repository structure
```
minipaged/
  pyproject.toml
  src/minipaged/
    requests/  scheduler/
    kv/{block.py, pool.py, table.py, manager.py}
    model/  runtime/  sampling/  server/  metrics/  simulation/
  tests/{unit,property,integration}/
  benchmarks/{traces,raw,plots}/  benchmarks/methodology.md
  docs/architecture.md  docs/scheduler.md  docs/kv-memory.md  docs/invariants.md  docs/decisions/
  examples/
```

---

## Part 3 — Roadmap

Epic order is a **strict dependency chain** (brief mandate): simulator → contiguous baseline → paged allocator → scheduler → sharing → model adapter → API → benchmark study → portfolio hardening.

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Domain simulator** | Request/sequence types, arrival-trace generator, runtime event log, deterministic clock/simulation mode | Synthetic requests move waiting→running→completed without a model |
| **1 — Contiguous KV baseline** | Simple contiguous-memory model; reserved/used/wasted metrics | Variable-length trace demonstrates fragmentation/waste numerically |
| **2 — Paged KV allocator** | Fixed-size blocks, free list, block tables, alloc/free, invariant tests | Same trace runs paged and produces utilization statistics |
| **3 — Scheduler** | Continuous batching (token/sequence budget), prefill vs decode work, admission control | Runtime replays arrivals and produces scheduling timelines |
| **4 — Prefix sharing / COW** | Refcounting, shared prefix blocks, COW where mutation requires | Shared-prefix workload reduces physical KV usage; lifecycle tests pass |
| **5 — Real model adapter** | One small causal LM integrated; scheduling/memory abstractions stay framework-independent | Requests generate real tokens through the scheduler |
| **6 — Streaming API** | Minimal HTTP/SSE or CLI streaming; cancellation if practical | Concurrent clients observe token streams |
| **7 — Benchmark study** | Static/sequential vs continuous batching; contiguous sim vs paged allocator; full metric set | Committed raw results + plots per §1.8 |
| **8 — Portfolio hardening** | Diagrams, benchmark report, visual block allocator, README, website assets | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
Chunked prefill; preemption/recompute; CUDA Graph capture; quantized KV cache; adapter/LoRA routing; C++ scheduler core; multi-GPU simulation; OpenAI-compatible API subset; trace-viewer web UI.

### Definition of Done
Allocator and scheduler independently testable; deterministic traces show paged vs contiguous behavior; continuous batching demonstrated; ≥1 real model generates through the runtime; benchmark report covers TTFT/ITL/throughput/queue time/KV utilization; invariants committed as tests; limitations candidly documented; website-ready visualization exists.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Do not integrate a real model before allocator/scheduler tests exist (Phases 0–4 gate Phase 5).
2. Do not call a mature serving runtime as the implementation.
3. Keep educational abstractions explicit.
4. Never fabricate performance numbers.
5. Prefer a smaller correct runtime over broad incomplete compatibility.
6. Make memory/accounting behavior inspectable in logs/tests.

### Per-task requirements (brief mandate)
For every task specify: files/modules expected to change; public interfaces/data structures; invariants and tests; the acceptance command; a benchmark checkpoint if applicable; and any architectural decision requiring an ADR.

### Kickoff prompt
> Read `03-minipaged-spec.md` in full. Produce an engineering plan with epics in exactly this dependency order: simulator → contiguous baseline → paged allocator → scheduler → sharing → model adapter → API → benchmark study → portfolio hardening. For each task list the files/modules to change, the public interfaces/data structures (define Request, SequenceState, PhysicalBlock, BlockPool, BlockTable, SchedulerDecision before implementation), the invariants and tests attached, the acceptance command, benchmark checkpoints, and ADRs needed for decisions D1–D7. D8 (target model/hardware) needs my sign-off before Phase 5. Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (Phases 1, 2, 7)  →  /ship
```
Skip `/qa` unless the Phase 6 HTTP server warrants a quick runnable check; skip `/cso` (no sensitive surface in MVP).
