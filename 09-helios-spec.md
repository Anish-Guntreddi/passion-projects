# Helios — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** Helios — Production-Oriented LLM Inference Platform (Portfolio Flagship)
**Portfolio position:** 09 of 09 · **Capstone — do not start first.** Grows stronger as ForgeLM, KernelForge, MiniPaged, FlashLite and ReleaseGuard mature. Designed around replaceable interfaces, never a copy-paste monolith.
**Source of truth:** "09 - Helios - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution (early phases have no dependency on the other repos)

---

## Part 1 — Product Requirements Document

### 1.1 Overview
Helios demonstrates end-to-end inference-platform engineering: API design, request routing, scheduling, model serving, observability, autoscaling, benchmarking, progressive delivery, and optional custom runtime/kernel components from the other portfolio repos. It exposes a small LLM inference platform supporting one or more open causal models through a consistent API while making runtime/performance characteristics visible.

### 1.2 Users
Application developer sending chat/completion requests; performance engineer running benchmark workloads; operator observing health/capacity/release behavior.

### 1.3 Functional requirements (MVP)
- **FR1** HTTP API with streaming generation; request validation; configurable generation options; request IDs + trace propagation.
- **FR2** Model/runtime abstraction (`RuntimeBackend` interface); ≥1 mature backend first (vLLM / HF-PyTorch / TensorRT-LLM depending on hardware — choose one).
- **FR3** Routing to model instance/backend via a ModelRegistry (model ID → backend/config/hardware requirements/status).
- **FR4** Admission/concurrency control: accept/reject/queue on configured concurrency/queue/capacity rules; bounded queues; cancellation/timeouts; overload response policy.
- **FR5** Observability: structured metrics for TTFT, token latency, total latency, queue delay, prompt/output tokens, errors; OpenTelemetry traces; Prometheus metrics; health/readiness endpoints; explicit timestamps for arrival, scheduling/start, first token.
- **FR6** Benchmark/load-test client with workload traces varying prompt length, output length, concurrency, arrival pattern.
- **FR7** Docker packaging; Kubernetes deployment (kind/k3d locally); configuration/secrets separation; CI/CD.
- **FR8** Capacity/performance report (see §1.9).
- **Post-MVP integration milestones:** MiniPaged educational backend; FlashLite custom attention experiment; KernelForge primitive experiment; ReleaseGuard canary/SLO deployment controller.

### 1.4 Core interfaces (defined in Phase 0, before implementation)
- **GenerationRequest:** request ID, model ID, prompt/messages normalized to backend input, max tokens, temperature/top-p/top-k as supported, streaming flag, timeout/cancellation context.
- **GenerationChunk:** request ID, token/text delta, token index, timestamp/latency metadata, finish reason, usage summary.
- **RuntimeBackend:** load / start / health / generate_stream / cancel / stats.
- **AdmissionController:** accept/reject/queue per configured rules.
- **Metrics contract:** request_total, request_errors, active_requests, queue_depth, queue_delay, TTFT, request_duration, output_token_count, inter-token timing summary; backend/model labels with **cardinality controlled**.

### 1.5 Architecture principles (enforced in review)
**A. Backend isolation** — core API never depends on one runtime's request/response types. **B. Measurement-first** — every performance experiment produces raw artifacts tied to config + hardware. **C. Streaming is first-class** — streaming and cancellation are architecture concerns, not afterthoughts. **D. Backpressure/admission control** — unbounded queues are unacceptable. **E. SLO observability** — metrics distinguish queue, prefill/TTFT, decode/token latency, and overall latency where backend data permits. **F. Progressive complexity** — single-node/single-GPU before multi-GPU/distributed. **G. Reuse, don't merge repos blindly** — integrate other portfolio projects via packages/submodules/releases/interfaces only when stable; keep them independently understandable and correctly attributed.

### 1.6 Non-goals (first MVP)
Building every inference component from scratch; all model families; multi-region production; billing/commercial SaaS control plane; arbitrary user-uploaded models; complex GPU cluster scheduler; training/fine-tuning; replacing Kubernetes or established engines.

### 1.7 Security / privacy baseline
No model/API secrets in repo; request content **not** logged by default (metadata only, unless explicit safe dev mode); dependency/container scans in CI; least-privilege Kubernetes permissions; external auth optional for MVP but architecture leaves the boundary for it; configurable trace/log retention.

### 1.8 SLO / operational model
Define a demo SLO **only after** baseline data exists. Candidate indicators: availability/success rate; P95/P99 TTFT for a workload class; P95 request latency or TPOT; queue-delay bound. No arbitrary "99.99%" claims — SLOs are teaching/demo targets grounded in measured local environment.

### 1.9 Capacity engineering output (final report must answer)
Which workload saturates first and why; concurrency at which throughput stops improving; how TTFT degrades as the queue grows; how prompt/output lengths shift the bottleneck; which backend/configs lie on the latency–throughput Pareto frontier; observed GPU memory/capacity limits; what optimization was attempted and whether evidence supported it.

### 1.10 Benchmark scenarios
A. Interactive chat (short/medium prompt, modest output, latency-sensitive). B. Prefill-heavy (large prompt, short output). C. Decode-heavy (small prompt, long generation). D. Concurrency sweep 1→2→4→8→… to saturation. E. Bursty arrivals. F. Mixed lengths (scheduler stress). G. Failure/overload (backend unavailable, queue full, cancellation). Distributions required, never only averages.

### 1.11 Test strategy
Unit: validation, routing, admission, metrics labeling, configuration. **Contract: all RuntimeBackend implementations pass shared streaming/cancellation/error contract tests.** Integration: API→backend→stream; telemetry emitted; health/readiness. Load: concurrency, queue bound, graceful overload. Failure: backend crash/unavailable, slow backend, cancellation, termination/draining, telemetry outage. Deployment: container smoke test; Kubernetes readiness/rollout test.

### 1.12 Open decisions (recommended defaults)
- **D1** API language → *default per brief: Python (FastAPI) control plane with explicit typed boundaries; Go/Rust alternatives if platform signal preferred.*
- **D2** First model/runtime → *human decision based on available hardware; default: vLLM with a small open model, else HF-PyTorch backend.*
- **D3** Runtime in-process vs managed subprocess/service → *ADR before Phase 1.*
- **D4** Backend-provided metrics vs Helios-measured timestamps → *default: Helios-measured timestamps as source of truth; backend stats supplementary.*
- **D5** Queue/admission policy → *default: bounded FIFO + max-concurrency; ADR.*
- **D6** Model-loading configuration approach → *ADR.*
- **D7** Local K8s GPU complexity → *per brief: allow native Docker/local-GPU path before K8s GPU integration.*
- **D8** Auth omitted for local MVP → *default: yes, with an explicit architecture boundary for adding it.*
- **D9** Integration mechanism for external portfolio repos → *default: versioned package/release + adapter, feature-gated; never vendored copies.*

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Control/API plane | Python + FastAPI (typed boundaries) per D1 | Easiest ML runtime integration per brief default |
| Runtime | vLLM or basic PyTorch/HF backend per D2; TensorRT-LLM only if compatible hardware | Brief requirement |
| Packaging | Docker | Brief requirement |
| Orchestration | Kubernetes — kind/k3d locally; optional cloud later | Brief requirement; local reproducibility never depends on cloud |
| Observability | OpenTelemetry + Prometheus + Grafana | Brief requirement |
| Load testing | Custom async benchmark harness + a standard HTTP load tool where applicable | Brief requirement |
| CI | GitHub Actions | Brief requirement |
| IaC | Terraform for optional cloud env only | Brief requirement |
| Testing | pytest; shared backend contract-test suite; load/failure/deployment suites | §1.11 |

### Repository structure
```
helios/
  pyproject.toml
  services/{api,router,benchmark}/
  src/helios/
    api/
    runtime/{interfaces,vllm,hf,minipaged}/   # minipaged = later adapter only
    routing/  admission/  streaming/  observability/  config/  model_registry/
  tests/{unit,contract,integration,load,failure}/
  deploy/{docker,kubernetes,helm-or-kustomize}/
  observability/{otel,prometheus,grafana}/
  benchmarks/{scenarios,configs,raw,plots}/  benchmarks/methodology.md
  docs/architecture.md  docs/runtime-interface.md  docs/metrics.md  docs/capacity.md  docs/operations.md  docs/decisions/
  .github/workflows/
```
Request flow: Client → API gateway/Helios API → auth placeholder/local dev identity → validation → router/model registry → admission controller → runtime backend adapter (mature backend A | optional MiniPaged | optional alternative) → GPU/model executor → streaming token path → client. Cross-cutting: OTel tracing, Prometheus metrics, structured logs, benchmark/load generator, config/model registry, K8s deployment, CI/CD/release.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Architecture & repo contract** | Choose D1/D2; define GenerationRequest/Chunk + RuntimeBackend; config schema; benchmark result schema; CI; dev environment; **mock backend** | Mock backend streams deterministic tokens through API tests |
| **1 — Real single-model backend** | One small open model/runtime; startup/health; streaming generation | Documented local/GPU setup serves concurrent requests correctly |
| **2 — Observability foundation** | Request IDs, structured logs, OTel spans, Prometheus metrics, explicit arrival/start/first-token timestamps | One request traceable end-to-end; metrics exported |
| **3 — Admission/concurrency control** | Bounded queue/concurrency, cancellation/timeouts, overload policy | Load beyond capacity stays bounded and observable |
| **4 — Benchmark harness** | Workload traces (prompt/output length, concurrency, arrival pattern); TTFT/TPOT/latency/queue/tokens-sec/GPU metrics | Reproducible baseline performance report |
| **5 — Runtime abstraction proof** | Second backend or mock alternative (HF vs vLLM, or vLLM vs MiniPaged later) | Same API/harness runs two backends via config switch |
| **6 — Container/K8s deployment** | Image, health/readiness, resource requests, config/secrets, local cluster deploy, service exposure | Reproducible deploy script from fresh clone |
| **7 — Reliability & operations** | Graceful shutdown/draining, backend-failure behavior, readiness changes, timeout/cancel, load shedding, restart tests, basic SLO definitions, runbook | Scripted failure scenarios + operational notes |
| **8 — CI/CD + ReleaseGuard integration** | Automated test/build/image pipeline; optional ReleaseGuard canary promote/rollback on Helios latency/error signals | Healthy release + intentionally degraded release demos |
| **9 — MiniPaged integration** | Adapter (never copied scheduler code), only after MiniPaged exposes a stable interface; honest limitations | Common benchmark harness runs both backends |
| **10 — Custom kernel experiment** | One stable FlashLite/KernelForge primitive behind a feature flag, only if the integration surface is reasonable | Controlled A/B benchmark with correctness validation |
| **11 — Capacity/scaling study** | Single GPU first; replicas/distributed only if hardware exists; Pareto frontier + capacity recommendations | Report per §1.9 |
| **12 — Portfolio release** | Polished architecture, benchmark report, demo video, website hero diagram, release notes, evidence-backed resume metrics | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
Multi-model routing; LoRA/adapters; prefix-cache-aware routing; quantized configs; speculative decoding; distributed prefill/decode; multi-GPU tensor parallel; autoscaling on queue/TTFT; GPU-aware K8s scheduling study; request priority classes; web performance dashboard.

### Definition of Done
One command/dev guide launches a usable backend/API; streaming + cancellation work; load bounded by admission policy; end-to-end telemetry works; benchmark harness produces reproducible raw results/plots; Docker/K8s deploy works (or GPU limitation explicitly documented); ≥1 failure/overload scenario demonstrated; runtime interface proven by ≥2 implementations (or one real + one custom/mock); README/website show architecture + measured results; any MiniPaged/FlashLite/KernelForge integration is modular and attributed to its source repo; resume claims traceable to committed evidence.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Do not start by integrating all portfolio projects.
2. Establish a boring, correct mature-backend baseline first.
3. Preserve the runtime/backend abstraction.
4. Add observability before optimization.
5. Benchmark before and after every performance change.
6. Never fabricate GPU or latency numbers.
7. Keep local reproducibility independent of paid cloud infrastructure.
8. Keep custom kernel/runtime integrations feature-gated and reversible.
9. Treat graceful overload/failure as core correctness.
10. Prefer an explainable coherent platform over feature count.

### Per-task requirements (brief mandate)
Each task specifies: component owner/module; external/runtime interface; acceptance tests; telemetry requirements; load/failure acceptance criteria where relevant; local development command; whether an ADR is required before the architectural commitment; and dependencies on other repos/releases.

### Kickoff prompt
> Read `09-helios-spec.md` in full. This is the portfolio capstone: do not plan integration of the other repos before Phase 8. Produce an engineering plan with epics in strict dependency order: interfaces/mock → first runtime → observability → admission → benchmarks → deployment → reliability → CI/CD → optional repo integrations → capacity study → portfolio release. Each task must state its component owner/module, external/runtime interface, acceptance tests, telemetry requirements, load/failure criteria where relevant, local dev command, ADR requirements, and cross-repo dependencies. Decisions D1–D3 need my sign-off before Phase 0/1; use §1.12 defaults for the rest with ADRs. Then implement Phase 0 only (interfaces + mock streaming backend + API tests) and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /benchmark (Phases 4, 5, 10, 11)  →  /cso (Phases 6–8: secrets, RBAC, log-content policy)  →  /qa (Phase 6+: streaming API against local deploy)  →  /ship
```
`/cso` matters here: request-content logging policy (§1.7) and K8s least-privilege are explicit requirements.
