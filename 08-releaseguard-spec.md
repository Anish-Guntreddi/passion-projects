# ReleaseGuard — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** ReleaseGuard — SLO-Aware Progressive Delivery and Automated Rollback Controller
**Portfolio position:** 08 of 09 · Track E (SRE/platform) · self-contained · later integrates with Helios (its Phase 8)
**Source of truth:** "08 - ReleaseGuard - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
ReleaseGuard is a software-lifecycle/SRE project: a system that takes a build from commit to production-like deployment, observes a canary, evaluates service-level signals, and makes an **auditable** promote/pause/rollback decision. The important implementation is the **decision/controller logic and evidence pipeline** — not a collection of YAML files.

### 1.2 Demo application (deployment target, deliberately simple)
A small instrumented HTTP service with `/health`, `/work`, `/version`, and a small data dependency/mock. Two image versions must be able to demonstrate: healthy release; latency regression; elevated errors; resource/memory pressure or dependency failure.

### 1.3 Functional requirements (MVP)
- **FR1** CI (GitHub Actions): build/test container image; image vulnerability/dependency scan; immutable version tagging/digest usage.
- **FR2** Local Kubernetes environment (kind/k3d/minikube) for reproducible development; deployment manifests via **one** primary approach (Helm or Kustomize); stable + canary workloads; controlled traffic split or simulated canary sampling.
- **FR3** OpenTelemetry instrumentation + Prometheus metrics; latency/errors/version queryable **per stable/canary identity**.
- **FR4** ReleaseGuard controller/service with:
  - Typed policy config (YAML/JSON with validation) defining: metrics/signals; absolute SLO thresholds; relative canary-vs-baseline guardrails where useful; evaluation interval/window; minimum sample count/traffic; consecutive healthy/unhealthy window counts; maximum rollout duration; **action on missing telemetry**.
  - Decision states: PROMOTE, PAUSE, ROLLBACK, INCONCLUSIVE.
  - Decision state machine: PENDING → DEPLOYING → OBSERVING → HEALTHY/DEGRADED/INCONCLUSIVE → PROMOTING/ROLLING_BACK/PAUSED → COMPLETED/FAILED; every transition carries reason codes and timestamped evidence.
  - Deployment audit log answering: *"Why did ReleaseGuard make this decision?"*
  - Kubernetes deploy/rollback adapter — **safe and idempotent**.
- **FR5** Failure-injection demo scripts (latency, error rate, missing telemetry, saturation/dependency failure).
- **FR6** **The controller must never silently treat missing data as healthy.**
- Example signals: HTTP 5xx rate, request P95/P99 latency, successful throughput, saturation/resource signal, optional business-success metric.

### 1.4 Non-goals
Full Argo Rollouts/Flagger replacement; production multi-cluster control plane; arbitrary deployment strategies; ML anomaly detection in MVP; managing real customer traffic; building a monitoring vendor.

### 1.5 Deliverable artifacts (website/resume)
commit→CI→artifact→canary→decision→promotion/rollback architecture diagram; canary traffic timeline; dashboard screenshots; one successful promotion and one automatic rollback incident; latency/error graphs around an injected regression; audit-log decision explanation. Resume narrative filled from evidence: *"Built an SLO-aware progressive-delivery controller that evaluated canary telemetry and automatically promoted, paused or rolled back deployments based on latency/error-budget guardrails; integrated CI, container security, Kubernetes, OpenTelemetry/Prometheus and reproducible failure-injection tests."*

### 1.6 Security / safety design
Least-privilege Kubernetes RBAC for the controller; no production credentials committed; CI secrets avoided locally or held in GitHub secret mechanisms; images referenced immutably for release decisions; security scans as gates with documented false-positive handling; append-oriented/traceable audit trail; SBOM + image/dependency scanning (signing is a later phase).

### 1.7 Test strategy
Unit: policy parsing, window aggregation, threshold logic, relative comparisons, state transitions.
Integration: Prometheus query adapter; Kubernetes adapter with fixtures/fake client.
E2E: healthy v1→v2 promotion; latency-regression rollback; elevated-error rollback; missing-telemetry pause/inconclusive.
Idempotency: repeated reconciliation must not duplicate rollout actions.
Failure: Prometheus unavailable; transient Kubernetes API failure; controller restart during observation/action.

### 1.8 Required demo scenarios (each with graphs + audit events recorded)
A. Healthy canary → promote. B. Injected latency regression (P95/P99) → rollback. C. Injected error regression → rollback. D. Missing metrics → evaluator refuses unsafe promotion. E. Brief transient blip → consecutive-window policy avoids overreacting when configured.

### 1.9 Open decisions (recommended defaults)
- **D1** Go vs Python controller → *default per brief: Go (stronger Kubernetes/controller signal); document why.*
- **D2** Traffic-splitting mechanism that stays lightweight locally → *default: replica-count-based split or deterministic client-side sampling; ADR.*
- **D3** Prometheus query strategy → *default: instant + range queries via HTTP API behind a metrics-query interface.*
- **D4** Audit persistence → *default: append-only JSONL file (simplest robust option); SQLite or K8s CR/status as alternatives; ADR.*
- **D5** Standalone service vs operator/CRD pattern → *default: standalone controller service for MVP; CRD/operator is stretch.*
- **D6** Absolute vs relative thresholds in MVP → *default: absolute thresholds first, relative guardrails second.*
- **D7** Terraform cloud deployment → *default per brief: stretch, not MVP.*
- Policy threshold values are **human-review decisions**.

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Controller | Go (per D1) | Kubernetes/controller portfolio signal |
| Demo service | Go (or Python/TypeScript); keep simple | Brief allows any; one language reduces toil |
| Kubernetes | kind or k3d local cluster | Brief requirement; cloud never required for MVP |
| Manifests | Helm **or** Kustomize — choose one primary (ADR) | Brief requirement |
| Packaging | Docker/OCI, immutable tags + digests | FR1 |
| CI | GitHub Actions | Brief requirement |
| Observability | OpenTelemetry + Prometheus; Grafana optional for visualization | Brief requirement |
| Policy config | YAML/JSON with typed validation | FR4 |
| IaC | Terraform only for optional cloud env (stretch) | D7 |
| Security | SBOM (e.g. syft) + image/dependency scanning (e.g. trivy/grype); signing later | §1.6 |
| Testing | Go unit tests; integration with fake clients; cluster e2e scripts | §1.7 |

### Repository structure
```
releaseguard/
  controller/cmd/  controller/internal/{policy,evaluator,metrics,rollout,audit}/
  demo-service/
  deploy/{kubernetes,helm-or-kustomize,local-cluster}/
  observability/{otel,prometheus,grafana}/
  .github/workflows/
  tests/{unit,integration,e2e,failure}/
  experiments/{scenarios,raw,plots}/
  docs/architecture.md  docs/slo-policy.md  docs/security.md  docs/incident-demo.md  docs/decisions/
```
Flow: Git commit → CI test/build/scan → immutable image → deployment request → stable + canary workloads → telemetry collector/query layer → evaluator (policy engine | statistics/window aggregator | decision state machine) → K8s deploy/rollback adapter → audit/event log → dashboard/CLI.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Demo service + CI** | Instrumented service, tests, container build, version endpoint, local run | CI produces immutable tagged/digested image |
| **1 — Local deployment foundation** | Reproducible kind/k3d cluster, stable deployment, service, smoke-test script | Fresh clone can deploy locally |
| **2 — Telemetry** | OTel traces/metrics, Prometheus scrape/query, dashboard/basic scripts | Latency/errors/version queryable per stable/canary identity |
| **3 — Manual canary** | Two versions with controlled split/routing; no automated decisions yet | Healthy and unhealthy canary telemetry can be generated |
| **4 — Policy/evaluator core** | Typed policy schema, metrics-query interface, evaluation windows, threshold/relative comparisons, unit tests | Recorded telemetry fixtures produce deterministic decisions |
| **5 — Release state machine** | Lifecycle + audit log; PAUSE/PROMOTE/ROLLBACK decision objects | Simulated rollout executes full state sequence |
| **6 — Kubernetes action adapter** | Idempotent promotion/rollback application | Unhealthy canary auto-rolls back; healthy canary promotes |
| **7 — Failure scenarios** | Scripted latency/error injection, missing telemetry, saturation/dependency failure | Policies respond safely; audits explain why |
| **8 — Secure delivery** | SBOM, vuln/dependency scanning, least-privilege SA/RBAC review; optional signing | Security checks run in the release pipeline |
| **9 — Reliability hardening** | Controller restart/idempotency, duplicate events, API/query failures, timeouts, audit persistence | Failure tests §1.7 pass |
| **10 — Portfolio hardening** | Demo recording, diagrams, dashboards, incident/postmortem, README, website plots | Fresh-clone reproduction verified |

### Stretch goals (post-MVP only)
CRD/operator API; multi-step rollouts (5%→25%→50%→100%); statistical significance/sequential testing; error-budget burn-rate policies; service-mesh traffic control; Slack/GitHub status integration; policy-as-code checks; Sigstore/cosign; Terraform cloud demo; chaos automation.

### Definition of Done
Fresh clone stands up the local environment; stable/canary independently observable; healthy canary auto-promotes; unhealthy canary auto-rolls back; **missing telemetry cannot silently promote**; every decision has an evidence/audit record; security/CI gates run automatically; failure scenarios are scripted/reproducible; website-ready dashboards + incident story exist.

---

## Part 4 — Claude Code Handoff

### Agent execution rules (hard constraints)
1. Never auto-promote with missing evidence.
2. Make all rollout actions idempotent.
3. Keep the local MVP independent of paid cloud services.
4. Do not invent SLO results.
5. Treat controller logic as the primary code — deployment YAML is not the accomplishment.

### Per-task requirements (brief mandate)
Each task specifies: an observable acceptance criterion; rollback/idempotency considerations; required metrics and labels; tests; a local reproducibility command; and which policy thresholds need human review.

### Kickoff prompt
> Read `08-releaseguard-spec.md` in full. Produce an engineering plan with epics in strict dependency order: service/CI → cluster/deploy → telemetry → manual canary → evaluator → state machine → automated actions → failure tests → security → hardening → portfolio. Every task must state an observable acceptance criterion, rollback/idempotency considerations, required metrics/labels, tests, and a local reproducibility command. ADR decisions D1–D6 before their phases (defaults in §1.9 unless I override); flag all policy threshold values for my review before Phase 4 lands. Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours  →  /autoplan  →  [implement per phase]  →  /review  →  /cso (Phase 8: RBAC, supply chain)  →  /qa (Phase 7 demo scenarios against the local cluster)  →  /ship
```
`/cso` genuinely applies here — RBAC, supply-chain scanning and secrets handling are core requirements, not add-ons.
