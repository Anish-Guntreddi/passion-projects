# ForgeLM — Project Spec (PRD · Tech Stack · Roadmap)

**Project:** ForgeLM — From-Scratch Language Model Training Stack
**Portfolio position:** 01 of 09 · Track A (ML systems) · prerequisite for MiniPaged
**Source of truth:** "01 - ForgeLM - Fable Project Planning Brief" (Google Drive)
**Status:** Ready for Claude Code execution

---

## Part 1 — Product Requirements Document

### 1.1 Overview
ForgeLM is a portfolio-grade language-model engineering project proving the author understands how a decoder-only Transformer is built, trained, evaluated, checkpointed, and sampled — without high-level pretrained-model wrappers. It must be educational enough to explain every component and engineered seriously enough to look like a real small-scale training system.

### 1.2 Problem statement
High-level LLM libraries hide the mechanics that matter for model engineering. ForgeLM exposes them through a small, readable implementation and a disciplined experiment harness. The repo must answer: how raw text becomes tokens; how tensors flow through attention/MLP blocks; what memory/compute costs arise from model dimensions and sequence length; how initialization, optimizer settings, clipping and precision affect training; how checkpoint correctness/reproducibility is verified; and how architectural choices change quality and resource use.

### 1.3 Users & success definition
Primary "user" is a technical reviewer (interviewer, hiring manager). The project succeeds when a reviewer can clone the repo, train a small model end-to-end on a documented dataset, reproduce an evaluation run, inspect scaling/benchmark results, and understand how every architectural component maps to the Transformer computation.

### 1.4 Functional requirements (MVP)
- **FR1** Reproducible Python environment; all execution is CLI + config driven.
- **FR2** Trainable tokenizer (small BPE or equivalent educational tokenizer): train vocabulary, encode, decode, save/load vocab/merges, deterministic special-token handling, round-trip tests, compression statistics.
- **FR3** Dataset pipeline: read local text, tokenize, fixed-length next-token examples, deterministic train/validation split; optional memory-mapped/sharded representation after baseline.
- **FR4** Decoder-only Transformer from basic PyTorch modules: token embeddings + output projection, causal self-attention with multi-head reshape/merge, RoPE, RMSNorm, SwiGLU (or equivalent gated MLP), residual connections, causal masking. GQA may be Phase 2 / stretch.
- **FR5** Explicit model config: vocab_size, context_length, d_model, n_layers, n_heads, optional n_kv_heads, d_ff, dropout, RoPE params, dtype.
- **FR6** Training engine: cross-entropy next-token objective, AdamW, LR schedule, gradient clipping, gradient accumulation, mixed precision where hardware permits, validation intervals, logging.
- **FR7** Checkpointing: save/resume including optimizer state and RNG state where practical; resumed training produces a documented equivalent trajectory within expected numerical variation.
- **FR8** Generation: greedy plus temperature / top-k / top-p sampling; must run from a saved checkpoint, not only live training state.
- **FR9** Evaluation: validation loss/perplexity, deterministic eval mode, configurable token budget, generation sample reports.
- **FR10** Benchmark harness: tokens/sec, step time, peak GPU memory where available, CPU-fallback behavior, sensitivity to batch size / context length / model size.
- **FR11** At least three model-size experiments (≈ small/medium/larger within compute budget) with recorded parameter count, tokens trained, step time, throughput, peak memory and validation loss.
- **FR12** Automated unit + integration tests (see §1.8).

### 1.5 Non-goals (MVP)
Distributed multi-node training; RLHF/RLVR; production inference serving; billion-parameter training; web-scale data pipelines; custom CUDA kernels; beating mature libraries on throughput. These are later projects or stretch phases only.

### 1.6 Deliverable artifacts (website/resume)
- Architecture diagram of the forward/training path.
- Model-config table (static or interactive).
- Loss curves across ≥3 model configurations.
- Memory/throughput plots vs batch size and sequence length.
- Sample generations with documented decoding settings.
- "What I implemented myself" section.
- Target resume narrative (numbers filled in only from committed evidence): *"Built a decoder-only Transformer training stack from scratch in PyTorch, implementing tokenization, attention, RoPE, RMSNorm, SwiGLU, optimizer/training infrastructure, checkpoint recovery and evaluation; trained multiple model sizes and characterized compute, memory and validation-loss scaling."*

### 1.7 Engineering quality bar
No notebook-only implementations (notebooks may analyze results only); no magic global config; type/shape assumptions documented; no copied full model implementation from a framework; small understandable functions/classes; benchmark code separated from model-correctness code; reproducibility documented; CI gates tests/lint/type checks.

### 1.8 Test requirements
Unit: tokenizer round trips + special tokens; RoPE shape/determinism; RMSNorm vs reference formula; causal mask prevents future-token dependence; attention shapes; parameter-count sanity; sampling constraints; checkpoint round trip.
Integration: overfit a tiny batch/dataset; train→save→load→generate; resume from checkpoint; CPU smoke test; GPU smoke test if CI/hardware permits.
Numerical tests use tolerances, never exact float equality.

### 1.9 Open decisions (require human sign-off before the relevant phase; recommended defaults given)
- **D1** Tokenizer complexity vs schedule → *default: minimal byte-level BPE, ~500 LOC ceiling.*
- **D2** Dataset: small, legal, reproducible → *default: TinyStories or a public-domain text corpus; document license.*
- **D3** CPU-only dev path vs CUDA expectation → *default: CPU path must work for tests; GPU for real runs.*
- **D4** Weight tying in MVP → *default: yes (simple, standard).*
- **D5** GQA in MVP or stretch → *default: stretch.*
- **D6** Precision strategy → *default: bf16 autocast if GPU supports it, else fp32.*
- **D7** Checkpoint format + RNG-state guarantees → *default: torch.save dict with model/optimizer/scheduler/RNG states + config hash; ADR required.*
- **D8** Compute budget for the three scaling runs → *human decision.*
Prefer the simplest design that preserves the educational objective.

### 1.10 Risks
Tokenizer scope creep (mitigate via D1 LOC ceiling); scaling runs exceeding compute budget (mitigate: dry-run cost estimate before Phase 5); irreproducibility from unpinned versions (mitigate: lockfile + recorded env in every benchmark artifact).

---

## Part 2 — Tech Stack Plan

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | Brief requirement |
| Framework | PyTorch | Brief requirement; only basic modules (no transformers-library model wrappers) |
| Config | Typed dataclasses **or** Pydantic (pick one, ADR it) | Explicit config layer; no Hydra unless justified — keep it simple |
| CLI | Typer (or argparse) | Config-driven execution per FR1 |
| Testing | pytest | Standard |
| Lint/format | Ruff | Brief requirement |
| Type checking | Pyright (or mypy) | Brief requirement |
| Packaging | pyproject.toml | Brief requirement |
| Experiment output | Local JSON/CSV first; optional W&B/MLflow only after local reproducibility works | Brief requirement |
| Acceleration | torch.compile — only after baseline correctness | Brief requirement |
| CI | GitHub Actions: lint + type + unit tests + CPU smoke test | Quality bar §1.7 |

### Repository structure
```
forgelm/
  pyproject.toml
  README.md
  configs/
  src/forgelm/
    tokenizer/  data/  model/  training/
    evaluation/ generation/ benchmarks/ cli/
  tests/unit/  tests/integration/
  scripts/
  benchmarks/results/  benchmarks/plots/  benchmarks/methodology.md
  docs/architecture.md  docs/model-math.md  docs/reproducibility.md  docs/decisions/
  examples/
```
Component boundaries are load-bearing: tokenizer owns text↔token; data owns datasets/packing/splits; model owns architecture only; training owns optimization/precision/checkpointing/schedules; evaluation owns loss/perplexity/generation eval; benchmarks own throughput/memory experiments; CLI/config composes without introducing model logic.

---

## Part 3 — Roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0 — Repo foundation** | Scaffold, environment, lint/type/test CI, config schema, seed handling, smoke-test command, architecture ADR | CI green; placeholder end-to-end command runs deterministically |
| **1 — Tokenization & data** | Tokenizer train/encode/decode; dataset loader; sequence construction; dataset statistics; unit tests | Text → deterministic training batches, reconstructable |
| **2 — Transformer correctness** | RMSNorm, RoPE, attention, MLP, TransformerBlock, full decoder; reference/cross-check tests on tiny tensors | Forward pass numerically sane, shape-safe, trainable on toy overfit dataset |
| **3 — Training system** | AdamW path, scheduler, clipping, mixed precision, grad accumulation, validation, checkpoint save/resume | Interrupted training resumes with documented equivalent trajectory |
| **4 — Generation & evaluation** | Checkpoint loader, generation modes, perplexity/loss evaluator, sample-report format | Trained checkpoint evaluates and generates via clean CLI |
| **5 — Scaling experiment** | ≥3 configurations trained; params, tokens, step time, throughput, peak memory, val loss recorded | Benchmark dataset + plots committed with methodology |
| **6 — Portfolio hardening** | Polished README, diagrams, reproducibility guide, benchmark report, website assets, release tag | Fresh-clone instructions verified |

Benchmark plan (Phase 5 minimum): tokens/sec vs model size; peak memory vs context length; peak memory vs batch size; val loss vs training tokens per config; optional torch.compile A/B. Every result records hardware, software versions, dtype, seed, model config, dataset, warmup policy, sample count.

### Definition of Done
Fresh clone installs and passes tests; small training run reproducible; checkpoint resumes and generates; ≥3 configs benchmarked; methodology documented; architecture explained without external library internals; README shows measured results; website diagrams/plots exist; **no resume metric claimed without committed evidence.**

---

## Part 4 — Claude Code Handoff

### Agent execution rules (verbatim from brief; enforce as hard constraints)
1. Inspect current repository state before every phase.
2. Do not replace educational components with high-level model wrappers.
3. Implement correctness before optimization.
4. Add tests in the same change as functionality.
5. Keep commits narrowly scoped and reviewable.
6. Never invent benchmark results.
7. Record architecture changes in ADRs (`docs/decisions/`).
8. Prefer deterministic local development over external managed services.
9. Do not proceed past a failed acceptance criterion.
10. Preserve readable code suitable for technical-interview discussion.

### Kickoff prompt (paste into Claude Code with this spec in the repo)
> Read `01-forgelm-spec.md` in full. Produce a detailed engineering plan with: epics ordered by the Part 3 phase dependencies; small tasks sized as atomic commits/PRs; exact acceptance criteria per task; interfaces/data structures to define before implementation (config schema, tokenizer API, checkpoint format); test requirements attached to each task; benchmark checkpoints after Phases 2, 3 and 5; ADRs required for decisions D1–D7 before implementing the affected component; a repo-bootstrap sequence for Phase 0; a risk register; and explicit "do not implement yet" boundaries (everything in §1.5 and D5). Then implement Phase 0 only and stop for review.

### Suggested gstack sequence
```
/office-hours      (pressure-test D1–D8 defaults)
/autoplan          (turn this spec into the phased engineering plan)
[implement Phase N]
/review            (each phase; focus: shape safety, determinism, test coverage)
/benchmark         (after Phases 3 and 5)
/ship              (per phase)
/retro             (after Phase 6)
```
Skip `/qa` (no browser surface) and `/cso` (no sensitive surface).
