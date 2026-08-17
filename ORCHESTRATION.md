# Orchestration State — Nine-Project Portfolio Build

**Mission**: Implement all nine spec'd projects to MVP exit criteria via parallel Sonnet subagents with Codex review gates, then ship a GitHub Pages showcase site.
**Orchestrator**: Claude Fable 5. **Started**: 2026-08-17.
**Definition of done (per project)**: all MVP roadmap phases complete (stretch goals excluded) · tests green in WSL · benchmark claims backed by committed raw artifacts · README + ADRs written · fresh-clone reproduction verified.

## Environment (verified 2026-08-17)

RTX 4090 24GB (visible in WSL2) · WSL2 Ubuntu 24.04 · Docker 28.4 · Codex CLI 0.128.0 · gh authenticated (Anish-Guntreddi). Native builds run in WSL: `wsl -d Ubuntu -- bash -lc '...'`; repo mounts at `/mnt/c/Users/guntr/Documents/passion-projects`.

## Status board

| Project | Wave | Current phase | Status | Notes |
|---|---|---|---|---|
| 01 ForgeLM | 1 | 4–5 | 0–3 DONE (5e0ef7c, 18090c7: 190 tests, gate-verified) · 4–5 in wave wf_3ede6770 | generation/eval + scaling experiment |
| 02 KernelForge | 1b | 4–5 | 0–3 DONE (289480c/84a7308, 77ad541/68b54a3: 101 tests, doc numbers recomputed at gate) · 4–5 in wave wf_3ede6770 | GEMM + AI primitives; V0 reduction benchmark backfill first |
| 03 MiniPaged | 2 | 2–3 | 0–1 DONE (235402e: 123 tests, fresh-clone verified) · 2–3 in wave wf_3ede6770 | paged KV allocator + scheduler |
| 04 FlashLite | 2 | 0–1 | in impl wf_cbae7b4a | started — KernelForge 0–1 committed; conventions mirrored from it |
| 05 ArcServe | 1 | 2–3 | 0–1 DONE (a1baf98) · 2–3 in impl wf_cbae7b4a | continues from partial socket.hpp edit (header-only, didn't link) |
| 06 PebbleDB | 2 | 2–3 | 0–1 DONE (9550b96: 74 tests × ASan/UBSan+Release) · 2–3 in wave wf_3ede6770 | MemTable + SSTable |
| 07 xv6-plus | 1 | 4–5 | 0–3 DONE (b8f916e, c78a9e1: QEMU suite green, zombie-filter fix) · 4–5 in wave wf_3ede6770 | scheduler experiment + VM extension |
| 08 ReleaseGuard | 1 | 4–5 | 0–3 DONE (c15d36f, 2d6892a: live kind verified, namespace fix) · 4–5 in wave wf_3ede6770 | policy/evaluator core + release state machine |
| 09 Helios | 3 | — | waiting (capstone) | must be last |
| Showcase site | 4 | — | waiting (all) | GitHub Pages |

## Wave plan

- **Wave 0** (in flight): WSL provisioning — base toolchain (g++, cmake, ninja, Go, QEMU, riscv-gcc) then CUDA toolkit 12.x. Scaffolding + CI skeleton.
- **Wave 1**: ForgeLM · ArcServe · xv6-plus · ReleaseGuard in parallel, phases 0–1 each, then continue phase-by-phase. **1b**: KernelForge once nvcc lands.
- **Wave 2**: MiniPaged · FlashLite · PebbleDB as their dependencies clear.
- **Wave 3**: Helios.
- **Wave 4**: Showcase site → GitHub Pages deploy.

## Quality loop (every phase of every project)

1. Sonnet implementer builds the phase + tests (never touches git).
2. Tests executed in WSL; real output required.
3. Codex reviews the project directory against its spec (`codex exec -s read-only`).
4. Sonnet fixer resolves blocking findings; re-review until clean.
5. Verifier checks the spec's phase **exit criterion** goal-backward.
6. Orchestrator commits (small, scoped) and updates this board.

## Decision log

- **2026-08-17 · Monorepo layout** — all projects under `projects/<nn-name>/`; showcase site deploys from this repo. Splittable later.
- **2026-08-17 · Spec open-decisions** — adopt each spec's recommended default; ADR it in the project; conservative compute budgets where specs say "human decision".
- **2026-08-17 · Codex lanes** — reviewer lane (`codex exec -s read-only`, effort=high) on every increment; offload lane for mechanical tasks. NOTE: Codex on Windows cannot write (sandbox hard-falls to read-only); offload pattern = Codex drafts content, orchestrator reviews and applies. Caught a real bug in Codex's draft this way (`--index-url` vs `--extra-index-url`).
- **2026-08-17 · CI at root** — per-project workflows at `.github/workflows/ci-<name>.yml` with paths filters; KernelForge compile-only (no GPU runners); pushed 25f749f.
- **2026-08-17 · MiniPaged/PebbleDB started early** — their "dependencies" were pedagogical (human learning order), not code interfaces; parallel agents don't need them.
- **2026-08-17 · Session restart mid-wave** — six-pack (wf_c60f9aab) and KernelForge 2–3 (wf_4f978ff9) died with their session, leaving substantial uncommitted work (ForgeLM model+training, KernelForge 2–3, xv6 accounting, ReleaseGuard telemetry, MiniPaged/PebbleDB scaffolds, ArcServe header stub). Recovery: gate workflow wf_85e7b308 runs the full quality loop (test → Codex → fix → verify) over the six uncommitted trees; impl workflow wf_cbae7b4a takes ArcServe 2–3 forward from the stub and starts FlashLite 0–1 (dep committed). Same precedent as MiniPaged: FlashLite's KernelForge dependency is conventions, not code interfaces.
- **2026-08-17 · Gate rule for GPU benchmarks** — correctness tests may run in parallel across projects, but benchmark artifacts that back README claims are regenerated serialized (quiet GPU) after gates pass, then committed separately (constraint 9).
- **2026-08-17 · Gate outcome (wf_85e7b308)** — 5/6 commit-ready on first verify; KernelForge blocked on doc-vs-data transcription errors in methodology §10/§9.2 + README (V3 histogram table computed from wrongly transcribed values, reversing the uniform/skewed direction claim). Orchestrator recomputed all numbers from committed raw JSONL and corrected docs before committing. Lesson: every gate now includes recompute-doc-claims-from-raw as a standard verifier step.
- **2026-08-17 · GPU benchmark lock** — measured-claim benchmark runs must hold the repo-level lock (`mkdir /mnt/c/.../.gpu-bench-lock`, atomic; rmdir to release; gitignored). Development and correctness tests parallelize freely; only measurement serializes. FlashLite 0-1 launched before this protocol — its baseline artifacts get a quiet-GPU sanity re-check before committing.
