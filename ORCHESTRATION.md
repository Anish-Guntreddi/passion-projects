# Orchestration State — Nine-Project Portfolio Build

**Mission**: Implement all nine spec'd projects to MVP exit criteria via parallel Sonnet subagents with Codex review gates, then ship a GitHub Pages showcase site.
**Orchestrator**: Claude Fable 5. **Started**: 2026-08-17.
**Definition of done (per project)**: all MVP roadmap phases complete (stretch goals excluded) · tests green in WSL · benchmark claims backed by committed raw artifacts · README + ADRs written · fresh-clone reproduction verified.

## Environment (verified 2026-08-17)

RTX 4090 24GB (visible in WSL2) · WSL2 Ubuntu 24.04 · Docker 28.4 · Codex CLI 0.128.0 · gh authenticated (Anish-Guntreddi). Native builds run in WSL: `wsl -d Ubuntu -- bash -lc '...'`; repo mounts at `/mnt/c/Users/guntr/Documents/passion-projects`.

## Status board

| Project | Wave | Current phase | Status | Notes |
|---|---|---|---|---|
| 01 ForgeLM | 1 | 2–3 | 0–1 DONE (91/91 tests, committed 5e0ef7c) · 2–3 in wf_c60f9aab | Transformer + training system next |
| 02 KernelForge | 1b | 2–3 | phases 0–1 DONE (57/57 tests, sanitizer clean, artifacts committed 289480c/84a7308) | reduction/scan + atomics next |
| 03 MiniPaged | 2 | 0–1 | in-progress (wf_c60f9aab) | started early — dep was pedagogical, not code |
| 04 FlashLite | 2 | — | waiting (KernelForge) | |
| 05 ArcServe | 1 | 2–3 | 0–1 DONE (77 tests × 4 sanitizers, a1baf98) · 2–3 in wf_c60f9aab | epoll reactor + parser next |
| 06 PebbleDB | 2 | 0–1 | in-progress (wf_c60f9aab) | modeled on ArcServe build layout |
| 07 xv6-plus | 1 | 2–3 | 0–1 DONE (verifier PASS, b8f916e) · 2–3 in wf_c60f9aab | accounting + xvtop next |
| 08 ReleaseGuard | 1 | 2–3 | 0–1 DONE (race-clean, live kind, c15d36f) · 2–3 in wf_c60f9aab | telemetry + manual canary next |
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
