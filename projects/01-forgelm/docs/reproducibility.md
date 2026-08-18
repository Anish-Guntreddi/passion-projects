# Reproducibility

## Environment

- Host: Windows 11 Pro, native builds/tests run inside **WSL2 Ubuntu**.
- Python: 3.12 (`python3 -m venv .venv` at the project root — see
  `scripts/setup_env.sh`).
- PyTorch: installed from the **cu124** wheel index
  (`pip install torch --index-url https://download.pytorch.org/whl/cu124`),
  which works on CPU-only machines and uses CUDA automatically when a
  compatible GPU is visible (see `docs/decisions/0004-cpu-only-dev-path.md`).
- Exact installed versions are pinned in `requirements-lock.txt`
  (`pip freeze` output), committed alongside this file, and **actually
  installed by default**: `scripts/setup_env.sh` (no arguments) runs `pip
  install -r requirements-lock.txt`, not a fresh `pyproject.toml`
  resolution, so a fresh clone gets the exact same versions rather than
  whatever is newest on PyPI/the torch index that day. Regenerate the lock
  deliberately with `scripts/setup_env.sh --update-lock` whenever
  `pyproject.toml`'s dependency ranges change, review the diff, and commit
  it as its own change.
- GPU (when present): recorded per-run inside benchmark/smoke output
  (`torch.cuda.get_device_name(0)`), never hard-coded in docs.

**Measured on this machine, 2026-08-17** (verbatim from a real run, not
retyped by hand):

```
OS:            Ubuntu 24.04.1 LTS (WSL2, host: Windows 11 Pro)
Python:        3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
torch:         2.6.0+cu124
torch.version.cuda:  12.4
CUDA available:      True
GPU:                 NVIDIA GeForce RTX 4090 (24564 MiB)
NVIDIA driver:       591.86 (nvidia-smi-reported CUDA 13.1; the torch cu124
                     wheel targets the CUDA 12.4 runtime and runs fine
                     under a newer driver -- this is normal forward
                     compatibility, not a mismatch to "fix")
```

Full pinned dependency list: `requirements-lock.txt` (committed).

## Seeding

`forgelm.seed.set_seed(seed)` is the single reproducibility entry point:

- Seeds Python's `random`, NumPy, and torch (CPU + all visible CUDA
  devices).
- Sets `PYTHONHASHSEED` (affects subprocesses spawned afterward, not the
  current process's already-initialized hash seed).
- Requests `torch.use_deterministic_algorithms(True, warn_only=True)` and
  disables cuDNN benchmarking/enables cuDNN deterministic mode.

Every CLI command that needs reproducibility calls `set_seed()` before
doing anything else. Training calls it once at the start of a run and
additionally persists the full RNG state in checkpoints (D7, ADR 0009 —
verified by `tests/integration/test_checkpoint_resume.py`).

## Determinism guarantees at this phase

- **Tokenizer training** (`ByteLevelBPETokenizer.train`) is a pure function
  of the input text: no RNG involved. Byte-pair tie-breaks are resolved by
  explicit pair-value comparison, not dict/set iteration order, so retraining
  on the same corpus always yields the same merge list (asserted by
  `tests/unit/test_tokenizer.py::test_training_is_deterministic` and
  `tests/integration/test_pipeline.py::test_tokenizer_training_is_deterministic_on_example_corpus`).
- **Train/val split** (`forgelm.dataset.train_val_split`) is a pure
  function of token-stream length and `val_fraction`: no RNG.
- **Batch order** (`forgelm.dataset.make_dataloader`) is a deterministic
  function of the `seed` argument (backed by a dedicated
  `torch.Generator`), for single-process (`num_workers=0`) iteration.
- **The `forgelm smoke` command**: identical `(seed, size, device)` always
  produces a matching checksum **to within numeric tolerance** (never
  claimed as exact float equality, per the project-wide numerical-testing
  rule) -- on CPU this is bit-identical in practice for a fixed op
  sequence, but on CUDA it is only guaranteed to that tolerance, not
  bit-for-bit, because `torch.use_deterministic_algorithms(True,
  warn_only=True)` (see `forgelm.seed.set_seed`) *warns* rather than
  failing when an op (e.g. cuBLAS matmul without
  `CUBLAS_WORKSPACE_CONFIG` set) has no deterministic kernel available --
  it does not guarantee every op actually runs deterministically. Verified
  two ways in `tests/integration/test_smoke_cpu.py` and
  `test_smoke_gpu.py`: by calling `run_smoke()` twice in-process, and by
  invoking the actual `forgelm` CLI as a subprocess twice and comparing
  stdout.

Numerical comparisons in tests use a tolerance, never exact float equality
(project-wide rule) — even where CPU float ops are in practice bit-identical
for a fixed op sequence, so the same test code stays correct once GPU
non-associative-reduction paths are exercised.

## Determinism guarantees added in Phases 2–5

- **Training** (`forgelm.training.loop.Trainer`): `iter_batches(dataset,
  batch_size, seed)` is a pure function of its three arguments (see its
  docstring), so a training run's entire batch stream is reproducible
  from `(dataset, batch_size, seed)` alone. Combined with the checkpoint's
  full RNG-state capture (`forgelm.training.checkpoint`), an interrupted
  and resumed run reproduces the uninterrupted run's per-step losses
  within tolerance — verified by
  `tests/integration/test_checkpoint_resume.py`.
- **Generation** (`forgelm.generation.sampling.generate`): greedy decoding
  (`do_sample=False`) is exactly deterministic (pure argmax, no RNG
  involved at all). Sampling mode uses a dedicated `torch.Generator`
  seeded from `GenerationSettings.seed`, independent of the global RNG
  stream any other code in the process may have already consumed — the
  same `(prompt, GenerationSettings)` always reproduces the same
  continuation, verified by
  `tests/unit/test_generation_sampling.py::test_generate_sampling_is_reproducible_given_the_same_seed`.
- **Evaluation** (`forgelm.evaluation.perplexity.evaluate_loss`): walks
  the dataset with `shuffle=False`, so evaluation order needs no seed at
  all and is trivially reproducible.
- **Benchmarks** (`forgelm.benchmarks.harness.run_benchmark`): seeds
  before constructing the model/optimizer, so `final_train_loss`/
  `val_loss` are reproducible given the same config
  (`tests/unit/test_benchmark_harness.py::test_run_benchmark_is_reproducible_given_the_same_seed`)
  — timing fields (`tokens_per_sec`, `step_time_ms_*`) are explicitly
  *not* claimed reproducible to the same tolerance, since wall-clock
  timing is inherently sensitive to system load; see
  `benchmarks/methodology.md`.

## How to reproduce the full Phase 0–6 test run

```bash
cd projects/01-forgelm
bash scripts/setup_env.sh      # installs the exact pinned versions from
                                # requirements-lock.txt (reproducible default)
source .venv/bin/activate
pytest -v
```

The exact command used to produce the results recorded in this project's
final phase report is:

```bash
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/guntr/Documents/passion-projects/projects/01-forgelm && source .venv/bin/activate && pytest -q'
```

Last real result (2026-08-17, environment above, GPU tests included since
an RTX 4090 was visible; covers Phases 0–6 in full, including the six new
Phase 6 release-metadata regression tests in
`tests/unit/test_release_metadata.py`, which brought the suite from 263 to
269 tests, on top of everything Phase 0–5 already covered: generation/
evaluation/benchmark-harness unit + integration tests, the Phase 4 CLI
(`generate`/`evaluate`/`sample-report`) and Phase 5 CLI (`benchmark run`)
subprocess tests, and the exact top-k tie handling / `evaluate_loss`
token-budget-overshoot / `BenchmarkResult` dataset-identity regression
tests added during Phase 4–5 review rounds):

```
269 passed, 10 warnings in 858.59s (0:14:18)
```

(Wall-clock varies run to run because this WSL VM's GPU is frequently
shared with several other portfolio projects' agents running concurrently
-- not a regression in the suite itself; the from-scratch fresh clone
below, run when the machine was less contended, completed the identical
269-test suite in 64.29s. See `benchmarks/methodology.md`'s note on timing
fields never being claimed reproducible to a tight tolerance under
shared-machine load.)

All 10 warnings are the same expected `torch` notice that CuBLAS matmul
doesn't have a fully deterministic kernel without setting
`CUBLAS_WORKSPACE_CONFIG` (triggered by the GPU-exercising tests and the
`torch.compile` A/B unit test) -- exactly why `test_smoke_gpu.py` compares
checksums with a tolerance instead of exact equality rather than treating
it as a bug to silence.

`ruff check`, `ruff format --check`, and `pyright` all report clean (0
errors/warnings) on this same commit of the source.

Historical reference: Phase 0/1's original (smaller) suite was also
verified with CUDA hidden (`CUDA_VISIBLE_DEVICES="" pytest -v`), to
confirm the D3 "CPU path must work for tests" guarantee independently of
this machine's GPU and to mirror what the GitHub Actions runner (no GPU)
will see -- `89 passed, 2 skipped` (the 2 skips being `test_smoke_gpu.py`'s
`requires_cuda`-marked tests). The CPU-only path continues to be the
concrete, always-enforced guarantee via `.github/workflows/ci-forgelm.yml`
(a CPU-only GitHub Actions runner) on every push/PR, rather than a number
re-verified locally on every phase.

## Fresh-clone verification (Phase 6 exit criterion)

The spec's Phase 6 exit criterion and Definition of Done both require a
**real** fresh-clone check ("fresh clone installs and passes tests; small
training run reproducible"), not a re-run of tests inside the same working
tree that wrote them. This section is a from-scratch transcript, performed
2026-08-17 on the same machine/environment recorded above.

Because this project lives in a monorepo whose current Phase 6 changes are
still local/uncommitted at implementation time (the multi-project
orchestrator commits and tags releases, not this project's own tooling —
see ADR 0011, decision 3), "clone the pushed repo" isn't yet possible for
*this* revision. The closest faithful equivalent, and what was actually
run: stage this project's exact current working tree (everything that
would be committed — i.e. respecting `.gitignore`, so no `artifacts/`,
`.venv/`, or cache directories) as a standalone local git commit, then
perform a **real `git clone`** of that commit into an entirely separate
directory outside this repo, and follow `README.md` verbatim from there
with no shared state (fresh `.venv`, fresh `artifacts/`, nothing copied
over):

```bash
# 1. Stage the current working tree as a real commit, gitignore-respecting
rsync -a --exclude=.venv --exclude=artifacts --exclude=runs \
    --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude='*.egg-info' \
    projects/01-forgelm/ ~/forgelm-verify/staged-source/
cd ~/forgelm-verify/staged-source && git init -q && git add -A && \
    git commit -q -m "staged snapshot for fresh-clone verification"

# 2. A REAL git clone into an unrelated directory (local transport, but a
#    genuine `git clone`, not a copy) -- this is the "fresh clone"
git clone -q ~/forgelm-verify/staged-source ~/forgelm-verify/fresh-clone
cd ~/forgelm-verify/fresh-clone
ls artifacts 2>&1   # -> "No such file or directory" (correctly gitignored)
ls .venv 2>&1        # -> "No such file or directory" (correctly gitignored)

# 3. README.md's Quickstart, verbatim, from this fresh clone
bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/run_checks.sh    # ruff check + ruff format --check + pyright + pytest -v
```

**Results, verbatim from this run:**

- `git clone` of the staged 125-file commit: succeeds; `ls artifacts` /
  `ls .venv` both fail with "No such file or directory" in the clone,
  confirming nothing gitignored leaked into the commit.
- `bash scripts/setup_env.sh`: succeeds, `real 1m18.259s` — installs pinned
  `requirements-lock.txt` (including the `cu124` torch wheel, ~770 MB) into
  a brand-new `.venv` with no network access to anything beyond PyPI/the
  torch index, and `pip install -e . --no-deps` builds the `forgelm`
  package from the clone.
- `ruff check src tests`: `All checks passed!`
- `ruff format --check src tests`: `69 files already formatted`
- `pyright`: `0 errors, 0 warnings, 0 informations`
- `pytest -v`: **`269 passed, 10 warnings in 64.29s (0:01:04)`** — same 269
  tests, same 10 expected CuBLAS-determinism warnings, as the in-place run
  above; zero failures, zero skips (GPU visible in this clone too).
- The Quickstart's small run, executed step by step from the fresh clone
  (`forgelm smoke` ×2, `forgelm tokenizer train`/`encode`/`decode`,
  `forgelm dataset build` (flag-driven and config-only), `forgelm train
  --config configs/train_toy.yaml`, `forgelm generate`, `forgelm
  evaluate`, `forgelm sample-report`): every command succeeded, and the
  train → generate → evaluate output **matched the committed
  `benchmarks/sample_generations/toy_sample_report.json` byte-for-byte** on
  every numeric field (`loss=5.512617644900151`,
  `perplexity=247.79892844204278`, `tokens_evaluated=537280`,
  `num_batches=1050`) and on the generated text itself
  (`"Alice was beginning toas_ asath"hanthewasbeptoandwasased..."`) —
  concrete proof this project's determinism guarantees (above) hold across
  a genuinely fresh clone, fresh venv, and fresh CPython process, not just
  within one long-lived working tree.
- `forgelm train --config configs/train_toy.yaml` (the toy demo,
  `device: cpu` per that config) took **`real 0m2.375s`** wall-clock for
  the whole CLI process on this machine. **This corrects an inaccurate
  claim this page's earlier draft (and README.md/CHANGELOG.md/ADR 0011)
  inherited from an earlier phase** — "~40s on an RTX 4090" — which was
  wrong on two counts: the config sets `device: cpu` explicitly (it never
  touches the GPU at all), and the actual measured wall-clock is ~17x
  faster than claimed. Fixed at the same commit as this fresh-clone
  verification; see `README.md`'s Quickstart and "Sample generations"
  sections and ADR 0011 decision 2, which now read "~2.4s, CPU-only".
- `forgelm benchmark run` was intentionally **not** re-executed as part of
  this verification pass (it would contend for the shared RTX 4090 with
  other portfolio projects' GPU-benchmark-locked runs for no benefit —
  Phase 5's benchmark numbers are already committed, measured evidence
  that doesn't need re-measuring for a fresh-clone *smoke* check); every
  other Quickstart command was run for real.
- As a direct check on ADR 0011 decision 2's claim (the committed
  `benchmarks/sample_generations/toy_sample_report.{json,md}` is real,
  reproducible output, not hand-edited text): from this same fresh clone,
  `forgelm sample-report --config configs/sample_report_portfolio.yaml`
  — the exact documented regeneration command — was run against this
  clone's own freshly-trained toy checkpoint, and a `diff` of its output
  against the files committed in the main working tree came back **empty
  on both files** (`MARKDOWN: byte-identical`, `JSON: byte-identical`).
  Same seeds, same checkpoint-training recipe, same model, independently
  reproduced from a different clone → byte-identical output.

This confirms the spec's Phase 6 exit criterion ("Fresh clone instructions
verified") and Definition of Done ("Fresh clone installs and passes tests;
small training run reproducible") from an actual isolated clone, not by
inspection of this repository's own working tree.

## Reproducibility risks and mitigations

- **Unpinned versions** → `requirements-lock.txt` is committed and is what
  `scripts/setup_env.sh` installs from by default (`pip install -r`, not a
  fresh range resolution); regenerate deliberately via `--update-lock` and
  re-commit whenever `pyproject.toml` dependencies change.
- **Network dependency in tests** → none: the example corpus
  (`examples/alice_in_wonderland.txt`) is committed directly, so tokenizer
  training and dataset-pipeline tests run fully offline (see ADR 0003).
- **Hash randomization** → `PYTHONHASHSEED` is set by `set_seed()`, though
  this only affects subprocesses spawned after the call, not hashing
  already performed in the current interpreter — noted as a known
  limitation rather than a false guarantee.
