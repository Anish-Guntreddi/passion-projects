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

Every CLI command that needs reproducibility (currently: `forgelm smoke`)
calls `set_seed()` before doing anything else. Training (Phase 3) will call
it once at the start of a run and additionally persist the RNG state in
checkpoints (D7, not yet implemented).

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

## How to reproduce the test run reported for Phases 0–1

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
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/guntr/Documents/passion-projects/projects/01-forgelm && source .venv/bin/activate && pytest -v'
```

Last real result (2026-08-17, environment above, GPU tests included since
an RTX 4090 was visible):

```
91 passed, 2 warnings in 132.04s (0:02:12)
```

The 2 warnings are both the expected `torch` notice that CuBLAS matmul
doesn't have a fully deterministic kernel without setting
`CUBLAS_WORKSPACE_CONFIG` (triggered by the GPU smoke tests) -- exactly why
`test_smoke_gpu.py` compares checksums with a tolerance instead of exact
equality rather than treating it as a bug to silence.

`ruff check`, `ruff format --check`, and `pyright` all report clean (0
errors) on this same commit of the source.

Also verified with CUDA hidden (`CUDA_VISIBLE_DEVICES="" pytest -v`), to
confirm the D3 "CPU path must work for tests" guarantee independently of
this machine's GPU and to mirror what the GitHub Actions runner (no GPU)
will see:

```
89 passed, 2 skipped in 113.88s (0:01:53)
```

(the 2 skips are exactly `test_smoke_gpu.py`'s two `requires_cuda`-marked
tests, skipped cleanly rather than failing.)

## Reproducibility risks (Phase 0/1 scope) and mitigations

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
