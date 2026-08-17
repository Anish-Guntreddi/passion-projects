# ADR 0001: Build tooling and warnings policy

- **Status:** Accepted
- **Phase:** 0 (library foundation)
- **Related spec decisions:** none directly (Part 2 tech-stack plan: CMake +
  Ninja, GoogleTest, ASan/UBSan/TSan, Linux CI)

## Context

Phase 0 needs a build that (a) is reproducible on a fresh clone, (b) fails
loudly on warnings instead of accumulating them, and (c) supports the
sanitizer configurations the spec's test strategy (§1.8) and roadmap Phase 7
("TSan/stress pass") require, without those choices needing to be
re-litigated file by file as the project grows.

## Decision

- **CMake + Ninja**, C++20, `CMAKE_CXX_STANDARD_REQUIRED ON`,
  `CMAKE_CXX_EXTENSIONS OFF`. Matches the spec's tech-stack plan (Part 2)
  directly.
- **Warnings as errors by default** (`PEBBLEDB_WARNINGS_AS_ERRORS`, default
  `ON`), applied uniformly to the library, tools, and tests via an
  `INTERFACE` target (`pebbledb_warnings`, `cmake/CompilerWarnings.cmake`) so
  every consumer gets identical flags rather than each target repeating
  them. Baseline set: `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
  -Woverloaded-virtual -Wnull-dereference -Wformat=2
  -Wimplicit-fallthrough -Wunused` (`/W4 /WX` on MSVC, kept for portability
  even though Linux is the supported target — see ADR 0004).
- **Deliberately excluded:** `-Wconversion`, `-Wsign-conversion`,
  `-Wold-style-cast`. The WAL's byte-level I/O (checksum computation,
  fixed-width length/sequence encoding, POSIX `read()`/`write()`
  `size_t`↔`ssize_t` traffic) trips these constantly without a
  corresponding correctness benefit at this project's scope — every trip is
  a deliberate, bounds-checked narrowing (e.g. `key.size()` into a
  `uint32_t` header field, already guarded by `kMaxKeyLength`/
  `kMaxValueLength` — see `docs/file-format.md` and ADR 0007), not an
  accidental one `-Wconversion` would be catching. This mirrors the
  equivalent decision in the portfolio's ArcServe project (C++/Linux,
  socket and byte-buffer code with the same shape of trade-off).
- **Sanitizers are opt-in, mutually-scoped build variants**
  (`PEBBLEDB_ENABLE_ASAN`, `PEBBLEDB_ENABLE_UBSAN`, `PEBBLEDB_ENABLE_TSAN`),
  selected at configure time via `scripts/build.sh BuildType
  {asan|ubsan|tsan|none}`, each into its own `build/<BuildType>-<variant>`
  directory. ASan and UBSan may be combined; TSan is rejected in
  combination with ASan (`cmake/Sanitizers.cmake` fails the configure) since
  both instrument every memory access and cannot coexist in one binary.
  Sanitizer flags must be compiled into every translation unit including
  GoogleTest itself, which is why the variant is a configure-time choice
  (a separate build directory) rather than a runtime one.
- **GoogleTest** (v1.15.2, pinned tag) is fetched via `FetchContent` rather
  than vendored or system-installed, so a fresh clone builds without any
  pre-installed test framework. `gtest_force_shared_crt` is forced `ON` to
  match the rest of the build's CRT/exception settings.
- **clang-format / clang-tidy configs are checked in** (`.clang-format`,
  `.clang-tidy`) and their driver scripts (`scripts/format.sh`,
  `scripts/lint.sh`) are written to run identically in CI and locally, but
  **neither tool is required to be installed for local development**: this
  WSL2 dev environment does not have clang-format/clang-tidy installed and
  has no passwordless `sudo` to install them ad hoc mid-session. Both
  scripts self-document this in a header comment rather than silently
  failing, and CI (`.github/workflows/ci-pebbledb.yml`) installs both tools
  fresh in a container and runs exactly these scripts — see that workflow
  for what actually exercises `.clang-tidy`'s checks and `.clang-format`'s
  style.
- **`compile_commands.json`** is exported (`CMAKE_EXPORT_COMPILE_COMMANDS
  ON`) unconditionally so clangd and `scripts/lint.sh`'s clang-tidy
  invocation both have a compilation database without a separate step.

## Consequences

- Every warning the compiler can raise (within the excluded set above) is a
  hard build failure locally and in CI, on every configuration — there is
  no warning backlog to accumulate.
- Local development in this WSL2 environment cannot run clang-tidy/
  clang-format directly; contributors rely on CI's `static-analysis` job for
  that signal, or install the tools themselves. Correctness is unaffected —
  the compiler's own warnings-as-errors and the sanitizer builds catch a
  large fraction of what clang-tidy would flag for this kind of low-level
  code, so the local gap is style/idiom coverage, not correctness coverage.
- Adding a new sanitizer-sensitive dependency later must go through
  `FetchContent` (or an equivalent build-from-source path) rather than a
  prebuilt binary, since a prebuilt library would not carry the
  sanitizer's instrumentation and would produce spurious sanitizer reports
  or silently uninstrumented gaps.
