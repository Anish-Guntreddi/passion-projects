# ADR 0001: Build tooling and test framework

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: 0 (Build/tooling foundation)

## Context

Phase 0's deliverables (tech stack plan, §Part 2) are: CMake + Ninja with
warnings-as-errors, GoogleTest or Catch2, clang-tidy, ASan/UBSan/TSan, and a
Linux CI that builds, tests, and runs static analysis. Several concrete
choices had to be made that the spec leaves open (it says "GoogleTest or
Catch2", not which one; it doesn't specify exact warning flags or how
sanitizers are wired into the build).

## Decisions

1. **CMake ≥ 3.20 + Ninja**, single top-level `CMakeLists.txt` with an
   `arcserve_core` static library consumed by both the production binary
   (`arcserve_server`) and every test binary, so tests exercise exactly the
   same translation units that ship. Verified locally: CMake 3.28.3, Ninja
   1.11.1 (WSL2 Ubuntu 24.04).

2. **GoogleTest**, fetched via `FetchContent` pinned to tag `v1.15.2`, over
   Catch2. Rationale: GoogleTest's death-test and typed-test facilities are
   more useful later for the concurrency/failure test categories in §1.6
   (worker shutdown, TSan-compatible tests), and it's the framework this
   author has the most fluency reviewing failures in quickly. `FetchContent`
   was chosen over a git submodule or vendored copy because it keeps the
   repository free of a second build system's source tree while still
   pinning an exact, reproducible version; it requires network access at
   *configure* time only (verified reachable from the WSL2 build
   environment), not at every build.

3. **Warnings-as-errors**, but a deliberately **narrower** flag set than a
   maximally strict "core guidelines" profile:
   `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
   -Wnull-dereference -Wformat=2 -Wimplicit-fallthrough -Wunused -Werror`.
   Explicitly **excluded**: `-Wconversion`, `-Wsign-conversion`,
   `-Wold-style-cast`, `-Wcast-align`, `-Wdouble-promotion`. Rationale:
   POSIX socket code is inherently full of `int`/`size_t`/`ssize_t`
   crossings (`recv()`/`send()` return `ssize_t`; buffer sizes are
   `size_t`) and `sockaddr*` reinterpret-casts. Enabling those flags at
   Phase 0 would force either pervasive explicit casts that make the
   networking code harder to read for the sake of the linter, or scattered
   per-line suppressions — neither improves correctness at this project's
   scope, and both work against "code stays interview-explainable". The
   flag set can be tightened for a specific module later if a real class of
   bug shows up that stricter conversion warnings would have caught; that
   would be a follow-up ADR, not a silent ratchet.

4. **Sanitizers as CMake options**, not baked into a single build type:
   `ARCSERVE_ENABLE_ASAN` / `ARCSERVE_ENABLE_UBSAN` / `ARCSERVE_ENABLE_TSAN`,
   each an independent configure-time flag (`cmake/Sanitizers.cmake`). ASan
   and UBSan may be combined; TSan is mutually exclusive with ASan (both
   instrument every memory access and cannot coexist in one binary — this
   is enforced with a `message(FATAL_ERROR ...)` at configure time, not
   discovered at link time). `scripts/build.sh <BuildType> <asan|ubsan|
   tsan|none>` drives this from one place so CI and local runs use the same
   invocation.

5. **clang-tidy / clang-format configs are checked in** (`.clang-tidy`,
   `.clang-format`) even though neither binary is installed in the current
   WSL2 dev environment (no passwordless `sudo`, and neither is in the
   toolchain that was pre-verified for this project). The GitHub Actions
   Linux runner (`.github/workflows/ci.yml`) installs both fresh via `apt`
   on every run, so static analysis and format-checking are enforced in CI
   even though they could not be exercised locally during this phase. This
   is recorded as a known gap, not silently glossed over — see the
   project's phase report for the exact commands attempted.

## Consequences

- A fresh clone needs network access once (to fetch GoogleTest at
  configure time) and a C++20 GCC/Clang toolchain; no other external
  dependencies.
- Warning strictness can surprise a contributor used to a maximal
  `cppcoreguidelines`-style profile — this is intentional and explained
  above rather than left implicit in `cmake/CompilerWarnings.cmake`'s
  comments alone.
- Local verification of the clang-tidy/clang-format configuration itself
  (as opposed to the code they'd flag) is deferred to CI until those tools
  are installed in the dev environment.
