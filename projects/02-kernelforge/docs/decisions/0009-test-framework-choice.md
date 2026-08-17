# ADR 0009: Lightweight In-Repo Test Framework Instead of GoogleTest

- **Status:** Accepted
- **Date:** 2026-08-17

## Context
The tech-stack plan allows either GoogleTest *or* "a lightweight
deterministic C++ framework". Pulling GoogleTest in via CMake
`FetchContent` would make a fresh clone's first configure step depend on
network access to GitHub at build time, adds several minutes of extra
build time for a project whose actual test bodies are simple
(allocate → launch → compare against CPU reference), and pulls in a
large dependency tree for a portfolio project meant to stay
interview-explainable end to end (hard constraint 6). Internet access
*is* available in this dev environment (verified), so this is a build-
robustness and simplicity choice, not a hard blocker workaround.

## Decision
Use a small header-only test framework written for this repo:
`src/common/testing.hpp`. It provides:
- `KF_TEST(suite, name) { ... }` registration macro (self-registers into
  a static registry via a constructor trick, no external main-generation
  step needed).
- `KF_EXPECT_TRUE/EQ/NEAR(...)` and `KF_ASSERT_*` (fatal) checks that
  record file/line/expression text on failure.
- A single `kf::testing::run_all(argc, argv)` entry point used by every
  test binary's `main()`, returning the process exit code (0 = all
  passed), with `--filter=substr` support for running one suite.

This is deliberately minimal — it is not a general-purpose framework,
it is scoped to what this repo's correctness tests need (FR6: randomized
vs. reference, edge sizes, tolerances, deterministic seeds).

## Consequences
- Zero external dependencies for the test binaries: `cmake --build`
  works offline on a fresh clone.
- Slightly less tooling than GoogleTest (no death tests, no fixtures with
  inheritance, no built-in parameterized test sweeps) — sweeps over sizes
  are instead done with a plain `for` loop inside one `KF_TEST` body,
  which is simple enough for this project's test bodies.
- If a future contributor prefers GoogleTest, this can be swapped later
  without touching kernel code, since kernel launch wrappers have no
  dependency on the test framework.
