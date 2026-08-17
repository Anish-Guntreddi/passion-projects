#!/usr/bin/env python3
"""xv6-plus Phase 4 scheduler benchmark capture (spec Part 2: "Benchmark
analysis | Python (scripts/benchmark.py) + plots").

Boots one QEMU session, runs each of a fixed set of `schedbench
policy tickets...` commands (user/schedbench.c) in turn, and writes
each run's raw console output verbatim to benchmarks/raw/scheduler/ --
these are the source-of-truth artifacts benchmarks/methodology.md's
tables are computed from. This script produces raw data only; it does
not compute or print any derived statistic itself, on purpose -- see
methodology.md's own "Reproducing this data" section for the parsing
snippet that turns these files into the numbers actually claimed in
that document, kept separate so a stale claim in methodology.md is
easy to catch by re-running both.

Usage:
  scripts/run-tests.sh --skip-build   # build first if needed
  python3 scripts/benchmark.py        # capture into benchmarks/raw/scheduler/
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qemu_session import QemuSession  # noqa: E402

OUT_DIR = PROJECT_ROOT / "benchmarks" / "raw" / "scheduler"

# (name, shell command). Kept in sync by hand with
# benchmarks/methodology.md's "Run N" sections -- if you add a run
# here, add its table there too, computed from the new raw file, not
# copied from an old one.
RUNS = [
    ("baseline_unequal_tickets", "schedbench 0 60 20 10 5 60 20 10"),
    ("lottery_unequal_tickets", "schedbench 1 60 20 10 5 60 20 10"),
    ("lottery_zero_ticket_floor", "schedbench 1 0 0 30 30 30 0 0"),
    ("lottery_equal_tickets", "schedbench 1 10 10 10 10 10 10 10"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with QemuSession(PROJECT_ROOT) as q:
        q.boot_to_shell(timeout=25.0)
        for name, cmd in RUNS:
            print(f"[benchmark] {name}: {cmd}")
            t0 = time.monotonic()
            out = q.run_cmd(cmd, prompt_timeout=45.0)
            dt = time.monotonic() - t0
            assert "schedbench: done" in out, (
                f"{name} did not complete cleanly; got:\n{out}"
            )
            path = OUT_DIR / f"{name}.txt"
            with open(path, "w", newline="\n") as f:
                f.write(f"# command: {cmd}\n")
                f.write(f"# captured: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
                f.write(f"# wall_elapsed_seconds: {dt:.2f}\n")
                f.write(out)
            print(f"[benchmark]   wrote {path.relative_to(PROJECT_ROOT)} (elapsed {dt:.2f}s)")

    print("[benchmark] done -- see benchmarks/methodology.md to recompute its tables from these files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
