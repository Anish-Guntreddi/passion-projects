#!/usr/bin/env python3
"""xv6-plus Phase 7 observability-report capture (spec Part 3 roadmap
row 7: "Use tracing/xvtop to explain an end-to-end workload").

Boots one QEMU session and drives it through a fixed end-to-end
workload -- process creation via `exec()`, syscalls via `xtrace`,
scheduling via a backgrounded `schedbench` run sampled live through
`xvtop`, and memory via `vmfaulttest`'s pagefault telemetry -- writing
the full, unedited console transcript to
benchmarks/raw/observability/end_to_end_session.txt. Same role and
same raw-artifact-only philosophy as scripts/benchmark.py plays for
benchmarks/raw/scheduler/: this script produces the raw transcript
only; every number quoted in docs/observability-report.md is copied
directly from the file this script writes, not computed here and not
hand-typed there.

Sequencing note (see the "drain, don't send a second command" comment
below): schedbench is backgrounded (`&`) so xvtop can sample it
mid-run, but its own completion report is not synchronized with sh's
prompt -- backgrounding returns "$ " immediately, and schedbench's own
`printf()` calls (parent-collected, after `wait()`-ing for every
child) can still be in flight when a *second* foreground command's
output would otherwise race it on the shared console UART (no
per-line write atomicity -- the same hazard user/schedbench.c's own
module comment documents, and the reason every Phase 6 stress program
routes worker output through private pipes instead of the shared
console). This script avoids that race the same way any of those
programs do it in-kernel-image: give the racing writer(s) a bounded
window to finish *before* issuing the next command, rather than after.

Usage:
  scripts/run-tests.sh --skip-build   # build first if needed
  python3 scripts/capture_observability.py
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qemu_session import QemuSession  # noqa: E402

OUT_DIR = PROJECT_ROOT / "benchmarks" / "raw" / "observability"
OUT_PATH = OUT_DIR / "end_to_end_session.txt"

# (1<<SYS_exec)|(1<<SYS_open)|(1<<SYS_read)|(1<<SYS_write)|(1<<SYS_close)
# (kernel/syscall.h: exec=7, open=15, read=5, write=16, close=21)
TRACE_MASK = (1 << 7) | (1 << 15) | (1 << 5) | (1 << 16) | (1 << 21)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    sections: list[str] = []

    def log(tag: str, text: str) -> None:
        sections.append(f"===== {tag} =====\n{text}\n")
        print(f"[observability] {tag}")

    with QemuSession(PROJECT_ROOT) as q:
        q.boot_to_shell()
        log("boot", q.output())

        log("xvtop baseline (before workload)",
            q.run_cmd("xvtop 1 5", prompt_timeout=15.0))

        log(f"xtrace {TRACE_MASK} cat README  (creation via exec + file syscalls)",
            q.run_cmd(f"xtrace {TRACE_MASK} cat README", prompt_timeout=15.0))

        log("schedbench 1 60 20 10 5 &  (backgrounded scheduling workload)",
            q.run_cmd("schedbench 1 60 20 10 5 &", prompt_timeout=10.0))

        # count*period well under schedbench's TARGET_TICKS=30 window
        # (user/schedbench.c) so this command's own "$ " returns
        # before schedbench can finish and start writing to the
        # console too (see module docstring).
        log("xvtop 3 6  (live snapshots while schedbench runs, mid-workload)",
            q.run_cmd("xvtop 3 6", prompt_timeout=20.0))

        # Drain (send nothing -- draining isn't a second writer, so it
        # can't race schedbench's own output) until its backgrounded
        # report is fully flushed.
        drain_start = len(q._buf)
        q.drain(quiet_for=1.5, timeout=15.0)
        log("drain -- schedbench's own completion report surfaces here",
            q._buf[drain_start:].decode("utf-8", "replace"))

        log("vmfaulttest  (memory: pagefault telemetry across sbrklazy/touch/retouch)",
            q.run_cmd("vmfaulttest", prompt_timeout=15.0))

        log("xvtop final (after workload -- schedbench children reaped)",
            q.run_cmd("xvtop 1 5", prompt_timeout=15.0))

        log("regression check",
            q.run_cmd("echo observability_regression_ok", prompt_timeout=10.0))

    dt = time.monotonic() - t_start
    with open(OUT_PATH, "w", newline="\n") as f:
        f.write(f"# command: scripts/capture_observability.py\n")
        f.write(f"# captured: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        f.write(f"# wall_elapsed_seconds: {dt:.2f}\n")
        f.write(f"# trace_mask: {TRACE_MASK} = (1<<SYS_exec)|(1<<SYS_open)|(1<<SYS_read)|(1<<SYS_write)|(1<<SYS_close)\n")
        f.write("\n".join(sections))

    print(f"[observability] wrote {OUT_PATH.relative_to(PROJECT_ROOT)} (elapsed {dt:.2f}s)")
    print("[observability] see docs/observability-report.md for the narrative built from this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
