"""Phase 2 / invariant #5: a proc-table slot reused after a previous
occupant accumulated counters reports fresh (near-zero) counters, not
a continuation of the previous occupant's totals.

freeproc() (kernel/proc.c) resets runticks/waitticks/syscall_count to
0 when a slot returns to UNUSED, exactly like it already does for
Phase 1's trace_mask. This test runs accttest (which drives its own
syscall_count and waitticks up via a getpid() loop and pause()) to
completion -- reaped by the shell's own wait() before the next prompt
returns, per user/sh.c's runcmd()/main() -- then runs accttest again
and checks its very first ("start") report is nowhere near the first
run's final ("after_pause") totals. This holds regardless of which
physical proc-table slot gets reused for the second run: every
UNUSED slot at that point was zeroed by freeproc() when it was freed.

Invariants exercised: #1 (accounting never corrupts process lifecycle
state across slot reuse), #5 (a reused slot never shows a previous
occupant's telemetry).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _acct_helpers import parse_accttest_reports, require_stages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()

        out1 = q.run_cmd("accttest", prompt_timeout=15.0)
        assert "accttest: done" in out1, f"first accttest run did not complete; got:\n{out1}"
        reports1 = parse_accttest_reports(out1)
        (final1,) = require_stages(reports1, "after_pause")

        out2 = q.run_cmd("accttest", prompt_timeout=15.0)
        assert "accttest: done" in out2, f"second accttest run did not complete; got:\n{out2}"
        reports2 = parse_accttest_reports(out2)
        (start2,) = require_stages(reports2, "start")

    assert start2["syscalls"] < final1["syscalls"], (
        f"second accttest run's very first report already shows "
        f"syscalls={start2['syscalls']}, not less than the first run's "
        f"final syscalls={final1['syscalls']} -- looks like the reused "
        f"proc-table slot kept counting from the previous occupant instead "
        f"of resetting on free"
    )
    assert start2["waitticks"] == 0, (
        f"second accttest run's very first report shows "
        f"waitticks={start2['waitticks']} before it has ever paused -- "
        f"expected 0 (fresh slot); reports:\n{reports2}"
    )
