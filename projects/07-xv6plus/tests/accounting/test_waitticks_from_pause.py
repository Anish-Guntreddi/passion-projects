"""Phase 2 / FR2: waitticks stays at 0 until a process actually blocks
(sleep()s), then advances by roughly the requested duration after
pause(n) -- exercising kernel/proc.c's sleep(), which adds
(ticks - sleep_start) to p->waitticks on wakeup (ADR-0007: tick
granularity; docs/decisions/0009-accounting-counter-locking.md for
why sleep_start is snapshotted unlocked).

Drives user/accttest.c, which reports waitticks at three stages:
right after main() starts, after a getpid() loop (getpid() never
blocks), and after pause(PAUSE_TICKS=5).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _acct_helpers import parse_accttest_reports, require_stages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

PAUSE_TICKS = 5  # user/accttest.c's PAUSE_TICKS


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("accttest", prompt_timeout=15.0)

    assert "accttest: done" in out, f"accttest did not complete; got:\n{out}"

    reports = parse_accttest_reports(out)
    start, after_getpids, after_pause = require_stages(
        reports, "start", "after_getpids", "after_pause"
    )

    assert start["waitticks"] == 0, f"waitticks nonzero before any blocking call: {start}"
    assert after_getpids["waitticks"] == 0, (
        f"waitticks nonzero after a pure getpid() loop (getpid() never "
        f"blocks): {after_getpids}"
    )
    assert after_pause["waitticks"] >= PAUSE_TICKS - 1, (
        f"waitticks only advanced to {after_pause['waitticks']} after "
        f"pause({PAUSE_TICKS}), expected roughly {PAUSE_TICKS}; "
        f"reports:\n{reports}"
    )
    assert after_pause["waitticks"] <= PAUSE_TICKS * 4, (
        f"waitticks jumped to {after_pause['waitticks']} after "
        f"pause({PAUSE_TICKS}), far more than expected -- possible runaway "
        f"accounting; reports:\n{reports}"
    )
