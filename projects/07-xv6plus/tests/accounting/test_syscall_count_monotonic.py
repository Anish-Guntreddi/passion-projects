"""Phase 2 / FR2 exit criterion: "counter monotonicity where
appropriate." syscall_count only ever increases within one process's
lifetime -- there is no code path that decrements or resets it except
freeproc() returning the whole slot to UNUSED (covered separately by
test_slot_reuse_reset.py).

Drives user/accttest.c across its three report stages (start ->
after_getpids -> after_pause), each doing strictly more syscalls than
the last.

Invariants exercised: #1 (accounting never corrupts process lifecycle
state -- here, by never going backwards).
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
        out = q.run_cmd("accttest", prompt_timeout=15.0)

    assert "accttest: done" in out, f"accttest did not complete; got:\n{out}"

    reports = parse_accttest_reports(out)
    start, after_getpids, after_pause = require_stages(
        reports, "start", "after_getpids", "after_pause"
    )

    assert start["pid"] == after_getpids["pid"] == after_pause["pid"], (
        f"all three reports should be from the same process; got:\n{reports}"
    )
    assert start["syscalls"] < after_getpids["syscalls"], (
        f"syscall_count did not increase across a 5-iteration getpid() "
        f"loop: start={start['syscalls']}, after_getpids={after_getpids['syscalls']}"
    )
    assert after_getpids["syscalls"] < after_pause["syscalls"], (
        f"syscall_count did not increase across pause(): "
        f"after_getpids={after_getpids['syscalls']}, "
        f"after_pause={after_pause['syscalls']}"
    )
