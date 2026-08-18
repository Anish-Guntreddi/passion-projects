"""Phase 6 / spec roadmap row 6 ("syscall-heavy ... stress"). Drives
user/stresssyscalls.c: N_WORKERS (4) processes hammering the syscall
dispatch path (kernel/syscall.c: syscall()) with a tight getpid() loop
simultaneously across every hart, each reporting its own syscall_count
before/after via xvstat(2) over a private pipe.

Invariants exercised: #1 (syscall_count must never lose or corrupt an
update under real concurrent dispatch-path traffic across all harts --
unlike tests/accounting/test_syscall_count_monotonic.py's single
sequential process) and #8 (heavy syscall volume across every hart at
once must not destabilize the kernel).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stress_helpers import SYSCALLS_REPORT_RE, parse_reports  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

N_WORKERS = 4
SYSCALLS_PER_WORKER = 6000


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("stresssyscalls", prompt_timeout=60.0)

        assert "stresssyscalls: done" in out, (
            f"stresssyscalls did not complete -- concurrent syscall-heavy "
            f"load may have hung or crashed the kernel; got:\n{out}"
        )

        reports = parse_reports(SYSCALLS_REPORT_RE, out)
        assert len(reports) == N_WORKERS, (
            f"expected {N_WORKERS} worker reports, got {len(reports)}; "
            f"out:\n{out}"
        )

        pids = [r["pid"] for r in reports]
        assert len(set(pids)) == N_WORKERS, (
            f"worker reports do not all have distinct pids -- possible "
            f"proc-table/pid corruption under concurrent syscall load; "
            f"reports:\n{reports}"
        )

        for r in reports:
            delta = r["after"] - r["before"]
            assert delta >= SYSCALLS_PER_WORKER, (
                f"worker idx={r['idx']} pid={r['pid']}: syscall_count only "
                f"advanced by {delta} across a {SYSCALLS_PER_WORKER}-iteration "
                f"getpid() loop run concurrently with {N_WORKERS - 1} siblings "
                f"-- expected at least {SYSCALLS_PER_WORKER} (invariant #1: "
                f"a lost update under concurrent dispatch-path traffic); "
                f"reports:\n{reports}"
            )

        # Invariant #8: the kernel must still be fully healthy after a
        # syscall-dispatch storm across every hart at once.
        regress = q.run_cmd("echo syscall_stress_regression_ok", prompt_timeout=10.0)
        assert "syscall_stress_regression_ok" in regress, (
            f"shell did not respond normally after the syscall stress run; "
            f"got:\n{regress}"
        )
