"""Phase 3, invariant #5 regression test: "Zombie/free process slots
never remain visible as active telemetry." xvtop's collect()
(user/xvtop.c) must skip ZOMBIE slots, not just UNUSED ones -- ZOMBIE
is a real, non-UNUSED procstate, so it needs its own explicit check.

Drives user/xvtopzombie.c, a small deterministic support program: it
forks a child that exits immediately and is never wait()ed on (a
genuine zombie), polls xvstat(2) for that exact pid until it observes
XV_ZOMBIE (removing any race against the scheduler or a fixed pause),
prints that pid, then exec()s straight into `xvtop 1`. kexec() replaces
only the program image, not the pid or its children, so the zombie is
still there for xvtop's first refresh -- and must not appear as a row.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

ZOMBIE_PID_RE = re.compile(r"xvtopzombie: child pid=(\d+) is zombie, execing into xvtop")
ZOMBIE_ROW_RE = re.compile(r"^\d+\tzombie\t", re.MULTILINE)
HEADER = "PID\tSTATE\tSZ\tRUNTICKS\tWAITTICKS\tSYSCALLS\tNAME"


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("xvtopzombie", prompt_timeout=20.0)

    assert "xvtopzombie: exec failed" not in out, f"exec() into xvtop failed; got:\n{out}"

    m = ZOMBIE_PID_RE.search(out)
    assert m, (
        f"xvtopzombie never confirmed its child reached zombie state "
        f"(support-program failure, not the invariant under test); got:\n{out}"
    )
    zpid = m.group(1)

    assert HEADER in out, f"xvtop did not print its column header; got:\n{out}"

    assert not ZOMBIE_ROW_RE.search(out), (
        f"xvtop printed a row with state 'zombie' -- invariant #5 "
        f"violated, a zombie slot is visible as active telemetry; "
        f"got:\n{out}"
    )
    assert not re.search(rf"(?m)^{zpid}\t", out), (
        f"the known zombie pid {zpid} appeared as an xvtop row; got:\n{out}"
    )
