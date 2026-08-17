"""Phase 2 / FR3, invariant #7: xvstat(2) validates its arguments --
an out-of-range index and an invalid user pointer are rejected (-1),
not followed, and the syscall keeps working normally afterwards.

Drives user/xvstatbounds.c, which exercises all four cases in one
deterministic transcript: idx=-1, idx=NPROC, a bad user pointer with a
valid idx, then an ordinary idx=0 call that must still succeed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("xvstatbounds", prompt_timeout=15.0)

    assert "xvstatbounds: done" in out, (
        f"xvstatbounds did not complete -- a bad xvstat() call may have "
        f"destabilized the process instead of just returning -1; got:\n{out}"
    )

    def result_of(label: str) -> int:
        m = re.search(rf"xvstatbounds: {re.escape(label)} -> (-?\d+)", out)
        assert m, f"missing '{label}' result line; got:\n{out}"
        return int(m.group(1))

    assert result_of("idx=-1") == -1, f"xvstat(-1, ...) should be rejected; got:\n{out}"
    assert result_of("idx=NPROC") == -1, f"xvstat(NPROC, ...) should be rejected; got:\n{out}"
    assert result_of("bad addr") == -1, (
        f"xvstat(0, <bad user pointer>) should be rejected; got:\n{out}"
    )
    assert result_of("idx=0") == 0, f"xvstat(0, <valid pointer>) should succeed; got:\n{out}"
