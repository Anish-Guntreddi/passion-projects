"""Phase 1 / FR1 + FR7-style stress smoke check: tracing survives
fork(), including under proc-table-filling load, without corrupting
process lifecycle state.

`forktest` (stock upstream user/forktest.c) forks repeatedly until the
process table is full, then reaps everyone and reports "fork test OK".
Running it under `xtrace <fork mask>` should produce one traced fork
line per fork() call (dozens, given NPROC=64) and forktest must still
finish cleanly -- i.e. our proc.c changes (trace_mask copy in kfork(),
reset in freeproc()) don't break allocation/reap under pressure.

Invariants exercised: #1 (accounting never corrupts process lifecycle:
forktest's own pass/fail logic is the oracle here), #3 (trace_mask is
copied to every child, not just the first), #4 (scheduler keeps making
progress -- forktest completes -- even with the process table nearly
full and every process traced).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import mask_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    mask = mask_for(project_root, "fork")

    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd(f"xtrace {mask} forktest", prompt_timeout=30.0)

    assert "fork test OK" in out, (
        f"forktest did not report success while traced; got:\n{out}"
    )

    fork_lines = re.findall(r"\d+: syscall fork -> -?\d+", out)
    assert len(fork_lines) >= 5, (
        f"expected several traced fork() calls from forktest's "
        f"proc-table-filling loop, got {len(fork_lines)}; output:\n{out}"
    )
