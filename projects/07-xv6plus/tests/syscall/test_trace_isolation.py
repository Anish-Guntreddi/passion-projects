"""Phase 1 / FR1: tracing is per-process, not global.

Runs a traced command, then an unrelated untraced command, and checks
that the second command's own output segment contains zero trace
lines even though tracing was very recently active in the same shell
session. This exercises the reset-on-free / reset-on-alloc lifecycle
hooks in kernel/proc.c (allocproc/freeproc): each `sh` command is a
fresh fork+exec, so it must start with trace_mask == 0 regardless of
what the previous command did.

Invariants exercised: #5 (a freed/reused proc-table slot never keeps
showing telemetry from whoever used it last).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import mask_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

TRACE_LINE = re.compile(r"\d+: syscall \w+ -> -?\d+")


def run(project_root: Path) -> None:
    mask = mask_for(project_root, "write")

    with QemuSession(project_root) as q:
        q.boot_to_shell()

        traced_out = q.run_cmd(f"xtrace {mask} echo traced_marker", prompt_timeout=10.0)
        assert TRACE_LINE.search(traced_out), (
            f"sanity check failed: expected the traced command itself to "
            f"produce trace output; got:\n{traced_out}"
        )

        untraced_out = q.run_cmd("echo untraced_marker", prompt_timeout=10.0)

    assert "untraced_marker" in untraced_out, (
        f"plain echo run immediately after a traced command produced no "
        f"output; got:\n{untraced_out}"
    )
    assert not TRACE_LINE.search(untraced_out), (
        f"a command with no trace() call of its own must not show trace "
        f"output just because a prior command in the same shell was "
        f"traced; got:\n{untraced_out}"
    )
