"""Phase 2 / FR2, invariant #3: a fork()ed child's accounting counters
start fresh at 0, they are not copied from the parent's already-
accumulated totals -- the opposite choice from Phase 1's trace_mask,
which *is* inherited across fork(). See kernel/proc.c's kfork() and
docs/accounting.md "fork/exec/exit semantics".

Drives user/acctforktest.c, which runs the parent's own syscall_count
up via 20 getpid() calls *before* forking, so a bug that copied the
parent's already-accumulated count into the child (instead of leaving
allocproc()'s zeroed value alone) would be obvious in the child's
reported count.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

PARENT_RE = re.compile(r"acctforktest: parent pid=(\d+) syscalls=(\d+)")
CHILD_RE = re.compile(
    r"acctforktest: child pid=(\d+) syscalls=(\d+) runticks=(\d+) waitticks=(\d+)"
)


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("acctforktest", prompt_timeout=15.0)

    assert "acctforktest: done" in out, f"acctforktest did not complete; got:\n{out}"

    pm = PARENT_RE.search(out)
    cm = CHILD_RE.search(out)
    assert pm, f"missing parent report line; got:\n{out}"
    assert cm, f"missing child report line; got:\n{out}"

    parent_syscalls = int(pm.group(2))
    child_syscalls = int(cm.group(2))
    child_runticks = int(cm.group(3))
    child_waitticks = int(cm.group(4))

    assert parent_syscalls >= 20, (
        f"sanity check failed: parent's own syscall count ({parent_syscalls}) "
        f"should already be well above 20 from its getpid() loop; got:\n{out}"
    )
    assert child_syscalls < parent_syscalls, (
        f"child reported syscalls={child_syscalls}, not less than the "
        f"parent's already-accumulated syscalls={parent_syscalls} -- looks "
        f"like the child's counters were copied from the parent instead of "
        f"starting fresh at fork(); got:\n{out}"
    )
    assert child_waitticks == 0, (
        f"freshly forked child reported waitticks={child_waitticks}, "
        f"expected 0 (it has never called pause()); got:\n{out}"
    )
    assert child_runticks < 10, (
        f"freshly forked child reported implausibly large runticks="
        f"{child_runticks} for a program that has barely started running; "
        f"got:\n{out}"
    )
