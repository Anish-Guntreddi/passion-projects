"""Phase 5 / FR6, spec Â§1.9 VM benchmark plan: "fork/exec
interactions." A parent that sbrklazy()s memory without touching it
has nothing mapped there for uvmcopy() (kernel/vm.c) to copy into the
child on fork(); the child's own first touch in that region must be
serviced by the child's OWN independent vmfault() call (pagefaults
goes to 1 for the child), and the parent's own pagefaults counter must
stay untouched by the child's fault (separate struct proc, separate
counters, invariant #1: accounting never corrupts process lifecycle
state across process boundaries).

Drives user/vmforktest.c.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

CHILD_RE = re.compile(r"vmforktest: child pid=(\d+) pagefaults=(\d+)")
PARENT_RE = re.compile(r"vmforktest: parent pid=(\d+) pagefaults=(\d+)")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmforktest", prompt_timeout=15.0)

    assert "vmforktest: done" in out, f"vmforktest did not complete; got:\n{out}"

    child_m = CHILD_RE.search(out)
    parent_m = PARENT_RE.search(out)
    assert child_m, f"missing child report line; got:\n{out}"
    assert parent_m, f"missing parent report line; got:\n{out}"

    assert int(child_m.group(2)) == 1, (
        f"child reported pagefaults={child_m.group(2)}, expected exactly 1 "
        f"(its own independent fault on the inherited-but-unmapped lazy "
        f"region); got:\n{out}"
    )
    assert int(parent_m.group(2)) == 0, (
        f"parent reported pagefaults={parent_m.group(2)} after its child "
        f"faulted in a page, expected 0 -- the child's fault must not be "
        f"attributed to the parent's own counter; got:\n{out}"
    )
