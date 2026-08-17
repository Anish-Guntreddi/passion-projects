"""Phase 5 / FR6, spec Â§1.9 VM benchmark plan: "memory exhaustion
where practical" and "page-accounting cleanup on exit." Repeatedly
sbrklazy()+touch one page at a time until physical memory is
genuinely exhausted (kalloc() returns 0 inside vmfault(),
kernel/vm.c) -- the child is killed cleanly (kexit(-1), same path
test_oob_access_killed.py exercises for a different vmfault() failure
mode) -- then proves the pages it *did* grab were fully reclaimed on
reap (freeproc() -> proc_freepagetable() -> uvmfree()) by successfully
repeating a modest allocate-and-touch afterward.

Runs under the harness's ordinary QEMU session, not a deliberately
smaller one: kernel/memlayout.h fixes PHYSTOP at KERNBASE + 128MB
regardless of QEMU's -m flag (kernel/kalloc.c's kinit() hands
freerange() exactly that range at boot), so shrinking -m below what
this kernel already assumes about physical RAM is not a safety
margin, it is a different and unsupported configuration for this
exact kernel image -- see user/vmexhausttest.c's module comment and
docs/vm-extension.md "memory-exhaustion test configuration."

Drives user/vmexhausttest.c. Real captured run (docs/scheduler.md and
docs/vm-extension.md): exhaustion hit around page ~32000-34000
(~128-136MB nominal), well under vmexhausttest.c's PAGE_BUDGET=40000,
in about 1.3s.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

CHILD_RE = re.compile(r"vmexhausttest: child pid=(\d+) xstatus=(-?\d+)")
RECOVERY_RE = re.compile(r"vmexhausttest: recovery (ok|FAILED)")
RECOVERY_XSTATUS_RE = re.compile(r"vmexhausttest: recovery xstatus=(-?\d+)")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmexhausttest", prompt_timeout=60.0)

        assert "vmexhausttest: done" in out, (
            f"vmexhausttest did not complete -- memory exhaustion may have "
            f"destabilized the kernel instead of being handled cleanly; "
            f"got:\n{out}"
        )

        child_m = CHILD_RE.search(out)
        assert child_m, f"missing child result line; got:\n{out}"
        assert int(child_m.group(2)) == -1, (
            f"exhausting child's exit status was {child_m.group(2)}, expected "
            f"-1 (killed via the same recognized-but-unserviceable page-fault "
            f"path as test_oob_access_killed.py, this time due to kalloc() "
            f"returning 0 rather than an out-of-bounds address); got:\n{out}"
        )
        assert "reached PAGE_BUDGET" not in out, (
            f"the child ran to its PAGE_BUDGET without ever hitting real "
            f"physical-memory exhaustion -- either PAGE_BUDGET needs raising "
            f"or something is wrong with this build's available RAM; "
            f"got:\n{out}"
        )

        recovery_m = RECOVERY_RE.search(out)
        assert recovery_m, f"missing recovery result line; got:\n{out}"
        assert recovery_m.group(1) == "ok", (
            f"recovery allocation FAILED after the exhausting child was "
            f"reaped -- its pages were not fully reclaimed (page-accounting "
            f"cleanup on exit is broken); got:\n{out}"
        )

        recovery_x_m = RECOVERY_XSTATUS_RE.search(out)
        assert recovery_x_m, f"missing recovery xstatus line; got:\n{out}"
        assert int(recovery_x_m.group(1)) == 0, (
            f"recovery process exited with status {recovery_x_m.group(1)}, "
            f"expected 0 (clean success); got:\n{out}"
        )

        # Invariant #8: the kernel must still be fully healthy after a
        # real memory-exhaustion event, not just able to run one more
        # tiny recovery allocation.
        regress = q.run_cmd("echo vm_exhaustion_regression_ok", prompt_timeout=10.0)
        assert "vm_exhaustion_regression_ok" in regress, (
            f"shell did not respond normally after memory exhaustion and "
            f"recovery; got:\n{regress}"
        )
