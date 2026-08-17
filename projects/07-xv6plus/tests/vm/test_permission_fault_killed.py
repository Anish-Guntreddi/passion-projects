"""Phase 5 / FR6, invariant #6 (VM changes preserve user/kernel
isolation and mapping permissions). Spec Â§1.9 VM benchmark plan:
"invalid address/permission cases" -- added during review followup,
since test_oob_access_killed.py/user/vmoobtest.c only ever drove the
"invalid address" half of that bullet (va >= p->sz); this is the
"permission" half, previously untested and (per the pre-fix code)
misdiagnosed as "invalid address or out of memory" when it happened.

A process writing through a pointer into its own text segment (a real,
in-bounds, already-mapped address -- just not a writable one, since
kernel/exec.c's flags2perm() never sets PTE_W for text) must be killed
cleanly, with a diagnostic that correctly identifies this as a
permission violation, not an out-of-bounds/out-of-memory failure --
not panic the kernel, not hang, and not leave the kernel unable to
service the next command normally right after.

Drives user/vmpermtest.c.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

RESULT_RE = re.compile(r"vmpermtest: child pid=(\d+) wait_pid=(\d+) xstatus=(-?\d+)")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmpermtest", prompt_timeout=15.0)

        assert "vmpermtest: done" in out, (
            f"vmpermtest did not complete -- the permission violation may "
            f"have hung or otherwise destabilized the kernel; got:\n{out}"
        )

        assert "permission violation" in out, (
            f"expected kernel/trap.c's permission-violation diagnostic (not "
            f"the generic 'invalid address or out of memory' message -- "
            f"this access was in-bounds and already mapped, just not "
            f"writable); got:\n{out}"
        )
        assert "invalid address or out of memory" not in out, (
            f"got the generic out-of-bounds/out-of-memory diagnostic for "
            f"what should be a distinct permission-violation message; "
            f"got:\n{out}"
        )
        assert "write access" in out, (
            f"expected the diagnostic to identify this as a write access "
            f"(a store to a read-only page); got:\n{out}"
        )

        m = RESULT_RE.search(out)
        assert m, f"missing child result line; got:\n{out}"
        pid, wait_pid, xstatus = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert wait_pid == pid, f"parent's wait() reaped the wrong pid; got:\n{out}"
        assert xstatus == -1, (
            f"child's exit status was {xstatus}, expected -1 (kexit(-1) via "
            f"kernel/trap.c's killed-process path); got:\n{out}"
        )

        assert "parent still alive" in out, f"missing parent-alive line; got:\n{out}"

        # Invariant #8: the kernel must keep working normally right after.
        regress = q.run_cmd("echo vm_perm_regression_ok", prompt_timeout=10.0)
        assert "vm_perm_regression_ok" in regress, (
            f"shell did not respond normally after the permission "
            f"violation was serviced; got:\n{regress}"
        )
