"""Phase 5 / FR6, invariant #6 (VM changes preserve user/kernel
isolation and mapping permissions) and invariant #8 (observability/
extension code must not destabilize the kernel it observes). Spec
Â§1.9 VM benchmark plan: "invalid address/permission cases."

A process dereferencing a virtual address far beyond its own p->sz
must be killed cleanly (vmfault()'s `va >= p->sz` check,
kernel/vm.c -> kernel/trap.c's usertrap() -> kexit(-1)) -- not panic
the kernel, not hang, and not leave the kernel unable to service the
next command normally right after.

Drives user/vmoobtest.c.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

RESULT_RE = re.compile(r"vmoobtest: child pid=(\d+) wait_pid=(\d+) xstatus=(-?\d+)")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmoobtest", prompt_timeout=15.0)

        assert "vmoobtest: done" in out, (
            f"vmoobtest did not complete -- the out-of-bounds access may have "
            f"hung or otherwise destabilized the kernel; got:\n{out}"
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
        regress = q.run_cmd("echo vm_oob_regression_ok", prompt_timeout=10.0)
        assert "vm_oob_regression_ok" in regress, (
            f"shell did not respond normally after the out-of-bounds access "
            f"was serviced; got:\n{regress}"
        )
