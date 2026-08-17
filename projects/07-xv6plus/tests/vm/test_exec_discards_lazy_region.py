"""Phase 5 / FR6, spec Â§1.9 VM benchmark plan: "fork/exec
interactions." A process that sbrklazy()s memory without touching it,
then exec()s into a different program, must not crash or leak --
kexec() (kernel/exec.c) builds the new image in a fresh page table and
only frees the OLD one (uvmfree(), whatever happens to be mapped --
here, nothing in the pending region) after committing to the new
image.

Drives user/vmexectest.c, which exec()s into the stock upstream
`echo` with a marker argument once it has grown-but-not-touched a
lazy region.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmexectest", prompt_timeout=15.0)

        assert "vmexectest: exec failed" not in out, f"exec() into echo failed; got:\n{out}"
        assert "about to exec, never touched the sbrklazy'd region" in out, (
            f"missing pre-exec marker line; got:\n{out}"
        )
        assert "vmexectest_exec_ok" in out, (
            f"missing post-exec marker printed by echo -- exec() did not "
            f"complete cleanly; got:\n{out}"
        )

        # Invariant #8 regression check: normal operation continues.
        regress = q.run_cmd("echo vm_exec_regression_ok", prompt_timeout=10.0)
        assert "vm_exec_regression_ok" in regress, (
            f"shell did not respond normally after vmexectest's exec(); "
            f"got:\n{regress}"
        )
