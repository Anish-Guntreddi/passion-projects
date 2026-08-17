"""Phase 1 / FR1: `xtrace MASK cmd...` enables per-process tracing and
the kernel prints "pid: syscall NAME -> RET" for every syscall in the
mask, without corrupting the traced program's own output.

Covers: docs/tracing.md "basic enable + capture" scenario.
Invariants exercised: #3 (telemetry set up correctly across exec via
xtrace's own fork+trace()+exec sequence), #7 (trace() only touches an
int argument, no unsafe pointer use), #8 (tracing doesn't destabilize
a completely ordinary `echo`).
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
    mask = mask_for(project_root, "exec", "write")

    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd(f"xtrace {mask} echo hi", prompt_timeout=10.0)

    assert "hi" in out, f"xtrace echo hi: expected program output missing; got:\n{out}"

    assert re.search(r"\d+: syscall exec -> \d+", out), (
        f"expected a traced exec line (number/name/return); got:\n{out}"
    )

    write_lines = re.findall(r"\d+: syscall write -> \d+", out)
    assert len(write_lines) >= 1, (
        f"expected at least one traced write line (echo writes its argument "
        f"and a newline); got:\n{out}"
    )
