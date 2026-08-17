"""Phase 1 / FR1: trace() genuinely toggles, for a single still-running
process, not just "tracing until the next exec".

Drives user/tracetest.c, which calls trace(1<<SYS_getpid), makes one
getpid() call, calls trace(0), then makes a second getpid() call.
Exactly the first getpid() should produce a trace line.

Invariants exercised: #7 (trace()'s only argument is a plain int --
no user pointer is read, so there is nothing to validate/misvalidate
here by design; see docs/tracing.md "safe scope").
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

TRACE_GETPID = re.compile(r"\d+: syscall getpid -> \d+")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("tracetest", prompt_timeout=10.0)

    assert "tracetest: done" in out, f"tracetest did not complete; got:\n{out}"

    try:
        enabling_idx = out.index("tracetest: enabling")
        disabling_idx = out.index("tracetest: disabling")
        done_idx = out.index("tracetest: done")
    except ValueError as e:
        raise AssertionError(f"tracetest output missing an expected marker: {e}\ngot:\n{out}")

    enabled_segment = out[enabling_idx:disabling_idx]
    disabled_segment = out[disabling_idx:done_idx]

    assert TRACE_GETPID.search(enabled_segment), (
        f"expected a traced getpid() call while tracing was enabled; "
        f"got:\n{enabled_segment}"
    )
    assert not TRACE_GETPID.search(disabled_segment), (
        f"trace(0) should disable tracing for this same process, but a "
        f"trace line still appeared after it; got:\n{disabled_segment}"
    )
