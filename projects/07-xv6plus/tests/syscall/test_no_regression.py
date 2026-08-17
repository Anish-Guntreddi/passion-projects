"""Phase 1 exit criterion: "per-process tracing works without breaking
normal execution." This test is the negative half of that claim: with
trace() never called by anyone, the kernel behaves exactly like stock
xv6 -- no trace output leaks in, ordinary filesystem/shell workflows
work, and nothing panics or hits the "unknown sys call" fallback (which
would indicate the new SYS_trace entry corrupted the dispatch table).

Invariants exercised: #8 (observability code must not destabilize the
kernel it observes -- here, by simply not being invoked at all).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

TRACE_LINE = re.compile(r"\d+: syscall \w+ -> -?\d+")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        q.run_cmd("mkdir xv6plus_test_dir", prompt_timeout=10.0)
        q.run_cmd("echo regression_marker > xv6plus_test_dir/f", prompt_timeout=10.0)
        cat_out = q.run_cmd("cat xv6plus_test_dir/f", prompt_timeout=10.0)
        ls_out = q.run_cmd("ls xv6plus_test_dir", prompt_timeout=10.0)
        q.run_cmd("rm xv6plus_test_dir/f", prompt_timeout=10.0)
        full = q.output()

    assert "regression_marker" in cat_out, (
        f"cat did not show content written by a plain shell redirect; got:\n{cat_out}"
    )
    assert "f" in ls_out, f"ls did not list the file just created; got:\n{ls_out}"
    assert "unknown sys call" not in full, (
        f"kernel hit the unknown-syscall fallback during a plain session "
        f"(dispatch table regression?):\n{full}"
    )
    assert not TRACE_LINE.search(full), (
        f"trace output appeared even though trace() was never called by "
        f"anyone in this session:\n{full}"
    )
    assert "panic" not in full.lower(), f"kernel panicked during a plain regression run:\n{full}"
