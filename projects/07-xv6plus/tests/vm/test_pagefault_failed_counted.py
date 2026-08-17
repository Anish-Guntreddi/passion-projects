"""Phase 5 / FR6: direct verification that vmfault()'s failure path
actually increments p->pagefaults_failed -- added during review
followup. Every other Phase 5 test that drives vmfault()'s failure
path (test_oob_access_killed.py, test_memory_exhaustion_recovery.py,
test_permission_fault_killed.py) only ever *infers* the counter
incremented from the faulting process's own exit status (xstatus ==
-1); none of them ever reads the counter's actual value, because the
faulting process there is killed by the very fault it triggered and
never survives to read it.

user/vmpfailcount.c instead triggers vmfault()'s failure path
through xvstat(2) itself (a deliberately out-of-bounds output
pointer): a failing copyout() inside sys_xvstat() just returns -1 to
the syscall, it does not kill the calling process, so that process
reads its own pagefaults_failed both before and after and this test
asserts the exact +1 difference directly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

RESULT_RE = re.compile(
    r"vmpfailcount: pagefaults_failed before=(\d+) after=(\d+)"
)


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmpfailcount", prompt_timeout=15.0)

    assert "vmpfailcount: done" in out, (
        f"vmpfailcount did not complete; got:\n{out}"
    )
    assert "xvstat(0, bad_ptr) -> -1" in out, (
        f"the deliberately out-of-bounds xvstat(2) call should have failed "
        f"(-1), not succeeded; got:\n{out}"
    )

    m = RESULT_RE.search(out)
    assert m, f"missing pagefaults_failed before/after line; got:\n{out}"
    before, after = int(m.group(1)), int(m.group(2))
    assert after == before + 1, (
        f"pagefaults_failed went from {before} to {after} -- expected "
        f"exactly +1 from the single deliberately-failing xvstat(2) call "
        f"(a direct counter read, not an inference from a killed "
        f"process's exit status); got:\n{out}"
    )
