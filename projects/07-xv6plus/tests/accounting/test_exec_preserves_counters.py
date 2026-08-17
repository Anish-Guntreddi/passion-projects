"""Phase 2 / FR2, invariant #3: accounting counters survive exec(),
unlike fork() where they deliberately reset (see
test_fork_fresh_counters.py and kernel/proc.c's kfork()).

Drives user/acctexectest.c, which runs PRE_EXEC_GETPID_LOOPS=7
getpid() calls, then exec()s into user/acctreport.c (same struct
proc/pid, only the program image changes), which reports the syscall
count it inherited. kexec() never touches syscall_count/runticks/
waitticks -- if it accidentally reset them, the reported count would
be tiny (just acctreport's own xv_find_self overhead) instead of well
above 7.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

PRE_EXEC_GETPID_LOOPS = 7  # user/acctexectest.c's PRE_EXEC_GETPID_LOOPS

REPORT_RE = re.compile(r"acctreport: pid=(\d+) syscalls=(\d+)")


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("acctexectest", prompt_timeout=15.0)

    assert "acctexectest: about to exec" in out, (
        f"acctexectest did not reach its exec() call; got:\n{out}"
    )
    assert "acctexectest: exec failed" not in out, f"exec() into acctreport failed; got:\n{out}"

    m = REPORT_RE.search(out)
    assert m, f"missing acctreport report line after exec(); got:\n{out}"
    syscalls = int(m.group(2))

    assert syscalls > PRE_EXEC_GETPID_LOOPS, (
        f"acctreport (post-exec) reported syscalls={syscalls}, expected "
        f"strictly more than the {PRE_EXEC_GETPID_LOOPS} getpid() calls "
        f"made before exec() -- looks like exec() reset the syscall "
        f"counter instead of preserving it; got:\n{out}"
    )
