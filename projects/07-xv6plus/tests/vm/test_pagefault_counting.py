"""Phase 5 / FR6: basic page-fault-telemetry correctness. sbrklazy()
alone (growing sz without touching memory) must not count any faults;
touching N distinct pages must count exactly N; re-touching an
already-resident page must not double-count (vmfault()'s ismapped()
short-circuit, kernel/vm.c).

Drives user/vmfaulttest.c.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

REPORT_RE = re.compile(
    r"vmfaulttest: (?P<tag>\w+): pid=(?P<pid>\d+) "
    r"pagefaults=(?P<pagefaults>\d+) pagefaults_failed=(?P<pagefaults_failed>\d+) "
    r"sz=(?P<sz>\d+)"
)

NPAGES = 4  # user/vmfaulttest.c: NPAGES


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("vmfaulttest", prompt_timeout=15.0)

    assert "vmfaulttest: done" in out, f"vmfaulttest did not complete; got:\n{out}"

    reports = {m.group("tag"): {
        "pagefaults": int(m.group("pagefaults")),
        "pagefaults_failed": int(m.group("pagefaults_failed")),
        "sz": int(m.group("sz")),
    } for m in REPORT_RE.finditer(out)}

    for tag in ("start", "after_sbrklazy", "after_touch", "after_retouch"):
        assert tag in reports, f"missing '{tag}' report line; got:\n{out}"

    assert reports["start"]["pagefaults"] == 0, f"nonzero pagefaults before any sbrklazy(); reports:\n{reports}"
    assert reports["start"]["pagefaults_failed"] == 0, f"nonzero pagefaults_failed at start; reports:\n{reports}"

    assert reports["after_sbrklazy"]["pagefaults"] == 0, (
        f"pagefaults={reports['after_sbrklazy']['pagefaults']} right after "
        f"sbrklazy() -- growing sz alone must not fault anything in; "
        f"reports:\n{reports}"
    )
    assert reports["after_sbrklazy"]["sz"] > reports["start"]["sz"], (
        f"sz did not grow after sbrklazy(); reports:\n{reports}"
    )

    assert reports["after_touch"]["pagefaults"] == NPAGES, (
        f"pagefaults={reports['after_touch']['pagefaults']} after touching "
        f"{NPAGES} distinct pages, expected exactly {NPAGES}; reports:\n{reports}"
    )
    assert reports["after_touch"]["pagefaults_failed"] == 0, (
        f"unexpected pagefaults_failed after touching in-bounds pages; "
        f"reports:\n{reports}"
    )

    assert reports["after_retouch"]["pagefaults"] == NPAGES, (
        f"pagefaults changed from {NPAGES} to "
        f"{reports['after_retouch']['pagefaults']} after re-touching an "
        f"already-resident page -- vmfault()'s ismapped() check should have "
        f"prevented a second count; reports:\n{reports}"
    )
