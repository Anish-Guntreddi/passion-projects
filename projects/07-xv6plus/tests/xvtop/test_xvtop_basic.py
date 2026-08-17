"""Phase 3 / FR4 exit criterion: "Tool visibly reports active
processes and resource data."

Runs `xvtop 2 5` (two refreshes, 5 ticks apart) and checks each
refresh prints the expected column header, includes xvtop's own row
(it is always RUNNING at the moment it takes its own snapshot -- see
docs/xvtop.md), and ends with a correctly-formatted summary line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

HEADER = "PID\tSTATE\tSZ\tRUNTICKS\tWAITTICKS\tSYSCALLS\tNAME"
SUMMARY_RE = re.compile(r"-- (\d+) active process\(es\) --")
XVTOP_ROW_RE = re.compile(r"^\d+\trun\t\d+\t\d+\t\d+\t\d+\txvtop$", re.MULTILINE)


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("xvtop 2 5", prompt_timeout=15.0)

    assert out.count("=== xvtop refresh") == 2, (
        f"expected exactly 2 refresh headers from `xvtop 2 5`; got:\n{out}"
    )
    assert out.count(HEADER) == 2, f"expected the column header once per refresh; got:\n{out}"

    assert XVTOP_ROW_RE.search(out), (
        f"expected xvtop's own row (state 'run') in its own listing; got:\n{out}"
    )

    summaries = SUMMARY_RE.findall(out)
    assert len(summaries) == 2, f"expected one summary line per refresh; got:\n{out}"
    for n in summaries:
        assert int(n) >= 1, f"expected at least 1 active process (xvtop itself); got:\n{out}"
