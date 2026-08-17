"""Shared helpers for the Phase 4 (scheduler experiment) test suite.

Named `_sched_helpers` (not `_helpers`) for the same sys.modules-
collision reason `tests/accounting/_acct_helpers.py` isn't named
`_helpers` either -- see that module's docstring for the full
explanation. scripts/run_tests.py runs every discovered
tests/<category>/test_*.py module inside one Python process, and each
category's helper module is imported by its bare name.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Matches lines printed by user/schedbench.c's per-child report (read
# back from that child's own pipe by drain_pipe_to_console(), so these
# never interleave with another child's report -- see that program's
# module comment):
#   schedbench: report pid=<int> tickets=<int> start=<int> end=<int> elapsed=<int> runticks=<int> waitticks=<int> selections=<int>
REPORT_RE = re.compile(
    r"schedbench: report pid=(?P<pid>\d+) tickets=(?P<tickets>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+) elapsed=(?P<elapsed>\d+) "
    r"runticks=(?P<runticks>\d+) waitticks=(?P<waitticks>\d+) "
    r"selections=(?P<selections>\d+)"
)


def parse_schedbench_reports(out: str) -> List[Dict[str, int]]:
    """Parse every schedbench report line in `out`, in the order printed
    (argument/ticket order -- see user/schedbench.c's
    drain_pipe_to_console() loop -- not necessarily completion order)."""
    reports: List[Dict[str, int]] = []
    for m in REPORT_RE.finditer(out):
        reports.append({
            k: int(m.group(k))
            for k in ("pid", "tickets", "start", "end", "elapsed",
                      "runticks", "waitticks", "selections")
        })
    return reports
