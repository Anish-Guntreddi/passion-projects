"""Shared helpers for the Phase 6 (stress/race hardening) test suite.

Named `_stress_helpers` (not `_helpers`) deliberately -- same
sys.modules-collision reason `tests/accounting/_acct_helpers.py` and
`tests/scheduler/_sched_helpers.py` aren't named `_helpers` either:
scripts/run_tests.py runs every discovered tests/<category>/test_*.py
module inside one Python process, and each category's helper module is
imported by its bare name.
"""
from __future__ import annotations

import re
from typing import Dict, List

# user/stresssyscalls.c:
#   stresssyscalls: report idx=<int> pid=<int> before=<int> after=<int>
SYSCALLS_REPORT_RE = re.compile(
    r"stresssyscalls: report idx=(?P<idx>\d+) pid=(?P<pid>\d+) "
    r"before=(?P<before>\d+) after=(?P<after>\d+)"
)

# user/stressvm.c's clean-phase report:
#   stressvm: clean idx=<int> pid=<int> pagefaults_before=<int>
#   pagefaults_after=<int> failed_before=<int> failed_after=<int>
STRESSVM_CLEAN_RE = re.compile(
    r"stressvm: clean idx=(?P<idx>\d+) pid=(?P<pid>\d+) "
    r"pagefaults_before=(?P<pf_before>\d+) pagefaults_after=(?P<pf_after>\d+) "
    r"failed_before=(?P<failed_before>\d+) failed_after=(?P<failed_after>\d+)"
)

STRESSVM_EXHAUST_REAP_RE = re.compile(
    r"stressvm: exhaust reap wpid=(?P<wpid>\d+) xstatus=(?P<xstatus>-?\d+)"
)

# user/stresssched.c's spinner report:
#   stresssched: report idx=<int> pid=<int> tickets=<int> runticks=<int> selections=<int>
STRESSSCHED_REPORT_RE = re.compile(
    r"stresssched: report idx=(?P<idx>\d+) pid=(?P<pid>\d+) "
    r"tickets=(?P<tickets>\d+) runticks=(?P<runticks>\d+) "
    r"selections=(?P<selections>\d+)"
)

# user/stresssched.c's chaos-worker report:
#   stresssched: chaos done toggles=<int>
# CHAOS_TOGGLES=40 (user/stresssched.c) is a safety cap on the loop, not
# an expected/guaranteed count: each iteration blocks on pause(1) (>= 1
# tick) plus real scheduling latency while N_SPINNERS=5 competitors are
# actively RUNNABLE, so the TARGET_TICKS=20 wall-clock window binds
# first in practice -- see test_stress_scheduler_race.py's own use of
# this pattern for the real measured range.
STRESSSCHED_CHAOS_RE = re.compile(
    r"stresssched: chaos done toggles=(?P<toggles>\d+)"
)


def parse_reports(pattern: "re.Pattern[str]", out: str) -> List[Dict[str, int]]:
    """Parse every match of `pattern` in `out` into a list of {field: int}
    dicts, in the order printed."""
    reports: List[Dict[str, int]] = []
    for m in pattern.finditer(out):
        reports.append({k: int(v) for k, v in m.groupdict().items()})
    return reports
