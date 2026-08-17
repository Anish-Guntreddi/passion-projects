"""Shared helpers for the Phase 2 (process accounting) test suite.

Named `_acct_helpers` (not `_helpers`) deliberately: scripts/run_tests.py
runs every discovered tests/<category>/test_*.py module inside one
Python process, and each category's test modules load their own
`sys.path.insert(0, <own dir>)`-adjacent helper module by bare name
(`from _helpers import ...`, `from _acct_helpers import ...`). Two
categories both naming their helper module `_helpers` would collide in
`sys.modules` -- whichever category's tests run first "wins" that
cache entry, and every later category silently gets the wrong
module's contents (an ImportError here, since the two categories'
helpers export different names). tests/syscall/_helpers.py already
owns the plain `_helpers` name, so this module uses a
category-qualified one instead of colliding with it.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Matches lines printed by user/accttest.c's report():
#   accttest: <tag>: pid=<int> state=<int> sz=<int> runticks=<int> waitticks=<int> syscalls=<int>
ACCTTEST_REPORT = re.compile(
    r"accttest: (?P<tag>\w+): pid=(?P<pid>\d+) state=(?P<state>\d+) "
    r"sz=(?P<sz>\d+) runticks=(?P<runticks>\d+) waitticks=(?P<waitticks>\d+) "
    r"syscalls=(?P<syscalls>\d+)"
)


def parse_accttest_reports(out: str) -> Dict[str, Dict[str, int]]:
    """Parse every accttest report line in `out` into {tag: {field: int}}."""
    reports: Dict[str, Dict[str, int]] = {}
    for m in ACCTTEST_REPORT.finditer(out):
        reports[m.group("tag")] = {
            k: int(m.group(k)) for k in ("pid", "state", "sz", "runticks", "waitticks", "syscalls")
        }
    return reports


def require_stages(reports: Dict[str, Dict[str, int]], *tags: str) -> List[Dict[str, int]]:
    missing = [t for t in tags if t not in reports]
    assert not missing, f"accttest output missing report stage(s) {missing}; found {list(reports)}"
    return [reports[t] for t in tags]
