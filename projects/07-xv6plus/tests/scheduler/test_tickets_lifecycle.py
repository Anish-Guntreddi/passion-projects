"""Phase 4 / FR5, invariants #3 and #5: tickets follow the same
inherited-across-fork discipline Phase 1 established for trace_mask,
and a freshly-allocated proc-table slot -- whether brand new or
reused -- always starts at SCHED_DEFAULT_TICKETS, never a previous
occupant's configured value.

Drives user/tixtest.c twice: `tixtest 77` proves fork() inheritance
(child and parent both report tickets=77 after settickets(77));
`tixtest` (no argument, run second, in the same QEMU session) proves
the *next* run's "start" report -- whatever proc-table slot it lands
on -- doesn't inherit that leftover 77.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

START_RE = re.compile(r"tixtest: start: pid=(\d+) tickets=(\d+) selections=(\d+)")
CHILD_RE = re.compile(r"tixtest: child: pid=(\d+) tickets=(\d+) selections=(\d+)")
PARENT_RE = re.compile(r"tixtest: parent: pid=(\d+) tickets=(\d+) selections=(\d+)")

DEFAULT_TICKETS = 10  # kernel/sched.h: SCHED_DEFAULT_TICKETS


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()

        out1 = q.run_cmd("tixtest 77", prompt_timeout=15.0)
        assert "tixtest: done" in out1, f"first tixtest run did not complete; got:\n{out1}"

        start_m = START_RE.search(out1)
        child_m = CHILD_RE.search(out1)
        parent_m = PARENT_RE.search(out1)
        assert start_m and child_m and parent_m, f"missing a report line; got:\n{out1}"

        assert int(start_m.group(2)) == DEFAULT_TICKETS, (
            f"tixtest's own 'start' report (before settickets()) shows "
            f"tickets={start_m.group(2)}, expected the default "
            f"{DEFAULT_TICKETS}; got:\n{out1}"
        )
        assert int(child_m.group(2)) == 77, (
            f"forked child reports tickets={child_m.group(2)}, expected 77 -- "
            f"tickets should be inherited across fork() like trace_mask; "
            f"got:\n{out1}"
        )
        assert int(parent_m.group(2)) == 77, (
            f"parent reports tickets={parent_m.group(2)} after settickets(77), "
            f"expected it to still be 77; got:\n{out1}"
        )
        assert int(child_m.group(3)) <= 3, (
            f"freshly forked child reports implausibly large selections="
            f"{child_m.group(3)} -- expected near 0 (not inherited, unlike "
            f"tickets); got:\n{out1}"
        )

        out2 = q.run_cmd("tixtest", prompt_timeout=15.0)
        assert "tixtest: done" in out2, f"second tixtest run did not complete; got:\n{out2}"
        start2_m = START_RE.search(out2)
        assert start2_m, f"missing 'start' report line; got:\n{out2}"
        assert int(start2_m.group(2)) == DEFAULT_TICKETS, (
            f"second tixtest run's 'start' report shows tickets="
            f"{start2_m.group(2)}, expected the default {DEFAULT_TICKETS} -- "
            f"looks like the reused proc-table slot kept the first run's "
            f"settickets(77) instead of resetting on free; got:\n{out2}"
        )
