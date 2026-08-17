"""Phase 4 / FR5 exit criterion: "Benchmark compares fairness/
turnaround/response vs baseline scheduler." This is the *baseline*
half of that comparison: under SCHED_RR (unmodified upstream round-
robin), unequal ticket counts must have no measurable effect --
runticks stays roughly equal across children regardless of how many
tickets each was given, because schedule_roundrobin() (kernel/proc.c)
never reads p->tickets at all. See test_sched_lottery_fairness.py for
the contrasting lottery-policy half, and docs/scheduler.md for a
captured real run (runticks 16,12,11,13,13,12,11 for tickets
60,20,10,5,60,20,10 -- a 12x ticket spread producing under 1.5x
runticks spread) that this test's tolerance is chosen from.

Drives user/schedbench.c with a 12x ticket spread (60:5) under
SCHED_RR.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sched_helpers import parse_schedbench_reports  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("schedbench 0 60 20 10 5 60 20 10", prompt_timeout=45.0)

    assert "schedbench: done" in out, f"schedbench did not complete; got:\n{out}"
    reports = parse_schedbench_reports(out)
    assert len(reports) == 7, f"expected 7 report lines, got {len(reports)}; out:\n{out}"

    runticks = [r["runticks"] for r in reports]
    lo, hi = min(runticks), max(runticks)
    assert lo > 0, f"a child reported 0 runticks under baseline RR; reports:\n{reports}"
    # Round-robin gives every RUNNABLE process an equal-length turn
    # regardless of tickets, so the spread across children (whose
    # ticket counts range 5..60, a 12x spread) should stay small.
    # Generous factor-of-2 tolerance for scheduling/timing jitter.
    assert hi <= lo * 2, (
        f"runticks spread too wide under baseline RR ({lo}..{hi}) for a "
        f"policy that should ignore ticket counts entirely; reports:\n{reports}"
    )

    # And directly by ticket rank: the highest-ticket child must not
    # get a strongly larger share than the lowest-ticket child (that
    # would indicate tickets somehow still influenced round-robin
    # selection).
    by_tickets = sorted(reports, key=lambda r: r["tickets"])
    highest_tickets_runticks = by_tickets[-1]["runticks"]
    lowest_tickets_runticks = by_tickets[0]["runticks"]
    assert highest_tickets_runticks <= lowest_tickets_runticks * 2, (
        f"the highest-ticket child got more than 2x the lowest-ticket "
        f"child's runticks under baseline RR -- tickets should have no "
        f"effect under this policy; reports:\n{reports}"
    )
