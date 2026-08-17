"""Phase 4 / FR5, invariant #4 ("the scheduler always eventually
selects eligible work"): schedule_lottery()'s zero-ticket-floor
fallback (kernel/proc.c: pick_first_runnable(), triggered whenever the
whole RUNNABLE set's ticket total is exactly 0) keeps the scheduler
from stalling on RUNNABLE work even when every currently-RUNNABLE
process has 0 tickets. This test's workload -- four 0-ticket children
alongside three *finite*, wall-clock-bounded 30-ticket children that
eventually all exit -- is exactly the condition that fallback needs:
once the 30-ticket children finish, the RUNNABLE ticket total hits 0
and the 0-ticket children get picked. This is NOT a general proof that
a 0-ticket process always gets some CPU time under sustained
competition from a persistently-RUNNABLE nonzero-ticket process (it
does not: draw_and_run()'s weighted draw gives a 0-ticket process's
cumulative range zero width, so it structurally cannot win a weighted
draw while any nonzero-ticket process is also RUNNABLE) -- see
docs/decisions/0010-lottery-scheduler-design.md decision 5 and
docs/scheduler.md's "Zero-ticket floor" section for the full,
corrected statement of what this feature does and does not guarantee.
A captured real run in docs/scheduler.md showed 0-ticket children
getting runticks=1-2 vs. 26-29 for their 30-ticket peers -- clearly
disadvantaged, and every one of the 7 processes still completed and
was reaped.

Drives user/schedbench.c with four 0-ticket and three 30-ticket
children under SCHED_LOTTERY.
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
        out = q.run_cmd("schedbench 1 0 0 30 30 30 0 0", prompt_timeout=45.0)

    assert "schedbench: done" in out, (
        f"schedbench did not complete -- a zero-ticket process may never "
        f"have been selected, hanging the whole run (invariant #4 "
        f"violation); got:\n{out}"
    )
    reports = parse_schedbench_reports(out)
    assert len(reports) == 7, (
        f"expected 7 report lines (every child completed), got "
        f"{len(reports)}; out:\n{out}"
    )

    zero_ticket = [r for r in reports if r["tickets"] == 0]
    nonzero_ticket = [r for r in reports if r["tickets"] > 0]
    assert len(zero_ticket) == 4 and len(nonzero_ticket) == 3, (
        f"unexpected ticket split; reports:\n{reports}"
    )

    for r in zero_ticket:
        assert r["runticks"] >= 1, (
            f"pid={r['pid']} (0 tickets) got runticks={r['runticks']} -- "
            f"expected at least some CPU time via the zero-ticket-floor "
            f"fallback, not true starvation; reports:\n{reports}"
        )

    avg_zero = sum(r["runticks"] for r in zero_ticket) / len(zero_ticket)
    avg_nonzero = sum(r["runticks"] for r in nonzero_ticket) / len(nonzero_ticket)
    assert avg_nonzero > avg_zero * 3, (
        f"0-ticket children averaged {avg_zero:.1f} runticks, 30-ticket "
        f"children averaged {avg_nonzero:.1f} -- expected 0-ticket children "
        f"to be clearly disadvantaged (just not starved to exactly 0); "
        f"reports:\n{reports}"
    )
