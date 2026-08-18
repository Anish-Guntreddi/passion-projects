"""Phase 6 / spec roadmap row 6 ("scheduler stress"). Drives
user/stresssched.c: N_SPINNERS (5) processes competing for CPU under
SCHED_LOTTERY while a separate CHAOS process concurrently -- on its own
hart -- calls schedpolicy()/settickets() rapidly and repeatedly, racing
schedule_lottery()'s live two-pass draw and sched_lock against the
spinners' own in-flight scheduling decisions. Unlike
user/schedbench.c (Phase 4), which sets policy exactly once from a
quiescent parent before any competing workload is RUNNABLE, this
program switches policy and ticket counts *while* five competing
processes are actively being scheduled across every hart.

Invariants exercised: #4 (the scheduler always eventually selects
eligible work -- every spinner must complete with nonzero runticks and
selections despite the policy being yanked out from under it
mid-run), #2 (sched_lock is a leaf lock, ADR-0010 -- rapid concurrent
schedpolicy()/settickets() churn from a live competing process must
not expose a new lock-order edge), #8 (adversarial policy churn from a
process that is itself part of the competing workload must not
destabilize the kernel).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stress_helpers import (  # noqa: E402
    STRESSSCHED_CHAOS_RE,
    STRESSSCHED_REPORT_RE,
    parse_reports,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

N_SPINNERS = 5


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("stresssched", prompt_timeout=60.0)

        assert "stresssched: done" in out, (
            f"stresssched did not complete -- concurrent policy/ticket "
            f"churn against a live competing workload may have hung the "
            f"scheduler (invariant #4 violation) or crashed the kernel; "
            f"got:\n{out}"
        )
        chaos_match = STRESSSCHED_CHAOS_RE.search(out)
        assert chaos_match is not None, (
            f"the chaos worker (concurrent schedpolicy()/settickets() "
            f"churn) did not complete; got:\n{out}"
        )
        # CHAOS_TOGGLES=40 (user/stresssched.c) is a safety cap, not the
        # expected count: each iteration blocks on pause(1) (>= 1 tick)
        # plus real scheduling latency while N_SPINNERS=5 competitors are
        # actively RUNNABLE, so TARGET_TICKS=20 binds first in practice.
        # Five real captured runs on this project's dev host measured
        # toggles in [9, 11]; MIN_CHAOS_TOGGLES=3 is a liveness floor well
        # below that range -- it catches a real regression (e.g. the
        # chaos worker getting starved down to 0-2 toggles, which would
        # silently defeat this test's whole purpose: racing live policy
        # churn against the spinners) without being sensitive to ordinary
        # host/QEMU timing variance.
        MIN_CHAOS_TOGGLES = 3
        chaos_toggles = int(chaos_match.group("toggles"))
        assert chaos_toggles >= MIN_CHAOS_TOGGLES, (
            f"chaos worker only completed {chaos_toggles} "
            f"schedpolicy()/settickets() toggles (expected "
            f">= {MIN_CHAOS_TOGGLES}) -- too little concurrent policy "
            f"churn actually raced the spinners for this to be a "
            f"meaningful stress run; got:\n{out}"
        )

        reports = parse_reports(STRESSSCHED_REPORT_RE, out)
        assert len(reports) == N_SPINNERS, (
            f"expected {N_SPINNERS} spinner reports, got {len(reports)}; "
            f"out:\n{out}"
        )

        pids = [r["pid"] for r in reports]
        assert len(set(pids)) == N_SPINNERS, (
            f"spinner reports do not all have distinct pids; reports:\n{reports}"
        )

        for r in reports:
            assert r["runticks"] > 0, (
                f"spinner idx={r['idx']} pid={r['pid']} tickets={r['tickets']} "
                f"got runticks=0 -- never actually ran despite completing, "
                f"which should be impossible; reports:\n{reports}"
            )
            assert r["selections"] > 0, (
                f"spinner idx={r['idx']} pid={r['pid']} tickets={r['tickets']} "
                f"got selections=0 -- the scheduler never picked it even "
                f"once despite rapid concurrent policy churn (invariant #4: "
                f"the scheduler must always eventually select eligible "
                f"work); reports:\n{reports}"
            )

        # Invariant #8: the kernel must still be fully healthy after
        # adversarial concurrent scheduler-policy churn against a live
        # competing workload across every hart.
        regress = q.run_cmd("echo sched_stress_regression_ok", prompt_timeout=10.0)
        assert "sched_stress_regression_ok" in regress, (
            f"shell did not respond normally after the scheduler stress "
            f"run; got:\n{regress}"
        )
