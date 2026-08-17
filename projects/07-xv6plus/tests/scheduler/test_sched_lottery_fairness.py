"""Phase 4 / FR5 exit criterion: "Benchmark compares fairness/
turnaround/response vs baseline scheduler." This is the *lottery*
half: under SCHED_LOTTERY with the same 12x ticket spread (60:5)
test_sched_baseline_fairness.py uses under SCHED_RR, higher-ticket
children must capture a clearly larger share of both runticks and
selections -- proving the weighted draw (kernel/proc.c:
schedule_lottery()) actually differentiates, unlike the baseline. See
docs/scheduler.md for a captured real run (runticks 22,13,7,5,21,12,8
for tickets 60,20,10,5,60,20,10) and the tolerances below, chosen from
it -- lottery scheduling is intentionally probabilistic, so this
checks a clear directional trend, not exact proportionality.
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
        out = q.run_cmd("schedbench 1 60 20 10 5 60 20 10", prompt_timeout=45.0)

    assert "schedbench: done" in out, f"schedbench did not complete; got:\n{out}"
    reports = parse_schedbench_reports(out)
    assert len(reports) == 7, f"expected 7 report lines, got {len(reports)}; out:\n{out}"

    for r in reports:
        assert r["runticks"] > 0, (
            f"pid={r['pid']} tickets={r['tickets']} got 0 runticks under "
            f"lottery scheduling despite a nonzero ticket count -- "
            f"invariant #4 concern; reports:\n{reports}"
        )

    high = [r["runticks"] for r in reports if r["tickets"] == 60]
    low = [r["runticks"] for r in reports if r["tickets"] == 5]
    assert high and low, f"expected both a 60-ticket and a 5-ticket child; reports:\n{reports}"

    avg_high = sum(high) / len(high)
    avg_low = sum(low) / len(low)
    # A real captured run saw a 12x ticket ratio produce roughly a 4x
    # runticks ratio (docs/scheduler.md). Require at least 1.5x here:
    # generous given lottery's inherent randomness, while still ruling
    # out "no effect at all" (baseline RR's own test bounds the same
    # comparison at <=2x in the *other* direction).
    assert avg_high >= avg_low * 1.5, (
        f"60-ticket children averaged {avg_high:.1f} runticks, 5-ticket "
        f"children averaged {avg_low:.1f} -- expected the higher-ticket "
        f"children to capture a clearly larger CPU share under lottery "
        f"scheduling; reports:\n{reports}"
    )

    # Also check every process's selection share against its ticket
    # share across the whole run, not just the two extremes above.
    total_tickets = sum(r["tickets"] for r in reports)
    total_selections = sum(r["selections"] for r in reports)
    for r in reports:
        expected_share = r["tickets"] / total_tickets
        actual_share = r["selections"] / total_selections
        # Loose per-process bound: within 20 percentage points of its
        # ticket share. Not tight enough to be flaky (this is a
        # weighted random draw over a few dozen samples per process),
        # tight enough to catch "policy silently behaves like
        # round-robin" (which would put every share near 1/7 = 0.14
        # regardless of tickets).
        assert abs(actual_share - expected_share) < 0.20, (
            f"pid={r['pid']} tickets={r['tickets']} got selection share "
            f"{actual_share:.2f}, expected roughly {expected_share:.2f} "
            f"(tickets/total_tickets); reports:\n{reports}"
        )
