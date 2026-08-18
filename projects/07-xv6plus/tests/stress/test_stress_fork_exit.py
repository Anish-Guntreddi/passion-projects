"""Phase 6 / spec roadmap row 6 ("concurrent fork/exit ... stress").
Drives user/stressforkexit.c: ROUNDS rounds of CHILDREN_PER_ROUND
genuinely-concurrent fork/exit cycles (racing allocpid()/allocproc()/
freeproc()/wait_lock across all three harts), then a table-fill proof
that no proc-table slot leaked across that churn.

Invariants exercised: #1 (accounting/lifecycle state not corrupted by
concurrent fork/exit), #3 (telemetry init/fork/exit correctness under
load), #5 (zombie/free slots never leak -- the exact fill-count check
below is the direct proof: exactly NPROC-3 slots must still be free
after five rounds of concurrent churn, not fewer).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

ROUND_RE = re.compile(r"stressforkexit: round (\d+) children=(\d+) ok")
FILL_RE = re.compile(r"stressforkexit: fill count=(\d+)")

# kernel/param.h: NPROC. Kept as a literal here (not imported) the same
# way every other tests/*/test_*.py in this suite treats kernel
# constants -- see tests/scheduler/test_settickets_validation.py's
# SCHED_MAX_TICKETS literal for the precedent.
NPROC = 64
EXPECTED_ROUNDS = 5
CHILDREN_PER_ROUND = 8
# This process + init + sh hold the other 3 slots at fill time -- see
# user/stressforkexit.c's module comment for the exact accounting.
EXPECTED_FILL = NPROC - 3


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("stressforkexit", prompt_timeout=60.0)

        assert "stressforkexit: done" in out, (
            f"stressforkexit did not complete -- concurrent fork/exit stress "
            f"may have hung or corrupted proc-table state; got:\n{out}"
        )

        rounds = ROUND_RE.findall(out)
        assert len(rounds) == EXPECTED_ROUNDS, (
            f"expected {EXPECTED_ROUNDS} round-ok lines, got {len(rounds)}; "
            f"out:\n{out}"
        )
        for i, (r, n) in enumerate(rounds):
            assert int(r) == i and int(n) == CHILDREN_PER_ROUND, (
                f"round {i} reported r={r} children={n}, expected r={i} "
                f"children={CHILDREN_PER_ROUND}; out:\n{out}"
            )

        fill_m = FILL_RE.search(out)
        assert fill_m, f"missing fill-count line; got:\n{out}"
        fill_count = int(fill_m.group(1))
        assert fill_count == EXPECTED_FILL, (
            f"table-fill proof got {fill_count} successful forks, expected "
            f"exactly {EXPECTED_FILL} (NPROC={NPROC} minus the 3 slots held "
            f"by init/sh/this process) -- a mismatch here means either fewer "
            f"slots were free than expected (a leak from the {EXPECTED_ROUNDS} "
            f"earlier rounds -- invariant #5) or more (a double-count/other "
            f"corruption); out:\n{out}"
        )

        # Invariant #8: the kernel must still be fully healthy after
        # driving the proc table to its exact capacity and back down.
        regress = q.run_cmd("echo fork_exit_stress_regression_ok", prompt_timeout=10.0)
        assert "fork_exit_stress_regression_ok" in regress, (
            f"shell did not respond normally after the fork/exit stress run; "
            f"got:\n{regress}"
        )
