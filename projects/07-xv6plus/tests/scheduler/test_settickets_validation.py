"""Phase 4 / FR5: settickets(2) argument validation. Negative tickets
are rejected (-1) without changing anything; 0 is a valid, if
deliberately low-priority, configuration (see the zero-ticket-floor
discussion in docs/decisions/0010-lottery-scheduler-design.md and
test_lottery_zero_ticket_floor.py); an ordinary positive value is
accepted and observably takes effect; a value above SCHED_MAX_TICKETS
(kernel/sched.h) is rejected (-1) -- added during review followup, see
docs/decisions/0013-ticket-count-bound.md (an unbounded ticket count
would let a large-enough RUNNABLE total truncate through the lottery
draw's uint32 PRNG modulo, making some processes structurally
unreachable by any draw).

Drives user/tixvalidate.c.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("tixvalidate", prompt_timeout=15.0)

    assert "tixvalidate: done" in out, f"tixvalidate did not complete; got:\n{out}"

    def result_of(label: str) -> int:
        m = re.search(rf"tixvalidate: {re.escape(label)} -> (-?\d+)", out)
        assert m, f"missing '{label}' result line; got:\n{out}"
        return int(m.group(1))

    assert result_of("settickets(-1)") == -1, f"settickets(-1) should be rejected; got:\n{out}"
    assert result_of("settickets(0)") == 0, f"settickets(0) should be accepted; got:\n{out}"
    assert result_of("settickets(SCHED_MAX_TICKETS+1)") == -1, (
        f"settickets(SCHED_MAX_TICKETS+1) should be rejected (the ticket-"
        f"count upper bound, docs/decisions/0013-ticket-count-bound.md); "
        f"got:\n{out}"
    )
    assert result_of("settickets(SCHED_MAX_TICKETS)") == 0, (
        f"settickets(SCHED_MAX_TICKETS) (the boundary value itself) should "
        f"be accepted; got:\n{out}"
    )
    assert result_of("settickets(5)") == 0, f"settickets(5) should be accepted; got:\n{out}"

    m = re.search(r"tixvalidate: tickets_after=(\d+)", out)
    assert m, f"missing tickets_after line; got:\n{out}"
    assert int(m.group(1)) == 5, (
        f"tickets_after={m.group(1)}, expected 5 -- the last accepted "
        f"settickets() call in the sequence must be the value that took "
        f"effect; got:\n{out}"
    )
