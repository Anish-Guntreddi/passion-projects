"""Phase 4 / FR5: schedpolicy(2) argument validation, plus an
invariant #8 regression check (observability/extension code must not
destabilize the kernel it observes) -- toggling the global scheduling
policy twice, with one rejected out-of-range call in between, must not
leave the kernel in a state where an ordinary subsequent command
breaks.

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

        assert result_of("schedpolicy(99)") == -1, (
            f"schedpolicy(99) (outside {{SCHED_RR, SCHED_LOTTERY}}) should be "
            f"rejected; got:\n{out}"
        )
        assert result_of("schedpolicy(SCHED_LOTTERY)") == 0, (
            f"schedpolicy(SCHED_LOTTERY) should return the *previous* policy, "
            f"SCHED_RR (0) -- a fresh boot has never changed it before this "
            f"call, and the rejected schedpolicy(99) just above must not have "
            f"either; got:\n{out}"
        )
        assert result_of("schedpolicy(SCHED_RR)") == 1, (
            f"schedpolicy(SCHED_RR) should return the previous policy, "
            f"SCHED_LOTTERY (1), set by the call just above; got:\n{out}"
        )

        # Invariant #8: an ordinary command right after toggling the
        # scheduling policy twice (with one rejected call in between)
        # must still work normally.
        regress = q.run_cmd("echo scheduler_regression_ok", prompt_timeout=10.0)
        assert "scheduler_regression_ok" in regress, (
            f"shell did not respond normally after schedpolicy() toggling; "
            f"got:\n{regress}"
        )
