"""Phase 6 / spec roadmap row 6 ("memory-pressure ... stress"). Drives
user/stressvm.c: N_WORKERS (3, == NCPU) processes concurrently grow/
touch/shrink lazily-allocated memory across every hart (Part 1, clean
-- kalloc()/mappages() under real concurrent pressure, but well within
the physical budget), then N_WORKERS fresh processes race each other
to genuinely exhaust the *same shared* physical page pool at once
(Part 2) -- unlike tests/vm/test_memory_exhaustion_recovery.py's
single child, this is real multi-hart contention on kalloc()'s
freelist as it empties.

Invariants exercised: #6 (VM changes preserve isolation -- each
exhausted worker is killed cleanly, contained to itself, not a kernel
panic, under concurrent pressure this project's Phase 5 tests never
produced), #2 (no new lock-order hazard from concurrent kalloc()
pressure -- kalloc()'s own lock is upstream and untouched), #1/#3/#5
(pagefaults/pagefaults_failed counters stay accurate and every
worker's memory is fully reclaimed on reap, proven by the recovery
check), #8 (real concurrent memory exhaustion must not destabilize the
kernel for anything else running).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stress_helpers import (  # noqa: E402
    STRESSVM_CLEAN_RE,
    STRESSVM_EXHAUST_REAP_RE,
    parse_reports,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qemu_session import QemuSession  # noqa: E402

N_WORKERS = 3
PAGES_PER_ROUND = 300
ROUNDS = 4


def run(project_root: Path) -> None:
    with QemuSession(project_root) as q:
        q.boot_to_shell()
        out = q.run_cmd("stressvm", prompt_timeout=90.0)

        assert "stressvm: done" in out, (
            f"stressvm did not complete -- concurrent memory-pressure stress "
            f"may have hung or destabilized the kernel; got:\n{out}"
        )
        assert "stressvm: clean phase done" in out, (
            f"clean concurrent grow/touch/shrink phase did not complete "
            f"cleanly; got:\n{out}"
        )

        clean_reports = parse_reports(STRESSVM_CLEAN_RE, out)
        assert len(clean_reports) == N_WORKERS, (
            f"expected {N_WORKERS} clean-phase worker reports, got "
            f"{len(clean_reports)} -- console output may have interleaved; "
            f"out:\n{out}"
        )
        for r in clean_reports:
            gained = r["pf_after"] - r["pf_before"]
            assert gained >= ROUNDS * PAGES_PER_ROUND, (
                f"clean worker idx={r['idx']} pid={r['pid']} only gained "
                f"{gained} pagefaults across {ROUNDS} rounds of "
                f"{PAGES_PER_ROUND} pages each under concurrent kalloc() "
                f"pressure -- expected at least {ROUNDS * PAGES_PER_ROUND}; "
                f"reports:\n{clean_reports}"
            )
            assert r["failed_after"] == r["failed_before"], (
                f"clean worker idx={r['idx']} pid={r['pid']} saw "
                f"pagefaults_failed change ({r['failed_before']} -> "
                f"{r['failed_after']}) during a workload well within the "
                f"~128MB physical budget even with {N_WORKERS} concurrent "
                f"workers -- concurrent kalloc() pressure should not "
                f"spuriously fail an allocation that should succeed; "
                f"reports:\n{clean_reports}"
            )

        exhaust_reaps = parse_reports(STRESSVM_EXHAUST_REAP_RE, out)
        assert len(exhaust_reaps) == N_WORKERS, (
            f"expected {N_WORKERS} exhaust-phase reap lines, got "
            f"{len(exhaust_reaps)}; out:\n{out}"
        )
        killed = [r for r in exhaust_reaps if r["xstatus"] == -1]
        assert len(killed) == N_WORKERS, (
            f"expected all {N_WORKERS} concurrently-exhausting workers to be "
            f"killed cleanly (xstatus == -1) via the recognized-but-"
            f"unserviceable vmfault() failure path -- got {len(killed)}; a "
            f"worker surviving here would mean it never actually hit real "
            f"exhaustion despite {N_WORKERS} processes racing for the same "
            f"~128MB pool, which would undermine this test's whole premise; "
            f"reaps:\n{exhaust_reaps}"
        )

        assert "reached PAGE_BUDGET" not in out, (
            f"an exhaust worker ran to its PAGE_BUDGET without ever hitting "
            f"real physical-memory exhaustion, even with {N_WORKERS} "
            f"concurrent consumers of the same pool; got:\n{out}"
        )

        assert "stressvm: recovery ok" in out, (
            f"post-exhaustion recovery allocation failed -- memory from the "
            f"{N_WORKERS} concurrently-exhausted workers was not fully "
            f"reclaimed on reap; got:\n{out}"
        )
        assert "stressvm: recovery xstatus=0" in out, (
            f"post-exhaustion recovery process did not exit cleanly; "
            f"got:\n{out}"
        )

        # Invariant #8: the kernel must still be fully healthy after real
        # concurrent multi-hart memory exhaustion, not just able to run one
        # more tiny recovery allocation inside the same test program.
        regress = q.run_cmd("echo vm_stress_regression_ok", prompt_timeout=10.0)
        assert "vm_stress_regression_ok" in regress, (
            f"shell did not respond normally after the memory-pressure "
            f"stress run; got:\n{regress}"
        )
