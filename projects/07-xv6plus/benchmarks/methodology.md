# Benchmark methodology: scheduler experiment (Phase 4, FR5)

Phase 4's exit criterion is "Benchmark compares fairness/turnaround/
response vs baseline scheduler." This document is that comparison.
Every number below is computed directly from the raw console
transcripts in [`benchmarks/raw/scheduler/`](raw/scheduler/) -- each
one a plain-text capture of one real `qemu-system-riscv64` run via
`scripts/qemu_session.py`, with the exact shell command and a UTC
capture timestamp recorded at the top of the file. Nothing below is
hand-typed or estimated; every table cell was computed by parsing
those files (see the parsing snippet at the bottom of this document)
and cross-checked by hand against the raw `schedbench: report ...`
lines.

## Setup

- **Kernel:** this repository's own build (`make clean && make
  kernel/kernel fs.img`), `riscv64-unknown-elf-gcc` 13.2.
- **QEMU:** `qemu-system-riscv64` 8.2, `-machine virt -smp 3 -m 128M`
  (this project's standard test-harness configuration --
  `scripts/qemu_session.py`).
- **Workload driver:** `user/schedbench.c` -- see `docs/scheduler.md`
  for why it is wall-clock-bounded (`TARGET_TICKS=30`, ~3 real
  seconds per run) rather than a fixed iteration count, and why each
  child reports over a private pipe rather than the shared console.
- **CPU-only.** No GPU involved in this project; the portfolio's
  GPU-benchmark lock does not apply here.

## Run 1: unequal tickets, baseline (`SCHED_RR`) vs. lottery (`SCHED_LOTTERY`)

Same ticket assignment (`60 20 10 5 60 20 10`, seven children, a 12x
spread between the busiest and quietest), run once under each policy.
Raw files: `baseline_unequal_tickets.txt`, `lottery_unequal_tickets.txt`.

| tickets | SCHED_RR runticks | SCHED_LOTTERY runticks |
|---|---|---|
| 60 | 14, 13 | 22, 22 |
| 20 | 11, 13 | 17, 6 |
| 10 | 12, 13 | 6, 10 |
| 5 | 12 | 5 |

**Per-ticket-tier average runticks:**

| tickets | SCHED_RR avg | SCHED_LOTTERY avg |
|---|---|---|
| 60 | 13.50 | 22.00 |
| 20 | 12.00 | 11.50 |
| 10 | 12.50 | 8.00 |
| 5 | 12.00 | 5.00 |

Under `SCHED_RR`, runticks range 11-14 (a 12x ticket spread produces
only a 1.27x runticks spread -- round-robin ignores tickets, as
designed: `schedule_roundrobin()` never reads `p->tickets`). Under
`SCHED_LOTTERY`, the same 12x ticket spread produces a 4.4x runticks
spread (22 vs. 5) with a clear monotonic-by-tier ordering. This is the
headline fairness claim: **lottery scheduling measurably
differentiates by ticket share; the baseline measurably does not.**

**Selection-share accuracy (lottery only).** Total tickets = 185,
total selections = 95 across the 7 children. Comparing each process's
selection share (`selections / 95`) against its ticket share
(`tickets / 185`):

| pid | tickets | ticket share | selections | selection share | difference |
|---|---|---|---|---|---|
| 12 | 60 | 0.324 | 23 | 0.242 | 0.082 |
| 13 | 20 | 0.108 | 18 | 0.189 | 0.081 |
| 14 | 10 | 0.054 | 7  | 0.074 | 0.020 |
| 15 | 5  | 0.027 | 6  | 0.063 | 0.036 |
| 16 | 60 | 0.324 | 23 | 0.242 | 0.082 |
| 17 | 20 | 0.108 | 7  | 0.074 | 0.034 |
| 18 | 10 | 0.054 | 11 | 0.116 | 0.062 |

Every process's selection share is within 0.09 of its ticket share
over 95 total draws -- a reasonably tight fit for a weighted random
draw at this sample size, and well inside the generous Â±0.20 bound
`tests/scheduler/test_sched_lottery_fairness.py` checks on every test
run (deliberately loose to avoid flakiness; see that test's own
docstring).

## Run 2: zero-ticket floor (invariant #4's starvation bound)

Tickets `0 0 30 30 30 0 0` under `SCHED_LOTTERY`. Raw file:
`lottery_zero_ticket_floor.txt`.

| tickets | runticks | selections |
|---|---|---|
| 0 | 1, 1, 2, 2 | 2, 2, 3, 3 |
| 30 | 27, 29, 26 | 28, 30, 27 |

0-ticket average runticks: 1.50. 30-ticket average runticks: 27.33 --
an 18.2x ratio. Every 0-ticket process still got selected at least
twice and completed normally (all 7 processes reported and were
reaped; `schedbench: done` printed) -- the concrete, measured shape of
"disadvantaged, not starved."

## Run 3: equal tickets under lottery (sanity check)

Tickets `10 10 10 10 10 10 10` (all equal) under `SCHED_LOTTERY`. Raw
file: `lottery_equal_tickets.txt`. Runticks: 16, 13, 14, 11, 12, 11,
11 -- range 11-16 (1.45x), comparable in spread to Run 1's `SCHED_RR`
baseline (1.27x). This is the expected sanity result: when every
process has an equal ticket share, lottery scheduling's fairness
looks statistically similar to round-robin's -- the two policies only
diverge once tickets are actually unequal (Run 1), which is exactly
what a weighted lottery is supposed to do.

## Reproducing this data

```sh
python3 - <<'PYEOF'
import re, statistics
files = ["baseline_unequal_tickets", "lottery_unequal_tickets",
         "lottery_zero_ticket_floor", "lottery_equal_tickets"]
RE = re.compile(r"tickets=(\d+).*?runticks=(\d+).*?selections=(\d+)")
for name in files:
    text = open(f"benchmarks/raw/scheduler/{name}.txt").read()
    rows = [(int(a), int(b), int(c)) for a, b, c in RE.findall(text)]
    print(name, rows)
PYEOF
```

Or capture a fresh run: boot `scripts/qemu_session.py`'s `QemuSession`
and run any `schedbench policy tickets...` command (see
`docs/scheduler.md`'s "Benchmark" section and `user/schedbench.c`'s
module comment for the exact shape and the `MAXARGS=10` shell
argument limit). Re-running is expected to reproduce the same
*qualitative* result (baseline flat, lottery proportional, zero-ticket
disadvantaged-not-starved) but not bit-identical numbers: the lottery
PRNG is fixed-seed (ADR-0010) but real wall-clock scheduling jitter
across QEMU/host still varies run to run, which is exactly why every
threshold in `tests/scheduler/*.py` is a loose, documented bound
rather than an exact match against this specific captured data.

## What's deferred to Phase 8

Plotted graphs (this repo structure's `benchmarks/plots/`) are a
Phase 8 (portfolio hardening) deliverable per the roadmap ("before/
after scheduling or memory experiment graph," spec Â§1.7); Phase 4's
own exit criterion only requires the comparison itself, which the
tables above are. No `benchmarks/plots/` directory exists yet.
