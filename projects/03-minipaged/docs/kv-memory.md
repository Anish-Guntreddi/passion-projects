# KV memory

Status: covers Phase 1 (contiguous baseline). Phase 2 (paged allocator,
`kv/{block,pool,table,manager}.py`) and Phase 4 (prefix sharing/COW) will
extend this document, not replace it — the contiguous baseline stays as
the thing paged allocation is measured against.

## Why a baseline before a paged allocator

Phase 2's paged KV allocator only means something in contrast to what it
replaces. Phase 1 builds the naive alternative first — "reserve one
contiguous span sized to the worst case" — specifically so Phase 2 can
run the *same trace* through both and report a real utilization/waste
comparison (spec Part 3, Phase 2 exit criterion: "same trace runs paged
and produces utilization statistics").

## `ContiguousMemoryModel` (`src/minipaged/kv/contiguous.py`)

A fixed-capacity pool of abstract "token slots" (see
`docs/decisions/0002-contiguous-baseline-simulated-capacity.md` for why
slots, not real tensor bytes). Per active sequence it tracks:

| Field      | Meaning |
|------------|---------|
| `reserved` | `prompt_len + max_new_tokens`, committed at admission. A contiguous allocator cannot grow a live reservation in place (no gaps between reservations by construction), so it must commit to the worst case up front. |
| `used`     | `prompt_len + tokens_generated` — the sequence's actual current KV length, updated every decode tick via `update_used`. |
| `waste`    | `reserved - used`. Capacity held but not backing real data for this sequence, and unusable by anyone else either — a contiguous span cannot be subdivided and lent out. |
| `free`     | `capacity_tokens - reserved_tokens` (summed over all active reservations). Capacity not committed to anyone, available to a new arrival. |

`ContiguousMemoryMetrics` (`snapshot()`) additionally reports
`utilization = used / capacity` and `waste_ratio = waste / capacity` as a
point-in-time view, suitable for plotting over the course of a trace
replay.

## Admission outcomes

`try_reserve(request_id, size, used)`:

- Returns `True` and commits the reservation if `size <= free_tokens`.
- Returns `False` (does not raise) if `size` fits within *total* capacity
  but not *currently free* capacity — the caller (the engine) leaves the
  request queued and retries on a later tick. This is what keeps core
  invariant #5 ("scheduler never admits work requiring more KV capacity
  than policy permits") true without ever silently over-committing.
- Raises `CapacityError` if `size > capacity_tokens` — no amount of
  waiting will ever free enough contiguous space, so the engine treats
  this as an unrecoverable rejection (`ADMISSION_REJECTED` +
  `CANCELLED`, per D7's default) rather than queuing it forever.

## What makes waste measurable (Phase 1 exit criterion)

"Variable-length trace demonstrates fragmentation/waste numerically." Two
trace properties are what actually produce nonzero waste, both exposed by
`TraceConfig` (`src/minipaged/simulation/trace.py`):

1. **Variable `max_new_tokens`** across requests — every sequence still
   reserves its own worst case, so a long-budget sequence sitting next to
   short-budget ones cannot lend its unused capacity to them even while
   both are running simultaneously.
2. **Early completion** (`target_output_len < max_new_tokens`, driven by
   `completion_fraction_range < 1.0`) — a sequence that stops generating
   before exhausting its budget leaves `reserved - used > 0` for the rest
   of its (already-reserved, not-yet-released) lifetime.

`examples/phase1_fragmentation_demo.py` runs a `TraceConfig` with both
properties enabled and prints the resulting waste/utilization numbers
over the course of the replay — the runnable evidence for this phase's
exit criterion. `tests/unit/test_contiguous.py` asserts the same
numerically (waste > 0 for a variable-length trace; waste == 0 when every
sequence uses its full budget) as a regression test.
