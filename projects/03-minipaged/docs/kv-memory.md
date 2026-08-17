# KV memory

Status: covers Phase 1 (contiguous baseline) and Phase 2 (paged
allocator). Phase 4 (prefix sharing/COW) will extend this document
further, not replace it — the contiguous baseline stays as the thing
paged allocation is measured against.

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

## Phase 2: the paged allocator (`kv/{block,pool,table,manager}.py`)

Four cooperating pieces, matching FR5/FR9's public interfaces exactly:

| Type | File | Role |
|------|------|------|
| `PhysicalBlock` | `block.py` | One fixed-size slot: a `block_id` and a `ref_count`. `is_free` iff `ref_count == 0`. |
| `BlockPool` | `pool.py` | Owns every `PhysicalBlock` plus the free list. Four primitives per FR5: `allocate()` (fresh block, ref_count=1), `free(block_id)` (sole-owner teardown, requires ref_count==1), `retain(block_id)` (ref_count += 1, unused until Phase 4 sharing), `release(block_id)` (ref_count -= 1, frees at 0 — the general decrement, correct whether or not shared). |
| `BlockTable` | `table.py` | One sequence's logical block list. `grow_to(num_tokens)` lazily allocates only the shortfall from the pool, all-or-nothing (rolls back on `BlockPoolExhausted`). `blocks_needed_for(num_tokens, block_size)` is the one place the token-to-block rounding rule lives — reused by `PagedKVManager` and Phase 3's admission ledger. |
| `PagedKVManager` + `PagedKVEngine` | `manager.py` | Ties the pool/table into the simulation engine, mirroring `ContiguousKVEngine`'s three-hook composition exactly (plus a fourth, new hook — see below). |

### Admission: current need, not worst case (the actual paging benefit)

`PagedKVManager.admit(request_id, prompt_len)` needs only
`blocks_needed_for(prompt_len)` free blocks — never
`prompt_len + max_new_tokens`. This is the deliberate contrast with
Phase 1 (see `docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md`
for the full argument and its consequences). `PagedKVEngine._try_admit`
still hard-cancels a request whose own worst case
(`blocks_needed_for(prompt_len + max_new_tokens)`) exceeds *total* pool
capacity (mirroring `ContiguousKVEngine`'s `CapacityError` path, D7
default), and soft-rejects (queued, retried) when currently-free blocks
are merely insufficient right now.

### Decode-time growth can fail: `_can_decode` and `DECODE_STALLED`

Because admission does not reserve a sequence's future growth, a running
sequence's block table can need to grow at decode time with no free block
available — `SimulationEngine._can_decode(seq)` (new Phase 2 hook, see
`simulation/engine.py`) is checked immediately before `step_decode()`
runs; `PagedKVEngine` overrides it to call
`PagedKVManager.can_grow_to`, a pure, non-mutating check. If it returns
False, the sequence is left completely unchanged and an
`EventType.DECODE_STALLED` event is recorded instead of `DECODE_STEP` —
retried automatically next tick, since every tick re-attempts every
running sequence unconditionally. This is the S1.9 "KV exhaustion"
failure mode made observable and safe rather than a crash.

### `PagedKVMetrics` (block-quantized, contrast with `ContiguousMemoryMetrics`)

| Field | Meaning |
|-------|---------|
| `reserved_tokens` | `allocated_blocks * block_size` — physically committed, rounded up to whole blocks (not exact, unlike Phase 1's `reserved`). |
| `used_tokens` | Sum of `current_len` across active sequences — exact, same meaning as Phase 1's `used`. |
| `waste` | `reserved_tokens - used_tokens` — purely block-rounding overhead now (at most `block_size - 1` per active sequence), not "committed-but-never-used budget" as in Phase 1. |

### Same trace, both allocators: the actual comparison

`examples/phase2_paged_kv_demo.py` replays the *identical* trace and
total token capacity Phase 1's fragmentation demo used (400 token-slots,
seed 7) through both `ContiguousKVEngine` and `PagedKVEngine`
(`block_size=16` blocks summing to the same 400-token capacity) and
prints both peak waste ratios side by side. Measured result: **63.7%**
peak waste ratio for contiguous versus **31.5%** for paged, on the same
inputs — this is the Phase 2 exit criterion ("same trace runs paged and
produces utilization statistics") made concrete, and
`tests/unit/test_manager.py::test_paged_allocator_wastes_less_than_contiguous_on_the_same_trace`
asserts the same comparison numerically as a regression test (not a fixed
threshold, since exact numbers depend on the trace's random draw —
strictly `paged < contiguous` on the same trace/capacity, which is the
claim that matters and holds regardless of seed).

### Known limitation: optimistic admission can deadlock without a scheduler

Documented in full in
`docs/decisions/0004-paged-admission-optimistic-with-decode-stall.md`
and demonstrated by
`tests/unit/test_manager.py::test_two_sequences_can_deadlock_under_optimistic_admission`:
admitting on current need alone means multiple concurrently-running
sequences can jointly demand more blocks than the pool has, with none of
them ever able to complete and release. Phase 2 alone does not prevent
this — Phase 3's scheduler adds the KV-capacity-aware admission control
that does (see `docs/scheduler.md` and
`docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md`).
