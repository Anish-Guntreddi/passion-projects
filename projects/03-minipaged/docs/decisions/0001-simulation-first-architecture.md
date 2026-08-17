# 0001 — Simulation-first architecture

- **Status:** Accepted
- **Phase:** 0 (and load-bearing through Phase 4)
- **Related spec sections:** S1.6 (implementation strategy, "load-bearing"),
  agent execution rule #1

## Context

MiniPaged's whole point is to make scheduling and KV-memory allocation
decisions observable and testable. If a real Hugging Face/PyTorch model
were wired in from the start, every allocator or scheduler test would
also be a (slow, GPU-dependent, nondeterministic-unless-carefully-seeded)
model test. The spec is explicit that this is backwards: "the first
implementation simulates KV allocation independent of actual model
tensors. Only after allocator and scheduler invariants are well-tested is
a real model execution path connected."

## Decision

Phases 0-4 (domain simulator, contiguous baseline, paged allocator,
scheduler, prefix sharing) run entirely without a real model. Requests
carry a `prompt_len` and `max_new_tokens`/`target_output_len` instead of
real token ids and real generated text; a `SimClock` stands in for wall
time; a `SimulationEngine` moves requests through
`waiting -> running -> completed` by simulated ticks, not by running any
inference. KV usage is tracked as counts of abstract "token slots"
(Decision 0002), not real tensors.

Only in Phase 5 does `minipaged.model` stop being a boundary placeholder:
one small causal LM is integrated behind an interface the scheduler and
KV subsystem stay independent of (agent execution rule #1: "do not
integrate a real model before allocator/scheduler tests exist").

## Consequences

- Every Phase 0-4 test runs in milliseconds on any machine, no GPU or
  model download required.
- The "tokenizer boundary" (FR2) exists as a real seam
  (`Request.prompt_token_ids`) but is a documented no-op until Phase 5 —
  prompt token ids are `range(prompt_len)` placeholders
  (`SimulationEngine.load_trace`).
- Sampling logic (`minipaged.sampling.config.SamplingConfig`) is
  data-only until Phase 5; nothing consumes it yet.
- Anything that claims to measure real-world latency/throughput before
  Phase 5-7 would be a fabricated number — the spec's agent execution
  rule #4 ("never fabricate performance numbers") is enforced by this
  decision, not just by discipline: there is no model to measure yet.
