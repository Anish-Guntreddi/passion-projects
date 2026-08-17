# ADR 0003: Handwritten HTTP parser (spec decision D6)

- **Status**: Accepted
- **Date**: 2026-08-17
- **Phase**: Decided before Phase 1, per the spec's kickoff prompt ("D1 and
  D6 before Phase 1").

## Context

Spec §1.9, decision D6: *"Handwritten vs library parser → default per
brief: handwritten educational parser within narrow scope."* Mature options
exist (`llhttp`/Node's HTTP parser, `picohttpparser`, Boost.Beast's parser)
and would be faster to integrate and almost certainly more RFC-complete
than anything hand-rolled here.

## Decision

`HttpRequestParser` (`include/arcserve/protocol/http_parser.hpp` /
`src/protocol/http_parser.cpp`) is written from scratch: a byte-incremental
state machine (`RequestLine → Headers → Body → Done`) with no parsing
library dependency.

### Why handwritten

- **Portfolio-wide rule 5**: "Keep code interview-explainable. No copying
  whole implementations from mature libraries; educational components stay
  hand-built." A parser is exactly the kind of component a systems-role
  interview would ask to walk through line by line — "why does
  `parse_request_line` check for a CRLF split across the chunk boundary
  before searching `data`?" A vendored parser can't be walked through that
  way with the same credibility.
- **FR3 is explicitly about the incremental/fragmented-input property**,
  not about RFC completeness: *"Incremental parser for a small protocol
  handling fragmented input... unsupported/oversized messages rejected
  predictably."* The scope in `docs/protocol-scope.md` (ADR 0002) is
  narrow specifically so that a hand-written state machine covering it
  stays small enough to review and test exhaustively — the byte-at-a-time
  and every-split-offset unit tests
  (`tests/unit/test_http_parser.cpp`) are only practical to write and
  reason about because the parser's state space is small and fully owned.
- **Consistency with `docs/decisions/0001-...`**'s reasoning on warning
  strictness: this project repeatedly chooses "small and fully understood"
  over "complete and external" wherever the roadmap's actual learning
  goals (event loop, backpressure, worker pool — not HTTP-spec trivia)
  don't require the latter.

### Why this doesn't block later performance work

Spec decision D6's scope is "within narrow scope" — it is not a claim that
a hand-written parser will out-perform `picohttpparser`. Stretch goal
"HTTP parser performance comparison" (roadmap §Stretch goals) is exactly
where that question gets asked, empirically, with committed benchmark
data, not assumed at Phase 1. Rule 6 applies: no performance claim about
this parser (or comparison to a library one) will be made anywhere in this
project without measured, committed numbers.

## Consequences

- The parser only ever needs to correctly implement
  `docs/protocol-scope.md` — not general HTTP/1.1 — which is what keeps
  "handwritten" tractable. Expanding scope later means expanding the state
  machine deliberately, with new tests, not "swap in a real parser because
  the hand-written one hit a wall".
- Every parser error path is a named, tested case (`RejectsUnsupportedMethod`,
  `RejectsOversizedBody`, `RejectsTooManyHeaders`, etc. in
  `tests/unit/test_http_parser.cpp`) rather than "whatever the library
  does" — this is a direct requirement for the "malformed protocol"
  failure-mode testing called out in spec §1.6.
- If a future phase's fuzz testing (Phase 3, per the roadmap) finds a
  correctness gap, the fix is auditable in a ~350-line file, not a
  dependency upgrade.
