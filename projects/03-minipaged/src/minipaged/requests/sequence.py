"""SequenceState: per-request runtime execution progress.

FR9 calls out SequenceState as a public interface distinct from Request:
Request is the client-facing object (what was asked for). SequenceState is
the runtime's bookkeeping for how far generation has gotten once a request
is admitted -- has prefill happened yet, how many tokens has it generated
so far. Separating them keeps "what the client wants" isolated from "what
the runtime has done so far", which grows new fields (block tables, KV
pointers, ...) in later phases without ever touching Request.
"""

from __future__ import annotations

import dataclasses

from minipaged.requests.request import Request


@dataclasses.dataclass
class SequenceState:
    """Runtime execution progress for one admitted Request.

    ``target_output_len`` is the number of tokens this sequence will
    actually generate before completing -- always <= request.max_new_tokens.
    A real model stops early on an EOS token; Phase 0-1 has no model, so
    the trace generator (``minipaged.simulation.trace``) decides
    ``target_output_len`` up front as a stand-in for "where EOS would have
    landed". This is exactly what lets Phase 1's contiguous baseline show
    waste even for *completed* sequences: capacity is reserved for
    max_new_tokens but only target_output_len tokens are ever produced.
    """

    request: Request
    target_output_len: int
    tokens_generated: int = 0
    prefill_done: bool = False
    started_at: float | None = None
    completed_at: float | None = None

    def __post_init__(self) -> None:
        if self.target_output_len < 1:
            raise ValueError(f"target_output_len must be >= 1, got {self.target_output_len}")
        if self.target_output_len > self.request.max_new_tokens:
            raise ValueError(
                f"target_output_len ({self.target_output_len}) cannot exceed "
                f"request.max_new_tokens ({self.request.max_new_tokens})"
            )

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def current_len(self) -> int:
        """Prompt length + tokens generated so far -- the sequence's current KV length."""
        return self.request.prompt_len + self.tokens_generated

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.target_output_len

    def step_decode(self) -> None:
        """Advance by exactly one simulated decode tick."""
        if self.is_finished:
            raise RuntimeError(f"{self.request_id}: step_decode called after completion")
        self.tokens_generated += 1
