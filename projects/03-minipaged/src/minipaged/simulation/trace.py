"""Deterministic synthetic arrival-trace generation.

Produces ``TraceRequestSpec`` sequences for feeding into a
``SimulationEngine`` (Phase 0) or ``ContiguousKVEngine`` (Phase 1). Every
distribution draw goes through a single seeded ``random.Random`` instance,
so the same ``(config, seed)`` pair always reproduces the identical trace
-- this is what core invariant #7 ("deterministic simulated traces
reproduce identical allocation history under fixed seed/config") tests.
"""

from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True)
class TraceRequestSpec:
    """One synthetic request, fully specified before the engine sees it.

    ``target_output_len`` stands in for "where this sequence's EOS would
    land" -- see ``SequenceState``'s docstring. It is always
    ``<= max_new_tokens``.
    """

    prompt_len: int
    max_new_tokens: int
    target_output_len: int
    arrival_time: float

    def __post_init__(self) -> None:
        if self.prompt_len < 1:
            raise ValueError(f"prompt_len must be >= 1, got {self.prompt_len}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not (1 <= self.target_output_len <= self.max_new_tokens):
            raise ValueError(
                f"target_output_len ({self.target_output_len}) must be in "
                f"[1, max_new_tokens={self.max_new_tokens}]"
            )
        if self.arrival_time < 0:
            raise ValueError(f"arrival_time must be >= 0, got {self.arrival_time}")


@dataclasses.dataclass(frozen=True)
class TraceConfig:
    """Parameters of a synthetic trace.

    See ``docs/kv-memory.md`` for how ``prompt_len_range`` /
    ``max_new_tokens_range`` / ``completion_fraction_range`` drive Phase
    1's fragmentation demo: a wide length range plus early completion
    (``target_output_len < max_new_tokens``) is what makes a contiguous
    allocator's reserved-vs-used gap measurable.
    """

    num_requests: int = 20
    mean_interarrival: float = 1.0
    prompt_len_range: tuple[int, int] = (8, 64)
    max_new_tokens_range: tuple[int, int] = (8, 128)
    # Fraction of max_new_tokens a sequence actually uses before "EOS",
    # sampled uniformly from this range -- 1.0 means "always uses the full
    # budget" (no early completion, no waste-at-completion).
    completion_fraction_range: tuple[float, float] = (0.4, 1.0)

    def __post_init__(self) -> None:
        if self.num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {self.num_requests}")
        if self.mean_interarrival < 0:
            raise ValueError(f"mean_interarrival must be >= 0, got {self.mean_interarrival}")
        for name, rng in (
            ("prompt_len_range", self.prompt_len_range),
            ("max_new_tokens_range", self.max_new_tokens_range),
        ):
            lo, hi = rng
            if lo < 1 or hi < lo:
                raise ValueError(f"{name} must satisfy 1 <= lo <= hi, got {rng}")
        clo, chi = self.completion_fraction_range
        if not (0.0 < clo <= chi <= 1.0):
            raise ValueError(
                f"completion_fraction_range must satisfy 0 < lo <= hi <= 1, got {(clo, chi)}"
            )


def generate_trace(config: TraceConfig, seed: int) -> list[TraceRequestSpec]:
    """Deterministically generate a variable-length arrival trace.

    Arrivals follow a Poisson process (exponential interarrival times)
    with mean ``config.mean_interarrival``; prompt length and
    max_new_tokens are drawn uniformly (integers) from their configured
    ranges; ``target_output_len`` is ``max_new_tokens`` scaled by a
    uniformly sampled completion fraction, rounded and clamped to
    ``>= 1``. Everything is driven by one ``random.Random(seed)``
    instance, so ``(config, seed)`` fully determines the trace -- calling
    this twice with the same arguments returns equal (by value) results.
    """
    rng = random.Random(seed)
    specs: list[TraceRequestSpec] = []
    t = 0.0
    for _ in range(config.num_requests):
        if config.mean_interarrival > 0:
            t += rng.expovariate(1.0 / config.mean_interarrival)
        prompt_len = rng.randint(*config.prompt_len_range)
        max_new_tokens = rng.randint(*config.max_new_tokens_range)
        fraction = rng.uniform(*config.completion_fraction_range)
        target_output_len = max(1, min(max_new_tokens, round(max_new_tokens * fraction)))
        specs.append(
            TraceRequestSpec(
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                target_output_len=target_output_len,
                arrival_time=t,
            )
        )
    return specs
