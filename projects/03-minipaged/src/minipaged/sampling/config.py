"""Sampling configuration carried by every Request.

Phase 0-1 note: this is a data-only interface. No sampling *logic*
(applying temperature/top-k/top-p to real model logits) exists yet --
that lands in Phase 5 once a real model is wired in (FR1). Defining the
shape now lets ``Request`` (``minipaged.requests.request``) carry a
``sampling_config`` field per FR2 without prejudging Phase 5's
implementation, and without accepting nonsensical values (e.g. a negative
temperature) even in a simulation that never actually samples.
"""

from __future__ import annotations

import dataclasses


class SamplingConfigError(ValueError):
    """Raised when a SamplingConfig is constructed with invalid parameters."""


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Parameters that will govern token sampling once a real model exists.

    Defaults reproduce "always take the model's exact distribution at
    temperature 1, no truncation" -- the least surprising default for a
    config nothing consumes yet.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise SamplingConfigError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise SamplingConfigError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise SamplingConfigError(f"top_k must be >= 1 if set, got {self.top_k}")
