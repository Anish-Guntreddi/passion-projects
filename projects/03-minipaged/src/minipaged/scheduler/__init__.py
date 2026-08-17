"""Continuous-batching scheduler -- boundary placeholder.

Not implemented yet. Phase 3 of the roadmap adds admission control tied to
KV capacity, per-step token/sequence budgets, prefill-vs-decode work
selection, and the ``SchedulerDecision`` interface (FR9). Phase 1's
``ContiguousKVEngine._try_admit`` (``minipaged.kv.contiguous``) is a
deliberately simple stand-in admission policy for the *contiguous*
baseline only -- it is not this module and will not be reused unchanged
once a real scheduler exists.
"""

from __future__ import annotations
