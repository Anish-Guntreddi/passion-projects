"""Continuous-batching scheduler (Phase 3, FR4/FR9).

``SchedulerEngine`` (``scheduler.py``) subclasses Phase 2's
``PagedKVEngine`` and adds a per-step token budget (D3/D4) plus a
block-quantized worst-case admission ledger (D5, and what makes Phase
2's documented deadlock limitation unreachable once a request is
admitted -- see ``docs/decisions/0005-scheduler-token-budget-and-admission-ledger.md``).
``SchedulerDecision`` (``decision.py``) is FR9's public interface: one
per ``step()`` call, accumulated in ``SchedulerEngine.decisions`` as the
"scheduling timeline" that is this phase's exit criterion.

Phase 1's ``ContiguousKVEngine._try_admit``
(``minipaged.kv.contiguous``) and Phase 2's ``PagedKVEngine._try_admit``
(``minipaged.kv.manager``) remain deliberately simple stand-in admission
policies for their own allocators -- neither is this module, and neither
is reused unchanged here (``SchedulerEngine`` overrides admission
entirely rather than composing theirs).
"""

from __future__ import annotations

from minipaged.scheduler.decision import SchedulerDecision
from minipaged.scheduler.scheduler import SchedulerEngine

__all__ = ["SchedulerDecision", "SchedulerEngine"]
