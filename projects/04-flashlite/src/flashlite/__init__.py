"""FlashLite: from naive attention to IO-aware fused attention.

Phase 0 exposes the trusted PyTorch reference (:mod:`flashlite.reference`)
plus the shared correctness/benchmark infrastructure (:mod:`flashlite.compare`,
:mod:`flashlite.bench_schema`, :mod:`flashlite.stats`, :mod:`flashlite.timing`,
:mod:`flashlite.env_capture`) that every later CUDA variant is checked and
measured against.

Phase 1 adds the naive CUDA kernel (``flashlite._cuda_naive``, built from
``src/flashlite/cuda_naive/``) and the ``flashlite.ops.attention`` dispatcher
that selects a variant by name.
"""

from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import DEFAULT_SEED, make_qkv

__all__ = [
    "DEFAULT_SEED",
    "make_qkv",
    "reference_attention",
]

__version__ = "0.1.0"
