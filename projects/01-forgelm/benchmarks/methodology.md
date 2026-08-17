# Benchmark methodology

**Status: no benchmarks have been run yet.** This document will describe
the Phase 5 scaling-experiment methodology (tokens/sec vs. model size, peak
memory vs. context length / batch size, val loss vs. training tokens per
config, optional `torch.compile` A/B) once Phase 3 (training system) and
Phase 5 (scaling experiment) are implemented.

Portfolio-wide hard rule: **no benchmark or performance number is ever
fabricated.** Every number that eventually appears here or in
`benchmarks/results/` will be measured output from a real run on this
project's actual hardware, committed as a raw artifact alongside:

- hardware (CPU, GPU model + VRAM),
- software versions (Python, PyTorch, CUDA driver — see
  `requirements-lock.txt` once generated),
- dtype / precision mode used,
- random seed,
- exact model config,
- dataset identity + token count,
- warmup policy and sample count.

`benchmarks/results/` and `benchmarks/plots/` are currently empty
(placeholder directories only) — there is nothing to report until Phase 5.
