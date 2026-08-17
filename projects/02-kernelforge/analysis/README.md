# Analysis Pipeline

**Status: not yet populated.** Per spec FR7, this directory will hold the
Python (pandas/matplotlib) pipeline that turns committed
`benchmarks/raw/*.jsonl` records into plots under `benchmarks/plots/` —
this is a Phase 7 ("Portfolio release") deliverable. CUDA implementation
stays C++ throughout (this directory never contains kernel code).

Not needed for Phases 0-1: `benchmarks/methodology.md` presents the
Phase 1 results as tables read directly from the committed `.jsonl` files
(no plotting library required to state them honestly).

## Why nothing is here yet

`pandas` and `matplotlib` are not installed on this development machine
(verified 2026-08-17: `python3 -c "import pandas"` /
`import matplotlib` both raise `ModuleNotFoundError`), and installing
them is out of scope for Phases 0-1's exit criteria, which do not require
any plot. `requirements.txt` below is committed now so the Phase 7 setup
step is a single `pip install -r analysis/requirements.txt` rather than a
guess.

## Reading the data without this pipeline, today

Every `benchmarks/raw/*.jsonl` file is directly loadable with the
Python standard library alone:

```python
import json
records = [json.loads(line) for line in open("benchmarks/raw/transpose.jsonl")]
```

or, once Phase 7 installs the dependencies below:

```python
import pandas as pd
df = pd.read_json("benchmarks/raw/transpose.jsonl", lines=True)
```
