"""Phase 0 deliverable: 'benchmark result schema' (spec Part 3). Every
record this project ever commits under benchmarks/raw/ must validate
against benchmarks/schema/bench_result.schema.json -- this file checks the
Python producer (flashlite.bench_schema) and the JSON Schema document stay
in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from flashlite.bench_schema import (
    BenchResult,
    append_jsonl,
    finalize,
    to_json,
    to_json_dict,
)
from flashlite.env_capture import EnvironmentInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "schema" / "bench_result.schema.json"


def _make_result(**overrides) -> BenchResult:
    env = EnvironmentInfo(
        gpu_name="NVIDIA GeForce RTX 4090",
        compute_capability="8.9",
        cuda_driver_version="12.6",
        cuda_runtime_version="12.4",
        python_version="3.12.3",
        torch_version="2.6.0+cu124",
        locked_clocks=False,
        clock_lock_note="not locked (WSL2)",
        observed_sm_clock_mhz=2520,
        observed_mem_clock_mhz=10501,
        os_note="Linux (WSL2 Ubuntu) guest on Windows 11 host",
        timestamp_utc="2026-08-17T00:00:00Z",
    )
    result = BenchResult(
        variant="v0_reference",
        description="test result",
        env=env,
        batch=2,
        heads=4,
        seq_len=128,
        head_dim=64,
        causal=False,
        dtype="fp32",
        warmup_iters=5,
        measured_reps=20,
        seed=123456789,
        raw_timings_ms=[1.0 + 0.01 * i for i in range(20)],
        bytes_moved=1_000_000.0,
        gflops=0.0,
        tolerance_atol=1e-5,
        tolerance_rtol=1e-4,
        correctness_passed=True,
        correctness_note="matches reference within tolerance",
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def test_schema_file_exists_and_is_valid_json_schema() -> None:
    assert SCHEMA_PATH.exists(), f"missing {SCHEMA_PATH}"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft7Validator.check_schema(schema)


def test_finalize_computes_stats_and_bandwidth() -> None:
    result = _make_result()
    finalize(result)
    assert result.stats is not None
    assert result.stats.median_ms > 0
    assert result.effective_bandwidth_gb_s > 0


def test_finalize_requires_nonempty_raw_timings() -> None:
    result = _make_result(raw_timings_ms=[])
    with pytest.raises(ValueError):
        finalize(result)


def test_to_json_dict_requires_finalize_first() -> None:
    result = _make_result()
    with pytest.raises(ValueError, match="finalize"):
        to_json_dict(result)


def test_committed_record_validates_against_schema() -> None:
    result = _make_result()
    finalize(result)
    doc = to_json_dict(result)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=doc, schema=schema)


def test_correctness_failed_record_fails_schema_validation() -> None:
    """The schema requires correctness_passed == true (a result with a
    known correctness failure must never be committed to benchmarks/raw/).
    """
    result = _make_result(correctness_passed=False)
    finalize(result)
    doc = to_json_dict(result)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=doc, schema=schema)


def test_measured_reps_below_twenty_fails_schema_validation() -> None:
    result = _make_result(measured_reps=5, raw_timings_ms=[1.0] * 5)
    finalize(result)
    doc = to_json_dict(result)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=doc, schema=schema)


def test_to_json_round_trips_through_json_loads() -> None:
    result = _make_result()
    finalize(result)
    parsed = json.loads(to_json(result))
    assert parsed["variant"] == "v0_reference"
    assert parsed["schema_version"] == "1.0"
    assert parsed["kernel_family"] == "attention"
    assert len(parsed["raw_timings_ms"]) == 20


def test_tile_size_defaults_to_zero_and_validates() -> None:
    """ADR 0008: tile_size defaults to 0 ("not applicable") for variants
    with no tile-size concept (v0_reference, v1_naive) and must still
    validate against the schema -- the field is optional/additive, not
    required.
    """
    result = _make_result()
    assert result.tile_size == 0
    finalize(result)
    doc = to_json_dict(result)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=doc, schema=schema)


def test_tile_size_v2_tiled_record_validates() -> None:
    """ADR 0007/0008: a v2_tiled record with a populated tile_size (the
    kAttnTileDim launch-config constant) validates against the schema too.
    """
    result = _make_result(variant="v2_tiled", tile_size=32)
    finalize(result)
    doc = to_json_dict(result)
    assert doc["tile_size"] == 32
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=doc, schema=schema)


def test_append_jsonl_appends_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "attention.jsonl"
    result = _make_result()
    finalize(result)

    append_jsonl(str(path), result)
    append_jsonl(str(path), result)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert parsed["kernel_family"] == "attention"
