"""Unit tests for forgelm.evaluation.report (FR9's "sample-report format"
deliverable)."""

from __future__ import annotations

import json
from pathlib import Path

from forgelm.config import GenerationSettings
from forgelm.evaluation.report import GenerationSample, SampleReport, render_markdown, save_report


def _sample_report() -> SampleReport:
    return SampleReport(
        checkpoint_path="artifacts/checkpoints/final.pt",
        step=100,
        model_config={
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "context_length": 128,
            "vocab_size": 512,
        },
        eval_result={"loss": 2.5, "perplexity": 12.18, "num_batches": 3, "tokens_evaluated": 384},
        samples=[
            GenerationSample(
                prompt="Once upon a time",
                generated_text="Once upon a time there was a curious fox.",
                settings=GenerationSettings(max_new_tokens=20, do_sample=False),
            ),
            GenerationSample(
                prompt="The rabbit",
                generated_text="The rabbit hurried down the hole.",
                settings=GenerationSettings(
                    max_new_tokens=20, do_sample=True, temperature=0.8, top_k=40, seed=7
                ),
            ),
        ],
    )


def test_save_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, md_path = save_report(report, tmp_path / "report")
    assert json_path == tmp_path / "report.json"
    assert md_path == tmp_path / "report.md"
    assert json_path.exists()
    assert md_path.exists()


def test_save_report_json_round_trips_every_field(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, _md_path = save_report(report, tmp_path / "report")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["checkpoint_path"] == report.checkpoint_path
    assert payload["step"] == report.step
    assert payload["eval_result"] == report.eval_result
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["prompt"] == "Once upon a time"
    assert payload["samples"][0]["settings"]["do_sample"] is False
    assert payload["samples"][1]["settings"]["top_k"] == 40


def test_render_markdown_documents_decoding_settings() -> None:
    report = _sample_report()
    md = render_markdown(report)
    assert "greedy" in md
    assert "sampling" in md
    assert "top_k=40" in md
    assert "temperature=0.8" in md
    assert "Once upon a time" in md
    assert "The rabbit hurried down the hole." in md


def test_render_markdown_handles_no_eval_result() -> None:
    report = SampleReport(
        checkpoint_path="ckpt.pt",
        step=1,
        model_config={"d_model": 16},
        eval_result=None,
        samples=[
            GenerationSample(prompt="hi", generated_text="hi there", settings=GenerationSettings())
        ],
    )
    md = render_markdown(report)
    assert "## Evaluation" not in md
    assert "## Generations" in md


def test_save_report_replaces_any_existing_suffix(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, md_path = save_report(report, tmp_path / "nested" / "report.yaml")
    assert json_path == tmp_path / "nested" / "report.json"
    assert md_path == tmp_path / "nested" / "report.md"


def test_save_report_creates_parent_directories(tmp_path: Path) -> None:
    report = _sample_report()
    save_report(report, tmp_path / "a" / "b" / "c" / "report")
    assert (tmp_path / "a" / "b" / "c" / "report.json").exists()
