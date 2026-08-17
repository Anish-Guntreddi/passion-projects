"""Exercises `forgelm tokenizer ...` / `forgelm dataset ...` as real subprocess
CLI invocations (FR1: "all execution is CLI + config driven")."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI = [sys.executable, "-m", "forgelm.cli.main"]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=120)


def test_cli_smoke_reads_yaml_config_file() -> None:
    """`forgelm smoke --config configs/smoke.yaml` loads via forgelm.config.load_config."""
    proc = subprocess.run(
        CLI + ["smoke", "--config", "configs/smoke.yaml"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result == {
        "seed": 1337,
        "size": 64,
        "device": "cpu",
        "checksum": result["checksum"],
        "torch_version": result["torch_version"],
        "cuda_available": result["cuda_available"],
    }


def test_cli_tokenizer_train_encode_decode_round_trip(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("the quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8")
    tokenizer_path = tmp_path / "tok.json"

    train_proc = _run(
        "tokenizer",
        "train",
        "--input",
        str(corpus_path),
        "--output",
        str(tokenizer_path),
        "--vocab-size",
        "280",
    )
    assert train_proc.returncode == 0, train_proc.stderr
    train_result = json.loads(train_proc.stdout)
    assert train_result["vocab_size"] == 280
    assert tokenizer_path.exists()

    text = "the quick brown fox"
    encode_proc = _run("tokenizer", "encode", "--model", str(tokenizer_path), "--text", text)
    assert encode_proc.returncode == 0, encode_proc.stderr
    ids = json.loads(encode_proc.stdout)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)

    decode_proc = _run(
        "tokenizer", "decode", "--model", str(tokenizer_path), "--ids", ",".join(map(str, ids))
    )
    assert decode_proc.returncode == 0, decode_proc.stderr
    assert decode_proc.stdout.strip() == text


def test_cli_dataset_build(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "once upon a time in a small village there lived a curious fox. " * 30,
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tok.json"
    train_proc = _run(
        "tokenizer",
        "train",
        "--input",
        str(corpus_path),
        "--output",
        str(tokenizer_path),
        "--vocab-size",
        "300",
    )
    assert train_proc.returncode == 0, train_proc.stderr

    out_dir = tmp_path / "built"
    build_proc = _run(
        "dataset",
        "build",
        "--input",
        str(corpus_path),
        "--tokenizer",
        str(tokenizer_path),
        "--output",
        str(out_dir),
        "--context-length",
        "16",
        "--val-fraction",
        "0.2",
    )
    assert build_proc.returncode == 0, build_proc.stderr
    result = json.loads(build_proc.stdout)
    assert result["train_tokens"] + result["val_tokens"] == result["num_tokens"]
    assert (out_dir / "train_tokens.npy").exists()
    assert (out_dir / "val_tokens.npy").exists()
    assert (out_dir / "stats.json").exists()


def test_cli_tokenizer_train_and_dataset_build_from_config_alone_no_output_flag(
    tmp_path: Path,
) -> None:
    """FR1: `--config` alone (no `--output`) must fully reproduce an
    artifact-producing run, exercising configs/tokenizer_train.yaml and
    configs/dataset_build.yaml's exact shape against a small local corpus."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("once upon a time there was a small fox. " * 40, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    built_dir = tmp_path / "built"

    tok_config_path = tmp_path / "tokenizer_train.yaml"
    tok_config_path.write_text(
        f"input_path: {corpus_path}\noutput_path: {tok_path}\nvocab_size: 300\n"
    )
    train_proc = _run("tokenizer", "train", "--config", str(tok_config_path))
    assert train_proc.returncode == 0, train_proc.stderr
    assert tok_path.exists()

    ds_config_path = tmp_path / "dataset_build.yaml"
    ds_config_path.write_text(
        f"input_path: {corpus_path}\n"
        f"tokenizer_path: {tok_path}\n"
        f"output_dir: {built_dir}\n"
        "context_length: 16\n"
    )
    build_proc = _run("dataset", "build", "--config", str(ds_config_path))
    assert build_proc.returncode == 0, build_proc.stderr
    assert (built_dir / "train_tokens.npy").exists()
    assert (built_dir / "stats.json").exists()


def test_cli_tokenizer_train_without_config_or_output_reports_clean_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("hello world", encoding="utf-8")
    proc = _run("tokenizer", "train", "--input", str(corpus_path))
    assert proc.returncode != 0
    assert "output" in proc.stderr.lower()


def test_cli_reports_a_clean_error_for_bad_config(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("hi", encoding="utf-8")
    proc = _run(
        "tokenizer",
        "train",
        "--input",
        str(corpus_path),
        "--output",
        str(tmp_path / "tok.json"),
        "--vocab-size",
        "10",
    )
    assert proc.returncode != 0
    assert "vocab_size" in proc.stderr


# -- Phase 3: `forgelm train` -------------------------------------------------------


def _train_config_yaml(
    *,
    train_tokens_path: Path,
    val_tokens_path: Path,
    output_dir: Path,
    resume_from: Path | None = None,
) -> str:
    text = (
        "model:\n"
        "  vocab_size: 280\n"
        "  context_length: 16\n"
        "  d_model: 16\n"
        "  n_layers: 1\n"
        "  n_heads: 2\n"
        "  d_ff: 32\n"
        "training:\n"
        "  micro_batch_size: 4\n"
        "  max_steps: 6\n"
        "  warmup_steps: 1\n"
        "  eval_interval: 3\n"
        "  eval_batches: 1\n"
        "  device: cpu\n"
        "  precision: fp32\n"
        f"train_tokens_path: {train_tokens_path}\n"
        f"val_tokens_path: {val_tokens_path}\n"
        f"output_dir: {output_dir}\n"
        "checkpoint_interval: 3\n"
    )
    if resume_from is not None:
        text += f"resume_from: {resume_from}\n"
    return text


def test_cli_train_end_to_end_writes_checkpoints_and_resumes(tmp_path: Path) -> None:
    """FR1/FR6/FR7 via the real CLI: train from a config-built dataset,
    write periodic + final checkpoints, then resume from a mid-run
    checkpoint and continue to the same max_steps."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("once upon a time there was a small fox. " * 60, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    train_tok_proc = _run(
        "tokenizer",
        "train",
        "--input",
        str(corpus_path),
        "--output",
        str(tok_path),
        "--vocab-size",
        "280",
    )
    assert train_tok_proc.returncode == 0, train_tok_proc.stderr

    dataset_dir = tmp_path / "dataset"
    build_proc = _run(
        "dataset",
        "build",
        "--input",
        str(corpus_path),
        "--tokenizer",
        str(tok_path),
        "--output",
        str(dataset_dir),
        "--context-length",
        "16",
        "--val-fraction",
        "0.2",
    )
    assert build_proc.returncode == 0, build_proc.stderr

    ckpt_dir = tmp_path / "checkpoints"
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text(
        _train_config_yaml(
            train_tokens_path=dataset_dir / "train_tokens.npy",
            val_tokens_path=dataset_dir / "val_tokens.npy",
            output_dir=ckpt_dir,
        )
    )

    train_proc = _run("train", "--config", str(train_config_path))
    assert train_proc.returncode == 0, train_proc.stderr
    result = json.loads(train_proc.stdout)
    assert result["start_step"] == 0
    assert result["final_step"] == 6
    assert (ckpt_dir / "final.pt").exists()
    assert (ckpt_dir / "step_3.pt").exists()

    # FR6 "logging": the full per-step loss/lr/grad_norm (+ periodic
    # val_loss) trajectory must be persisted, not just the single final
    # summary `result` above.
    metrics_records = [
        json.loads(line)
        for line in (ckpt_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step_records = [r for r in metrics_records if r["kind"] == "step"]
    eval_records = [r for r in metrics_records if r["kind"] == "eval"]
    assert [r["step"] for r in step_records] == [1, 2, 3, 4, 5, 6]
    assert all({"loss", "lr", "grad_norm"} <= r.keys() for r in step_records)
    assert [r["step"] for r in eval_records] == [3, 6]  # eval_interval=3 + final step

    # Resume from the mid-run checkpoint with the identical config
    # (config_hash must match) -> continues from step 3 to step 6.
    resume_config_path = tmp_path / "train_resume.yaml"
    resume_config_path.write_text(
        _train_config_yaml(
            train_tokens_path=dataset_dir / "train_tokens.npy",
            val_tokens_path=dataset_dir / "val_tokens.npy",
            output_dir=ckpt_dir,
            resume_from=ckpt_dir / "step_3.pt",
        )
    )
    resume_proc = _run("train", "--config", str(resume_config_path))
    assert resume_proc.returncode == 0, resume_proc.stderr
    resume_result = json.loads(resume_proc.stdout)
    assert resume_result["start_step"] == 3
    assert resume_result["final_step"] == 6


def test_cli_train_resume_against_a_different_dataset_raises_a_clean_error(
    tmp_path: Path,
) -> None:
    """FR7: resuming a checkpoint against a *different* dataset than it was
    produced with (e.g. --resume-from a checkpoint while the config's
    train/val token paths now point at a regenerated/different corpus)
    must fail loudly via config_hash, not silently train on a different
    token stream. See forgelm.training.checkpoint.config_hash."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("once upon a time there was a small fox. " * 60, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    assert (
        _run(
            "tokenizer",
            "train",
            "--input",
            str(corpus_path),
            "--output",
            str(tok_path),
            "--vocab-size",
            "280",
        ).returncode
        == 0
    )

    def _build_dataset(corpus: Path, output: Path) -> None:
        proc = _run(
            "dataset",
            "build",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tok_path),
            "--output",
            str(output),
            "--context-length",
            "16",
            "--val-fraction",
            "0.2",
        )
        assert proc.returncode == 0, proc.stderr

    dataset_dir = tmp_path / "dataset"
    _build_dataset(corpus_path, dataset_dir)

    other_corpus_path = tmp_path / "other_corpus.txt"
    other_corpus_path.write_text(
        "a completely different sentence about something else entirely. " * 60, encoding="utf-8"
    )
    other_dataset_dir = tmp_path / "other_dataset"
    _build_dataset(other_corpus_path, other_dataset_dir)

    ckpt_dir = tmp_path / "checkpoints"
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text(
        _train_config_yaml(
            train_tokens_path=dataset_dir / "train_tokens.npy",
            val_tokens_path=dataset_dir / "val_tokens.npy",
            output_dir=ckpt_dir,
        )
    )
    train_proc = _run("train", "--config", str(train_config_path))
    assert train_proc.returncode == 0, train_proc.stderr

    # Resume from the checkpoint but point the config at the *other*
    # dataset's token arrays -- same model/training config otherwise.
    resume_config_path = tmp_path / "train_resume_mismatch.yaml"
    resume_config_path.write_text(
        _train_config_yaml(
            train_tokens_path=other_dataset_dir / "train_tokens.npy",
            val_tokens_path=other_dataset_dir / "val_tokens.npy",
            output_dir=ckpt_dir,
            resume_from=ckpt_dir / "step_3.pt",
        )
    )
    resume_proc = _run("train", "--config", str(resume_config_path))
    assert resume_proc.returncode != 0
    assert "config_hash" in resume_proc.stderr


def test_cli_train_without_config_or_output_reports_clean_error(tmp_path: Path) -> None:
    config_path = tmp_path / "no_output.yaml"
    config_path.write_text(
        "model:\n  vocab_size: 280\n"
        "training:\n  micro_batch_size: 4\n  max_steps: 2\n"
        "train_tokens_path: does_not_exist_train.npy\n"
        "val_tokens_path: does_not_exist_val.npy\n"
    )
    proc = _run("train", "--config", str(config_path))
    assert proc.returncode != 0
    assert "output" in proc.stderr.lower()


# -- Phase 4: `forgelm generate` / `forgelm evaluate` / `forgelm sample-report` ----


def _build_tokenizer_dataset_and_checkpoint(tmp_path: Path) -> dict[str, Path]:
    """Shared fixture-equivalent for the Phase 4 CLI tests below: a real
    tokenizer, a real built dataset, and a real trained checkpoint, all
    produced via actual `forgelm` subprocess invocations (not shortcuts),
    matching this file's existing "exercise the real CLI" convention."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("once upon a time there was a small fox. " * 60, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    assert (
        _run(
            "tokenizer",
            "train",
            "--input",
            str(corpus_path),
            "--output",
            str(tok_path),
            "--vocab-size",
            "280",
        ).returncode
        == 0
    )

    dataset_dir = tmp_path / "dataset"
    assert (
        _run(
            "dataset",
            "build",
            "--input",
            str(corpus_path),
            "--tokenizer",
            str(tok_path),
            "--output",
            str(dataset_dir),
            "--context-length",
            "16",
            "--val-fraction",
            "0.2",
        ).returncode
        == 0
    )

    ckpt_dir = tmp_path / "checkpoints"
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text(
        _train_config_yaml(
            train_tokens_path=dataset_dir / "train_tokens.npy",
            val_tokens_path=dataset_dir / "val_tokens.npy",
            output_dir=ckpt_dir,
        )
    )
    train_proc = _run("train", "--config", str(train_config_path))
    assert train_proc.returncode == 0, train_proc.stderr

    return {
        "tokenizer_path": tok_path,
        "train_tokens_path": dataset_dir / "train_tokens.npy",
        "val_tokens_path": dataset_dir / "val_tokens.npy",
        "checkpoint_path": ckpt_dir / "final.pt",
    }


def test_cli_generate_produces_text_continuing_the_prompt(tmp_path: Path) -> None:
    artifacts = _build_tokenizer_dataset_and_checkpoint(tmp_path)
    proc = _run(
        "generate",
        "--checkpoint",
        str(artifacts["checkpoint_path"]),
        "--tokenizer",
        str(artifacts["tokenizer_path"]),
        "--prompt",
        "once upon a time",
        "--max-new-tokens",
        "10",
        "--greedy",
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["prompt"] == "once upon a time"
    assert isinstance(result["generated_text"], str) and result["generated_text"]
    assert result["generated_text"].startswith("once upon a time")


def test_cli_generate_is_deterministic_with_greedy_decoding(tmp_path: Path) -> None:
    artifacts = _build_tokenizer_dataset_and_checkpoint(tmp_path)
    args = [
        "generate",
        "--checkpoint",
        str(artifacts["checkpoint_path"]),
        "--tokenizer",
        str(artifacts["tokenizer_path"]),
        "--prompt",
        "the fox",
        "--max-new-tokens",
        "8",
        "--greedy",
    ]
    result_a = json.loads(_run(*args).stdout)
    result_b = json.loads(_run(*args).stdout)
    assert result_a["generated_text"] == result_b["generated_text"]


def test_cli_evaluate_reports_loss_and_perplexity(tmp_path: Path) -> None:
    artifacts = _build_tokenizer_dataset_and_checkpoint(tmp_path)
    proc = _run(
        "evaluate",
        "--checkpoint",
        str(artifacts["checkpoint_path"]),
        "--tokens",
        str(artifacts["val_tokens_path"]),
        "--batch-size",
        "2",
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["loss"] > 0
    assert result["perplexity"] > 1.0
    assert result["tokens_evaluated"] > 0


def test_cli_evaluate_respects_max_tokens_budget(tmp_path: Path) -> None:
    artifacts = _build_tokenizer_dataset_and_checkpoint(tmp_path)
    proc = _run(
        "evaluate",
        "--checkpoint",
        str(artifacts["checkpoint_path"]),
        "--tokens",
        str(artifacts["val_tokens_path"]),
        "--batch-size",
        "2",
        "--max-tokens",
        "32",  # 2 * context_length(16) = one batch
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["tokens_evaluated"] == 32
    assert result["num_batches"] == 1


def test_cli_sample_report_writes_json_and_markdown(tmp_path: Path) -> None:
    artifacts = _build_tokenizer_dataset_and_checkpoint(tmp_path)
    output_base = tmp_path / "report"
    config_path = tmp_path / "sample_report.yaml"
    config_path.write_text(
        f"checkpoint_path: {artifacts['checkpoint_path']}\n"
        f"tokenizer_path: {artifacts['tokenizer_path']}\n"
        f"output_path: {output_base}\n"
        f"val_tokens_path: {artifacts['val_tokens_path']}\n"
        "prompts:\n"
        '  - "once upon a time"\n'
        '  - "the fox"\n'
        "generation:\n"
        "  max_new_tokens: 8\n"
        "  do_sample: false\n"
        "eval_batch_size: 2\n"
    )
    proc = _run("sample-report", "--config", str(config_path))
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["num_samples"] == 2
    assert result["eval_result"]["loss"] > 0

    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    assert json_path.exists()
    assert md_path.exists()
    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(report_payload["samples"]) == 2
    assert "## Generations" in md_path.read_text(encoding="utf-8")


def test_cli_generate_without_config_or_flags_reports_clean_error(tmp_path: Path) -> None:
    proc = _run("generate", "--prompt", "hi")
    assert proc.returncode != 0
    assert "checkpoint" in proc.stderr.lower() or "tokenizer" in proc.stderr.lower()


# -- Phase 5: `forgelm benchmark run` -----------------------------------------------


def test_cli_benchmark_run_writes_a_result_json(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("once upon a time there was a small fox. " * 60, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    assert (
        _run(
            "tokenizer",
            "train",
            "--input",
            str(corpus_path),
            "--output",
            str(tok_path),
            "--vocab-size",
            "280",
        ).returncode
        == 0
    )
    dataset_dir = tmp_path / "dataset"
    assert (
        _run(
            "dataset",
            "build",
            "--input",
            str(corpus_path),
            "--tokenizer",
            str(tok_path),
            "--output",
            str(dataset_dir),
            "--context-length",
            "16",
            "--val-fraction",
            "0.2",
        ).returncode
        == 0
    )

    output_path = tmp_path / "bench_result.json"
    bench_config_path = tmp_path / "bench.yaml"
    bench_config_path.write_text(
        "name: cli-smoke\n"
        "model:\n"
        "  vocab_size: 280\n"
        "  context_length: 16\n"
        "  d_model: 16\n"
        "  n_layers: 1\n"
        "  n_heads: 2\n"
        "  d_ff: 32\n"
        "training:\n"
        "  micro_batch_size: 4\n"
        "  max_steps: 1000\n"
        "  warmup_steps: 1\n"
        "  eval_interval: 1000\n"
        "  eval_batches: 1\n"
        "  device: cpu\n"
        "  precision: fp32\n"
        f"train_tokens_path: {dataset_dir / 'train_tokens.npy'}\n"
        f"val_tokens_path: {dataset_dir / 'val_tokens.npy'}\n"
        "bench_warmup_steps: 1\n"
        "measured_steps: 3\n"
    )

    proc = _run(
        "benchmark", "run", "--config", str(bench_config_path), "--output", str(output_path)
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["name"] == "cli-smoke"
    assert result["measured_steps"] == 3
    assert result["tokens_per_sec"] > 0
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["name"] == "cli-smoke"
