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
