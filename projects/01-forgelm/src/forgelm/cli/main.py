"""Typer entry point. Composes the library modules; contains no model/training logic."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer

from forgelm.cli.benchmark_cli import run_benchmark_cli
from forgelm.cli.dataset_cli import build_dataset_artifacts
from forgelm.cli.evaluation_cli import build_sample_report, evaluate_checkpoint
from forgelm.cli.generation_cli import generate_text
from forgelm.cli.smoke import run_smoke
from forgelm.cli.tokenizer_cli import decode_ids, encode_text, train_tokenizer
from forgelm.cli.train_cli import run_training
from forgelm.config import (
    BenchmarkRunConfig,
    DatasetBuildConfig,
    EvaluateConfig,
    GenerateConfig,
    GenerationSettings,
    SampleReportConfig,
    SmokeConfig,
    TokenizerTrainConfig,
    TrainConfig,
    load_config,
)

app = typer.Typer(help="ForgeLM: a from-scratch decoder-only Transformer training stack.")
tokenizer_app = typer.Typer(help="Train/encode/decode with the byte-level BPE tokenizer.")
dataset_app = typer.Typer(help="Build token arrays + statistics from raw text.")
benchmark_app = typer.Typer(help="Throughput/memory/val-loss measurement runs (Phase 5).")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(dataset_app, name="dataset")
app.add_typer(benchmark_app, name="benchmark")

_CONFIG_HELP = "YAML config file (see configs/). Overrides the individual flags below."


@app.command()
def smoke(
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, dir_okay=False, help=_CONFIG_HELP
    ),
    seed: int = typer.Option(1337, help="RNG seed."),
    size: int = typer.Option(64, help="Square matrix size for the placeholder computation."),
    device: str = typer.Option("cpu", help="'cpu' or 'cuda'."),
) -> None:
    """Deterministic placeholder end-to-end command (Phase 0 exit criterion)."""
    config = (
        load_config(config_path, SmokeConfig)
        if config_path is not None
        else SmokeConfig(seed=seed, size=size, device=device)
    )
    result = run_smoke(config)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@tokenizer_app.command("train")
def tokenizer_train(
    input_path: Path | None = typer.Option(
        None, "--input", exists=True, dir_okay=False, help="Raw text file to train on."
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Where to save the trained tokenizer (JSON). Overrides the "
        "config file's 'output_path' when both are given; one of the two "
        "is required.",
    ),
    vocab_size: int = typer.Option(512, help="Target vocabulary size."),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, dir_okay=False, help=_CONFIG_HELP
    ),
) -> None:
    """Train a byte-level BPE tokenizer on a local text file."""
    if config_path is not None:
        config = load_config(config_path, TokenizerTrainConfig)
    else:
        if input_path is None:
            raise typer.BadParameter("either --config or --input is required")
        config = TokenizerTrainConfig(
            input_path=input_path, output_path=output_path, vocab_size=vocab_size
        )
    try:
        result = train_tokenizer(config, output_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@tokenizer_app.command("encode")
def tokenizer_encode(
    model_path: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    text: str = typer.Option(..., "--text"),
) -> None:
    """Encode text to a JSON list of token ids."""
    ids = encode_text(model_path, text)
    typer.echo(json.dumps(ids))


@tokenizer_app.command("decode")
def tokenizer_decode(
    model_path: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    ids: str = typer.Option(..., "--ids", help="Comma-separated token ids."),
) -> None:
    """Decode a comma-separated list of token ids back to text."""
    id_list = [int(x) for x in ids.split(",") if x.strip()]
    text = decode_ids(model_path, id_list)
    typer.echo(text)


@dataset_app.command("build")
def dataset_build(
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        help="Directory to write train/val token arrays + stats.json into. "
        "Overrides the config file's 'output_dir' when both are given; one "
        "of the two is required.",
    ),
    input_path: Path | None = typer.Option(None, "--input", exists=True, dir_okay=False),
    tokenizer_path: Path | None = typer.Option(None, "--tokenizer", exists=True, dir_okay=False),
    context_length: int = typer.Option(128, help="Fixed sequence length per training example."),
    val_fraction: float = typer.Option(0.1, help="Fraction of tokens held out for validation."),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, dir_okay=False, help=_CONFIG_HELP
    ),
) -> None:
    """Tokenize a text file, split it, and write train/val token arrays + stats."""
    if config_path is not None:
        config = load_config(config_path, DatasetBuildConfig)
    else:
        if input_path is None or tokenizer_path is None:
            raise typer.BadParameter("either --config or both --input and --tokenizer are required")
        config = DatasetBuildConfig(
            input_path=input_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            context_length=context_length,
            val_fraction=val_fraction,
        )
    try:
        result = build_dataset_artifacts(config, output_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def train(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        dir_okay=False,
        help="YAML config file mapping to forgelm.config.TrainConfig (required -- the "
        "nested model/training config is not exposed as individual flags).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        help="Directory to write checkpoints into. Overrides the config file's "
        "'output_dir' when both are given; one of the two is required.",
    ),
) -> None:
    """Train (or resume) a model end to end from pre-built token arrays (FR6/FR7)."""
    config = load_config(config_path, TrainConfig)
    try:
        result = run_training(config, output_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def generate(
    checkpoint_path: Path | None = typer.Option(
        None, "--checkpoint", exists=True, dir_okay=False, help="Trained checkpoint (.pt)."
    ),
    tokenizer_path: Path | None = typer.Option(
        None, "--tokenizer", exists=True, dir_okay=False, help="Trained tokenizer (.json)."
    ),
    prompt: str = typer.Option("", "--prompt", help="Prompt text to continue."),
    device: str = typer.Option("cpu", "--device", help="'cpu' or 'cuda'."),
    max_new_tokens: int = typer.Option(50, "--max-new-tokens"),
    temperature: float = typer.Option(1.0, "--temperature"),
    top_k: int = typer.Option(0, "--top-k", help="0 = disabled."),
    top_p: float = typer.Option(0.0, "--top-p", help="0 = disabled."),
    do_sample: bool = typer.Option(
        True, "--sample/--greedy", help="Sample vs. greedy argmax decoding."
    ),
    seed: int = typer.Option(1337, "--seed"),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, dir_okay=False, help=_CONFIG_HELP
    ),
) -> None:
    """Generate a continuation for a prompt from a trained checkpoint (FR8)."""
    if config_path is not None:
        config = load_config(config_path, GenerateConfig)
    else:
        if checkpoint_path is None or tokenizer_path is None:
            raise typer.BadParameter(
                "either --config or both --checkpoint and --tokenizer are required"
            )
        config = GenerateConfig(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            prompt=prompt,
            device=device,
            generation=GenerationSettings(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k or None,
                top_p=top_p or None,
                do_sample=do_sample,
                seed=seed,
            ),
        )
    try:
        result = generate_text(config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def evaluate(
    checkpoint_path: Path | None = typer.Option(
        None, "--checkpoint", exists=True, dir_okay=False, help="Trained checkpoint (.pt)."
    ),
    tokens_path: Path | None = typer.Option(
        None, "--tokens", exists=True, dir_okay=False, help="Token array (.npy) to evaluate on."
    ),
    batch_size: int = typer.Option(8, "--batch-size"),
    max_tokens: int = typer.Option(0, "--max-tokens", help="Token budget; 0 = the whole array."),
    device: str = typer.Option("cpu", "--device", help="'cpu' or 'cuda'."),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, dir_okay=False, help=_CONFIG_HELP
    ),
) -> None:
    """Loss/perplexity of a trained checkpoint on a token array (FR9)."""
    if config_path is not None:
        config = load_config(config_path, EvaluateConfig)
    else:
        if checkpoint_path is None or tokens_path is None:
            raise typer.BadParameter(
                "either --config or both --checkpoint and --tokens are required"
            )
        config = EvaluateConfig(
            checkpoint_path=checkpoint_path,
            tokens_path=tokens_path,
            batch_size=batch_size,
            max_tokens=max_tokens or None,
            device=device,
        )
    try:
        result = evaluate_checkpoint(config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("sample-report")
def sample_report(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        dir_okay=False,
        help="YAML config mapping to forgelm.config.SampleReportConfig (required -- prompts "
        "and decoding settings are not exposed as individual flags).",
    ),
    output_path: Path | None = typer.Option(
        None, "--output", help="Overrides the config file's 'output_path' when given."
    ),
) -> None:
    """Evaluate + generate documented samples from a checkpoint, written as
    JSON + Markdown (FR9's sample-report format deliverable)."""
    config = load_config(config_path, SampleReportConfig)
    if output_path is not None:
        config = dataclasses.replace(config, output_path=output_path)
    try:
        result = build_sample_report(config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@benchmark_app.command("run")
def benchmark_run(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        dir_okay=False,
        help="YAML config mapping to forgelm.config.BenchmarkRunConfig.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Where to write the result JSON. Overrides the config's 'output_path'.",
    ),
) -> None:
    """Run one throughput/memory/val-loss benchmark measurement (FR10)."""
    config = load_config(config_path, BenchmarkRunConfig)
    result = run_benchmark_cli(config, output_path)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
