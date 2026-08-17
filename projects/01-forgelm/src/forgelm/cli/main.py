"""Typer entry point. Composes the library modules; contains no model/training logic."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from forgelm.cli.dataset_cli import build_dataset_artifacts
from forgelm.cli.smoke import run_smoke
from forgelm.cli.tokenizer_cli import decode_ids, encode_text, train_tokenizer
from forgelm.config import DatasetBuildConfig, SmokeConfig, TokenizerTrainConfig, load_config

app = typer.Typer(help="ForgeLM: a from-scratch decoder-only Transformer training stack.")
tokenizer_app = typer.Typer(help="Train/encode/decode with the byte-level BPE tokenizer.")
dataset_app = typer.Typer(help="Build token arrays + statistics from raw text.")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(dataset_app, name="dataset")

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
