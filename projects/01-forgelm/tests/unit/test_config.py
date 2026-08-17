from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from forgelm.config import (
    ConfigError,
    DatasetBuildConfig,
    SmokeConfig,
    TokenizerTrainConfig,
    config_from_dict,
    load_config,
)

# -- generic loader: dataclass features (nesting, lists, optionals) ----------


@dataclasses.dataclass(frozen=True)
class _Inner:
    name: str
    weight: float = 1.0


@dataclasses.dataclass(frozen=True)
class _Outer:
    inner: _Inner
    tags: list[str]
    count: int
    ratio: float | None = None


def test_config_from_dict_builds_nested_dataclass() -> None:
    data = {"inner": {"name": "a", "weight": 2.0}, "tags": ["x", "y"], "count": 3}
    cfg = config_from_dict(data, _Outer)
    assert cfg.inner == _Inner(name="a", weight=2.0)
    assert cfg.tags == ["x", "y"]
    assert cfg.count == 3
    assert cfg.ratio is None


def test_config_from_dict_coerces_int_to_float() -> None:
    cfg = config_from_dict({"inner": {"name": "a", "weight": 5}, "tags": [], "count": 1}, _Outer)
    assert cfg.inner.weight == 5.0
    assert isinstance(cfg.inner.weight, float)


def test_config_from_dict_rejects_unknown_field() -> None:
    with pytest.raises(ConfigError, match="unknown field"):
        config_from_dict({"inner": {"name": "a"}, "tags": [], "count": 1, "bogus": 1}, _Outer)


def test_config_from_dict_rejects_wrong_type() -> None:
    with pytest.raises(ConfigError, match="expected int"):
        config_from_dict({"inner": {"name": "a"}, "tags": [], "count": "three"}, _Outer)


def test_config_from_dict_rejects_missing_required_field() -> None:
    with pytest.raises(ConfigError):
        config_from_dict({"tags": [], "count": 1}, _Outer)


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump({"seed": 7, "size": 8, "device": "cpu"}))
    cfg = load_config(path, SmokeConfig)
    assert cfg == SmokeConfig(seed=7, size=8, device="cpu")


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml", SmokeConfig)


def test_load_config_non_mapping_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, SmokeConfig)


# -- generic loader: PEP 604 (`X | None`) optionals ---------------------------
#
# `_Outer.ratio` is declared with the modern `float | None` syntax (not
# `typing.Optional[float]`), specifically to exercise the codepath the
# reviewer flagged: `typing.get_origin(X | None)` is `types.UnionType`, a
# *different* object from `typing.Union`. Only testing the omitted-default
# case (as the pre-existing `test_config_from_dict_builds_nested_dataclass`
# does) can pass even if PEP 604 unions aren't recognized at all, since an
# omitted field never reaches `_optional_inner` with a real value to check --
# it just falls back to the dataclass default. Explicit `null` and
# wrong-type values are the cases that actually exercise the recognition
# logic.


def test_config_from_dict_accepts_explicit_null_for_pep604_optional() -> None:
    data = {"inner": {"name": "a"}, "tags": [], "count": 1, "ratio": None}
    cfg = config_from_dict(data, _Outer)
    assert cfg.ratio is None


def test_config_from_dict_accepts_value_for_pep604_optional() -> None:
    data = {"inner": {"name": "a"}, "tags": [], "count": 1, "ratio": 0.5}
    cfg = config_from_dict(data, _Outer)
    assert cfg.ratio == 0.5


def test_config_from_dict_rejects_wrong_type_for_pep604_optional() -> None:
    data = {"inner": {"name": "a"}, "tags": [], "count": 1, "ratio": "not-a-float"}
    with pytest.raises(ConfigError, match="expected float"):
        config_from_dict(data, _Outer)


# -- domain config validation --------------------------------------------------


def test_smoke_config_defaults() -> None:
    cfg = SmokeConfig()
    assert cfg.seed == 1337
    assert cfg.size == 64
    assert cfg.device == "cpu"


@pytest.mark.parametrize("size", [0, -1])
def test_smoke_config_rejects_non_positive_size(size: int) -> None:
    with pytest.raises(ConfigError):
        SmokeConfig(size=size)


def test_smoke_config_rejects_bad_device() -> None:
    with pytest.raises(ConfigError):
        SmokeConfig(device="tpu")


def test_tokenizer_train_config_rejects_too_small_vocab(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello")
    with pytest.raises(ConfigError, match="vocab_size"):
        TokenizerTrainConfig(input_path=input_path, vocab_size=10)


def test_tokenizer_train_config_default_special_tokens(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello")
    cfg = TokenizerTrainConfig(input_path=input_path)
    assert cfg.special_tokens == ["<|bos|>", "<|eos|>", "<|pad|>"]


def test_tokenizer_train_config_output_path_defaults_to_none(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello")
    cfg = TokenizerTrainConfig(input_path=input_path)
    assert cfg.output_path is None


def test_tokenizer_train_config_output_path_loads_from_yaml(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello")
    config_path = tmp_path / "tok.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "tok.json"),
                "vocab_size": 300,
            }
        )
    )
    cfg = load_config(config_path, TokenizerTrainConfig)
    assert cfg.output_path == tmp_path / "tok.json"


def test_dataset_build_config_output_dir_defaults_to_none() -> None:
    cfg = DatasetBuildConfig(input_path=Path("x.txt"), tokenizer_path=Path("t.json"))
    assert cfg.output_dir is None


def test_dataset_build_config_output_dir_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_path": "corpus.txt",
                "tokenizer_path": "tok.json",
                "output_dir": str(tmp_path / "built"),
            }
        )
    )
    cfg = load_config(config_path, DatasetBuildConfig)
    assert cfg.output_dir == tmp_path / "built"


@pytest.mark.parametrize("context_length", [0, -5])
def test_dataset_build_config_rejects_non_positive_context_length(context_length: int) -> None:
    with pytest.raises(ConfigError, match="context_length"):
        DatasetBuildConfig(
            input_path=Path("x.txt"),
            tokenizer_path=Path("t.json"),
            context_length=context_length,
        )


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.5])
def test_dataset_build_config_rejects_out_of_range_val_fraction(val_fraction: float) -> None:
    with pytest.raises(ConfigError, match="val_fraction"):
        DatasetBuildConfig(
            input_path=Path("x.txt"),
            tokenizer_path=Path("t.json"),
            val_fraction=val_fraction,
        )
