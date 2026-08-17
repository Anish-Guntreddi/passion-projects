"""Typed configuration layer.

Every runnable unit in ForgeLM (smoke test, tokenizer training, dataset
construction, and later model/training/eval configs) is described by a
plain ``@dataclass``. Config files are YAML; :func:`load_config` parses a
YAML mapping into a validated dataclass instance, catching typos (unknown
fields), missing required fields, and type mismatches at load time instead
of failing deep inside training.

See ``docs/decisions/0001-config-layer-dataclasses.md`` for why dataclasses
were chosen over Pydantic.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from forgelm.constants import DEFAULT_SPECIAL_TOKENS


class ConfigError(ValueError):
    """Raised for any problem loading or validating a config."""


# --------------------------------------------------------------------------
# Generic YAML -> dataclass loader
# --------------------------------------------------------------------------


def _optional_inner(field_type: Any) -> tuple[bool, Any]:
    """Return (is_optional, inner_type) for ``X | None`` / ``Optional[X]``.

    Two distinct runtime representations exist for "a union of X and None"
    and both must be recognized:

    - ``typing.Optional[X]`` / ``typing.Union[X, None]``: origin is
      ``typing.Union``.
    - ``X | None`` (PEP 604 syntax, Python 3.10+): origin is
      ``types.UnionType`` -- a *different* object from ``typing.Union``,
      even though both spell the same concept. Checking only
      ``typing.Union`` silently fails to detect PEP 604 optionals: the field
      falls through as "not optional" with its raw ``UnionType`` object as
      ``field_type``, which then matches none of ``_coerce_value``'s type
      branches -- so an explicit ``null`` in YAML would wrongly raise
      "missing required value", and a wrong-type value would silently pass
      through with no validation at all.
    """
    origin = typing.get_origin(field_type)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(field_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return True, non_none[0]
    return False, field_type


def _coerce_value(field_name: str, value: Any, field_type: Any, path_prefix: str) -> Any:
    optional, field_type = _optional_inner(field_type)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path_prefix}{field_name}: missing required value")

    if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
        if not isinstance(value, dict):
            raise ConfigError(
                f"{path_prefix}{field_name}: expected a mapping for nested config "
                f"'{field_type.__name__}', got {type(value).__name__}"
            )
        return _build_dataclass(field_type, value, path_prefix=f"{path_prefix}{field_name}.")

    origin = typing.get_origin(field_type)
    if origin in (list, typing.List):  # noqa: UP006 - runtime typing.List check
        (item_type,) = typing.get_args(field_type) or (Any,)
        if not isinstance(value, list):
            raise ConfigError(f"{path_prefix}{field_name}: expected a list, got {value!r}")
        return [
            _coerce_value(f"{field_name}[{i}]", item, item_type, path_prefix)
            for i, item in enumerate(value)
        ]

    if field_type is Path:
        return Path(value)

    if field_type in (int, float, str, bool):
        if field_type is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if not isinstance(value, field_type) or (field_type is int and isinstance(value, bool)):
            raise ConfigError(
                f"{path_prefix}{field_name}: expected {field_type.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
        return value

    # Anything else (Any, custom types) passes through; the target
    # dataclass's __post_init__ is responsible for further validation.
    return value


def _build_dataclass[T](cls: type[T], data: dict[str, Any], path_prefix: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls)
    valid_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        label = path_prefix.rstrip(".") or cls.__name__
        raise ConfigError(
            f"{label}: unknown field(s) {unknown}; valid fields are {sorted(valid_names)}"
        )

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce_value(f.name, data[f.name], hints[f.name], path_prefix)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        label = path_prefix.rstrip(".") or cls.__name__
        raise ConfigError(f"{label}: {exc}") from exc


def load_config[T](path: str | Path, cls: type[T]) -> T:
    """Load a YAML file into a typed, validated instance of ``cls``."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return _build_dataclass(cls, raw)


def config_from_dict[T](data: dict[str, Any], cls: type[T]) -> T:
    """Build a typed config directly from a dict (used by the CLI and tests)."""
    return _build_dataclass(cls, data)


# --------------------------------------------------------------------------
# Phase 0: smoke-test config
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SmokeConfig:
    """Config for the deterministic placeholder end-to-end command.

    Exercises seeding + a real (if tiny) torch computation on a chosen
    device, so "the pipeline runs" is proven rather than asserted.
    """

    seed: int = 1337
    size: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ConfigError(f"SmokeConfig.size must be positive, got {self.size}")
        if self.device not in ("cpu", "cuda"):
            raise ConfigError(f"SmokeConfig.device must be 'cpu' or 'cuda', got {self.device!r}")


# --------------------------------------------------------------------------
# Phase 1: tokenizer + dataset configs
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TokenizerTrainConfig:
    """Config for training a :class:`forgelm.tokenizer.ByteLevelBPETokenizer`.

    ``output_path`` is optional here (rather than a required CLI flag) so a
    single config file is sufficient to reproduce a full artifact-producing
    run end to end (FR1: "all execution is CLI + config driven") -- the CLI
    still accepts ``--output`` too, which takes precedence when both are
    given, so scripted/ad-hoc invocations aren't forced to edit a YAML file
    just to change where the tokenizer gets written.
    """

    input_path: Path
    output_path: Path | None = None
    vocab_size: int = 512
    special_tokens: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS)
    )

    def __post_init__(self) -> None:
        min_size = 256 + len(self.special_tokens)
        if self.vocab_size < min_size:
            raise ConfigError(
                "TokenizerTrainConfig.vocab_size must be >= 256 + len(special_tokens) "
                f"(={min_size}) to fit the base byte vocabulary and special tokens, "
                f"got {self.vocab_size}"
            )


@dataclasses.dataclass(frozen=True)
class DatasetBuildConfig:
    """Config for turning raw text + a trained tokenizer into token arrays.

    ``output_dir`` is optional here for the same reason as
    ``TokenizerTrainConfig.output_path``: a single config file must be able
    to reproduce the full run, including where artifacts land.
    """

    input_path: Path
    tokenizer_path: Path
    output_dir: Path | None = None
    context_length: int = 128
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ConfigError(
                f"DatasetBuildConfig.context_length must be positive, got {self.context_length}"
            )
        if not (0.0 < self.val_fraction < 1.0):
            raise ConfigError(
                f"DatasetBuildConfig.val_fraction must be in (0, 1), got {self.val_fraction}"
            )
