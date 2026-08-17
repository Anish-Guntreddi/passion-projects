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


# --------------------------------------------------------------------------
# Phase 2: model architecture config
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Transformer architecture config (FR5).

    ``n_kv_heads`` is accepted for forward compatibility with grouped-query
    attention (GQA) but must currently be left unset or equal to
    ``n_heads`` -- GQA itself is deferred per D5 (spec default: stretch),
    see ``docs/decisions/0007-gqa-deferred.md``.

    ``dtype`` is the parameter storage dtype the model is *constructed*
    with. Training-time mixed precision (autocast) is a separate concern
    (D6, see ``docs/decisions/0008-precision-strategy.md``) and does not
    change this field -- a ``dtype="float32"`` model still trains under
    bf16 autocast on a supporting GPU; autocast casts activations on the
    fly without touching the stored master weights.
    """

    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None
    d_ff: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rmsnorm_eps: float = 1e-6
    tie_weights: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ConfigError(f"ModelConfig.vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ConfigError(
                f"ModelConfig.context_length must be positive, got {self.context_length}"
            )
        if self.d_model <= 0:
            raise ConfigError(f"ModelConfig.d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ConfigError(f"ModelConfig.n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ConfigError(f"ModelConfig.n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ConfigError(
                f"ModelConfig.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ConfigError(
                f"ModelConfig: d_model / n_heads (head_dim={head_dim}) must be even -- "
                "RoPE rotates head_dim/2 coordinate pairs"
            )
        if self.n_kv_heads is not None and self.n_kv_heads != self.n_heads:
            raise NotImplementedError(
                "grouped-query attention (n_kv_heads != n_heads) is deferred per D5 -- "
                "see docs/decisions/0007-gqa-deferred.md; leave n_kv_heads unset or "
                "equal to n_heads"
            )
        if self.d_ff <= 0:
            raise ConfigError(f"ModelConfig.d_ff must be positive, got {self.d_ff}")
        if not (0.0 <= self.dropout < 1.0):
            raise ConfigError(f"ModelConfig.dropout must be in [0, 1), got {self.dropout}")
        if self.rope_theta <= 0:
            raise ConfigError(f"ModelConfig.rope_theta must be positive, got {self.rope_theta}")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ConfigError(
                "ModelConfig.dtype must be one of 'float32'/'bfloat16'/'float16', "
                f"got {self.dtype!r}"
            )


# --------------------------------------------------------------------------
# Phase 3: training system config
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Training-loop config (FR6/FR7): optimizer, LR schedule, precision,
    gradient accumulation/clipping, validation cadence.

    ``micro_batch_size`` is the batch actually run through one forward/
    backward pass; ``grad_accum_steps`` > 1 accumulates gradients over
    that many micro-batches before an optimizer step, so the *effective*
    batch size is ``micro_batch_size * grad_accum_steps`` without needing
    that many examples resident in memory/compute at once at any one time
    (FR6 "gradient accumulation").
    """

    seed: int = 1337
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 100
    warmup_steps: int = 10
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    eval_interval: int = 20
    eval_batches: int = 5
    device: str = "cpu"
    precision: str = "auto"  # "auto" | "fp32" | "bf16" -- see D6

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ConfigError(
                f"TrainingConfig.micro_batch_size must be positive, got {self.micro_batch_size}"
            )
        if self.grad_accum_steps <= 0:
            raise ConfigError(
                f"TrainingConfig.grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )
        if self.max_steps <= 0:
            raise ConfigError(f"TrainingConfig.max_steps must be positive, got {self.max_steps}")
        if self.warmup_steps < 0:
            raise ConfigError(
                f"TrainingConfig.warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.warmup_steps > self.max_steps:
            raise ConfigError("TrainingConfig.warmup_steps must be <= max_steps")
        if self.lr <= 0:
            raise ConfigError(f"TrainingConfig.lr must be positive, got {self.lr}")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ConfigError(
                f"TrainingConfig.min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}"
            )
        if self.weight_decay < 0:
            raise ConfigError(
                f"TrainingConfig.weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not (0.0 < self.beta1 < 1.0) or not (0.0 < self.beta2 < 1.0):
            raise ConfigError(
                f"TrainingConfig beta1/beta2 must be in (0, 1), got {self.beta1}/{self.beta2}"
            )
        if self.grad_clip_norm <= 0:
            raise ConfigError(
                f"TrainingConfig.grad_clip_norm must be positive, got {self.grad_clip_norm}"
            )
        if self.eval_interval <= 0:
            raise ConfigError(
                f"TrainingConfig.eval_interval must be positive, got {self.eval_interval}"
            )
        if self.eval_batches <= 0:
            raise ConfigError(
                f"TrainingConfig.eval_batches must be positive, got {self.eval_batches}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ConfigError(f"TrainingConfig.device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ConfigError(
                f"TrainingConfig.precision must be 'auto'/'fp32'/'bf16', got {self.precision!r}"
            )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Top-level config for ``forgelm train`` (FR1: config-driven execution).

    Points at pre-built token arrays produced by ``forgelm dataset build``
    -- ``forgelm.training`` owns optimization/checkpointing only, not
    tokenization or dataset construction (see the module boundary table in
    ``docs/architecture.md``).
    """

    model: ModelConfig
    training: TrainingConfig
    train_tokens_path: Path
    val_tokens_path: Path
    output_dir: Path | None = None
    resume_from: Path | None = None
    checkpoint_interval: int = 0  # 0 = only ever write the final checkpoint

    def __post_init__(self) -> None:
        if self.checkpoint_interval < 0:
            raise ConfigError(
                "TrainConfig.checkpoint_interval must be non-negative, got "
                f"{self.checkpoint_interval}"
            )


# --------------------------------------------------------------------------
# Phase 4: generation + evaluation configs
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GenerationSettings:
    """Decoding settings for :func:`forgelm.generation.sampling.generate` (FR8).

    ``do_sample=False`` is greedy decoding: every step takes the argmax
    token and ``temperature``/``top_k``/``top_p`` are ignored entirely
    (documented here, not silently -- see
    ``forgelm.generation.sampling.generate``). ``do_sample=True`` samples
    from the (temperature-scaled, then optionally top-k/top-p filtered)
    softmax distribution using a seeded generator, so a report produced
    with these settings is reproducible from the recorded config alone.
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigError(
                f"GenerationSettings.max_new_tokens must be positive, got {self.max_new_tokens}"
            )
        if self.temperature <= 0:
            raise ConfigError(
                f"GenerationSettings.temperature must be positive, got {self.temperature}"
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ConfigError(f"GenerationSettings.top_k must be positive, got {self.top_k}")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ConfigError(f"GenerationSettings.top_p must be in (0, 1], got {self.top_p}")


@dataclasses.dataclass(frozen=True)
class GenerateConfig:
    """Top-level config for ``forgelm generate`` (FR8): load a checkpoint +
    tokenizer, generate a continuation for one prompt."""

    checkpoint_path: Path
    tokenizer_path: Path
    prompt: str = ""
    device: str = "cpu"
    generation: GenerationSettings = dataclasses.field(default_factory=GenerationSettings)

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ConfigError(f"GenerateConfig.device must be 'cpu' or 'cuda', got {self.device!r}")


@dataclasses.dataclass(frozen=True)
class EvaluateConfig:
    """Top-level config for ``forgelm evaluate`` (FR9): loss/perplexity of a
    checkpoint against a pre-built token array, with a configurable token
    budget (``max_tokens``, ``None`` = the whole array)."""

    checkpoint_path: Path
    tokens_path: Path
    batch_size: int = 8
    max_tokens: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"EvaluateConfig.batch_size must be positive, got {self.batch_size}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigError(
                f"EvaluateConfig.max_tokens must be positive when set, got {self.max_tokens}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ConfigError(f"EvaluateConfig.device must be 'cpu' or 'cuda', got {self.device!r}")


@dataclasses.dataclass(frozen=True)
class SampleReportConfig:
    """Top-level config for ``forgelm sample-report`` (FR9's "sample-report
    format" deliverable): one checkpoint evaluated against a val token
    array, plus generations for each of ``prompts`` under one documented
    :class:`GenerationSettings`, written as a single JSON + Markdown report.
    """

    checkpoint_path: Path
    tokenizer_path: Path
    output_path: Path
    val_tokens_path: Path | None = None
    prompts: list[str] = dataclasses.field(default_factory=lambda: [""])
    generation: GenerationSettings = dataclasses.field(default_factory=GenerationSettings)
    eval_batch_size: int = 8
    eval_max_tokens: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ConfigError("SampleReportConfig.prompts must contain at least one prompt")
        if self.eval_batch_size <= 0:
            raise ConfigError(
                f"SampleReportConfig.eval_batch_size must be positive, got {self.eval_batch_size}"
            )
        if self.eval_max_tokens is not None and self.eval_max_tokens <= 0:
            raise ConfigError(
                "SampleReportConfig.eval_max_tokens must be positive when set, got "
                f"{self.eval_max_tokens}"
            )
        if self.device not in ("cpu", "cuda"):
            raise ConfigError(
                f"SampleReportConfig.device must be 'cpu' or 'cuda', got {self.device!r}"
            )


# --------------------------------------------------------------------------
# Phase 5: benchmark harness config
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BenchmarkRunConfig:
    """Top-level config for ``forgelm benchmark run`` (FR10): one throughput
    / memory / val-loss measurement of a single model+training config
    against pre-built token arrays.

    ``bench_warmup_steps`` are run and timed-out-of-band before
    measurement starts (CUDA kernel selection / cuDNN autotune / optional
    ``torch.compile`` tracing all happen here, not during the measured
    window) -- distinct from ``training.warmup_steps``, which is the LR
    schedule's own warmup and affects the *loss*, not the *timing*
    methodology.
    """

    name: str
    model: ModelConfig
    training: TrainingConfig
    train_tokens_path: Path
    val_tokens_path: Path
    bench_warmup_steps: int = 10
    measured_steps: int = 50
    eval_interval: int | None = None
    torch_compile: bool = False
    output_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("BenchmarkRunConfig.name must be non-empty")
        if self.bench_warmup_steps < 0:
            raise ConfigError(
                "BenchmarkRunConfig.bench_warmup_steps must be non-negative, got "
                f"{self.bench_warmup_steps}"
            )
        if self.measured_steps <= 0:
            raise ConfigError(
                f"BenchmarkRunConfig.measured_steps must be positive, got {self.measured_steps}"
            )
        if self.eval_interval is not None and self.eval_interval <= 0:
            raise ConfigError(
                "BenchmarkRunConfig.eval_interval must be positive when set, got "
                f"{self.eval_interval}"
            )
