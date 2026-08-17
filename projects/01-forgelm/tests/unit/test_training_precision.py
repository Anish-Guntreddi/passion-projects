from __future__ import annotations

import pytest
import torch

from forgelm.training.precision import resolve_precision

# -- CPU branches (run everywhere, including GPU-less CI) -------------------------


def test_resolve_precision_fp32_request_on_cpu() -> None:
    policy = resolve_precision(torch.device("cpu"), "fp32")
    assert policy.device_type == "cpu"
    assert policy.autocast_dtype is None
    assert policy.enabled is False


def test_resolve_precision_auto_on_cpu_resolves_to_fp32() -> None:
    """D6's default: bf16 only if the GPU supports it -- CPU always falls
    back to plain fp32."""
    policy = resolve_precision(torch.device("cpu"), "auto")
    assert policy.autocast_dtype is None
    assert policy.enabled is False


def test_resolve_precision_bf16_request_on_cpu_raises() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        resolve_precision(torch.device("cpu"), "bf16")


def test_resolve_precision_rejects_unknown_request_string() -> None:
    with pytest.raises(ValueError, match="fp32"):
        resolve_precision(torch.device("cpu"), "fp16")


# -- CUDA branches, exercised on CPU-only CI via monkeypatch -----------------------
#
# These test the *branching logic*, not real CUDA execution: resolve_precision
# only calls torch.cuda.is_available()/is_bf16_supported() before deciding
# which dtype to report, it never launches a kernel, so simulating those two
# functions is a faithful, hardware-independent test of the policy.


def test_resolve_precision_auto_uses_bf16_when_cuda_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    policy = resolve_precision(torch.device("cuda"), "auto")
    assert policy.device_type == "cuda"
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.enabled is True


def test_resolve_precision_auto_falls_back_to_fp32_when_bf16_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    policy = resolve_precision(torch.device("cuda"), "auto")
    assert policy.autocast_dtype is None
    assert policy.enabled is False


def test_resolve_precision_bf16_request_on_cuda() -> None:
    policy = resolve_precision(torch.device("cuda"), "bf16")
    assert policy.device_type == "cuda"
    assert policy.autocast_dtype is torch.bfloat16
