"""Tolerance-based correctness comparison (mirrors KernelForge's
src/common/compare.hpp: elementwise |actual - expected| <= atol + rtol *
|expected|, the numpy.allclose convention), plus the three metrics the
FlashLite spec's test matrix requires be reported (SS1.7): max absolute
error, mean absolute error, and a relative-error figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Defaults for FP32 (ADR 0004: FP32 first), matching kernelforge::kDefaultAtol
# / kDefaultRtol exactly for cross-project consistency.
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4


@dataclass
class CompareResult:
    passed: bool
    total_elements: int
    num_mismatches: int
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    first_mismatch_index: int | None
    first_mismatch_actual: float | None
    first_mismatch_expected: float | None
    atol: float
    rtol: float

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASSED: {self.total_elements} elements, max_abs_diff={self.max_abs_diff:.3e}, "
                f"mean_abs_diff={self.mean_abs_diff:.3e}, max_rel_diff={self.max_rel_diff:.3e} "
                f"(atol={self.atol:.1e}, rtol={self.rtol:.1e})"
            )
        return (
            f"FAILED: {self.num_mismatches}/{self.total_elements} elements exceed tolerance "
            f"(atol={self.atol:.1e}, rtol={self.rtol:.1e}); max_abs_diff={self.max_abs_diff:.3e}, "
            f"mean_abs_diff={self.mean_abs_diff:.3e}, max_rel_diff={self.max_rel_diff:.3e}; "
            f"first mismatch at flat index {self.first_mismatch_index}: "
            f"actual={self.first_mismatch_actual:.6g} expected={self.first_mismatch_expected:.6g}"
        )


def allclose_compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> CompareResult:
    """Elementwise tolerance comparison. n == 0 trivially passes.

    Mismatch criterion (numpy.allclose convention, same as
    kernelforge::allclose): |actual - expected| > atol + rtol * |expected|.

    Any non-finite element counts as an automatic mismatch. A plain
    `abs_diff > atol + rtol * |expected|` comparison is not enough on its
    own: IEEE-754 NaN compares false against everything, including an `inf`
    threshold, so a NaN in `actual` or `expected` (or two mismatched
    infinities, where `inf - inf` is itself NaN) would silently fail to
    trip the mask and be reported as a pass. torch.isclose's default
    `equal_nan=False` treats any NaN operand as "not close" and correctly
    special-cases inf vs inf, so it is used for the pass/fail gate while
    the raw `abs_diff` array below is kept for the reported diagnostics
    (max/mean abs diff, rel diff), where NaN propagating into those numbers
    is the desired, visible signal.
    """
    if actual.shape != expected.shape:
        raise ValueError(f"allclose_compare: shape mismatch actual={tuple(actual.shape)} expected={tuple(expected.shape)}")

    a = actual.detach().to(dtype=torch.float64, device="cpu").flatten()
    e = expected.detach().to(dtype=torch.float64, device="cpu").flatten()
    n = a.numel()

    if n == 0:
        return CompareResult(
            passed=True, total_elements=0, num_mismatches=0, max_abs_diff=0.0, mean_abs_diff=0.0,
            max_rel_diff=0.0, first_mismatch_index=None, first_mismatch_actual=None,
            first_mismatch_expected=None, atol=atol, rtol=rtol,
        )

    abs_diff = (a - e).abs()
    mismatch_mask = ~torch.isclose(a, e, atol=atol, rtol=rtol, equal_nan=False)

    num_mismatches = int(mismatch_mask.sum().item())
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())

    rel_diff = abs_diff / (e.abs() + 1e-12)
    max_rel_diff = float(rel_diff.max().item())

    first_idx = None
    first_actual = None
    first_expected = None
    if num_mismatches > 0:
        first_idx = int(torch.nonzero(mismatch_mask, as_tuple=False)[0].item())
        first_actual = float(a[first_idx].item())
        first_expected = float(e[first_idx].item())

    return CompareResult(
        passed=(num_mismatches == 0),
        total_elements=n,
        num_mismatches=num_mismatches,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        first_mismatch_index=first_idx,
        first_mismatch_actual=first_actual,
        first_mismatch_expected=first_expected,
        atol=atol,
        rtol=rtol,
    )
