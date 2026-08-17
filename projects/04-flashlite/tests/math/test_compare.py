"""Regression tests for flashlite.compare.allclose_compare (SS1.7's
correctness gate). This comparator is the sole pass/fail decision for every
correctness test and for scripts/run_benchmarks.py's pre-timing check, so a
gap here means a kernel bug could get reported as a clean PASS -- directly
violating the spec's hard constraint 5 ("never suppress a numerical
mismatch to improve benchmark appearance").

Covers the NaN/inf edge cases: a plain `abs_diff > threshold` float
comparison lets NaN slip through undetected (NaN compares false against
everything, including an inf threshold), and `inf - inf` is itself NaN so
even a caught-looking inf case could regress the same way.
"""

from __future__ import annotations

import math

import torch

from flashlite.compare import allclose_compare


def test_basic_match_passes() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    e = torch.tensor([1.0, 2.0, 3.0])
    result = allclose_compare(a, e)
    assert result.passed
    assert result.num_mismatches == 0


def test_basic_mismatch_fails() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    e = torch.tensor([1.0, 2.0, 100.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1
    assert result.first_mismatch_index == 2


def test_nan_in_actual_is_a_mismatch() -> None:
    """A NaN produced by the kernel under test must never be reported as a
    pass, even when every other element matches exactly."""
    a = torch.tensor([1.0, float("nan"), 3.0])
    e = torch.tensor([1.0, 2.0, 3.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1
    assert result.first_mismatch_index == 1


def test_nan_in_expected_is_a_mismatch() -> None:
    """A NaN in the reference tensor (e.g. a malformed test fixture) must
    also be treated as a mismatch rather than silently ignored."""
    a = torch.tensor([1.0, 2.0, 3.0])
    e = torch.tensor([1.0, float("nan"), 3.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1
    assert result.first_mismatch_index == 1


def test_nan_vs_nan_is_a_mismatch() -> None:
    """Even when actual and expected are both NaN at the same position,
    equal_nan=False semantics apply: NaN is never considered a match."""
    a = torch.tensor([1.0, float("nan"), 3.0])
    e = torch.tensor([1.0, float("nan"), 3.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1
    assert result.first_mismatch_index == 1


def test_mismatched_infinities_are_a_mismatch() -> None:
    """inf vs -inf (and inf vs a large finite value) must be caught. This
    also guards the inf-vs-inf case below: inf - inf is itself NaN, so a
    naive abs_diff-based mask could regress silently here too."""
    a = torch.tensor([1.0, float("inf"), 3.0])
    e = torch.tensor([1.0, float("-inf"), 3.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1
    assert result.first_mismatch_index == 1


def test_matching_same_sign_infinities_pass() -> None:
    """+inf vs +inf at the same position is a legitimate match (this is
    the numpy.allclose / torch.isclose convention the module claims to
    follow), distinguishing it from the NaN-producing failure modes above."""
    a = torch.tensor([1.0, float("inf"), 3.0])
    e = torch.tensor([1.0, float("inf"), 3.0])
    result = allclose_compare(a, e)
    assert result.passed
    assert result.num_mismatches == 0


def test_inf_actual_vs_finite_expected_is_a_mismatch() -> None:
    a = torch.tensor([1.0, float("inf"), 3.0])
    e = torch.tensor([1.0, 2.0, 3.0])
    result = allclose_compare(a, e)
    assert not result.passed
    assert result.num_mismatches == 1


def test_empty_tensors_trivially_pass() -> None:
    a = torch.empty(0)
    e = torch.empty(0)
    result = allclose_compare(a, e)
    assert result.passed
    assert result.total_elements == 0


def test_summary_renders_without_raising_on_nan_result() -> None:
    """The summary() string formatting must not itself crash when a NaN
    mismatch is present (max_abs_diff etc. legitimately become NaN)."""
    a = torch.tensor([1.0, float("nan"), 3.0])
    e = torch.tensor([1.0, 2.0, 3.0])
    result = allclose_compare(a, e)
    text = result.summary()
    assert "FAILED" in text
    assert math.isnan(result.max_abs_diff)
