"""Phase 0 exit criterion: "Trusted PyTorch outputs exist across
representative shapes." This file is what makes V0 trusted: an independent
brute-force (pure-Python, no torch.matmul) computation for a tiny shape,
causal-masking behavioral checks, a shape/dtype sweep across the spec's
test matrix (SS1.7), extreme-logit stability, and a cross-check against
torch's own SDPA (ADR 0006 / D7) as a second, independently-implemented
opinion.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flashlite.reference.attention import reference_attention
from flashlite.reference.tensors import DEFAULT_SEED, make_extreme_qkv, make_qkv


def _brute_force_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Independent, deliberately naive Python-loop reference: no torch.matmul,
    no torch.softmax -- just nested loops and math.exp. Used only to
    double-check reference_attention on tiny shapes (this would be far too
    slow to run at any real size, which is exactly why reference_attention
    itself uses vectorized torch ops and this function exists only as a
    cross-check, not as the project's actual reference implementation).
    """
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            for i in range(S):
                scores = []
                for j in range(S):
                    if causal and j > i:
                        scores.append(float("-inf"))
                        continue
                    s = 0.0
                    for d in range(D):
                        s += float(q[b, h, i, d]) * float(k[b, h, j, d])
                    scores.append(s * scale)
                row_max = max(scores)
                exps = [math.exp(s - row_max) for s in scores]
                denom = sum(exps)
                probs = [e / denom for e in exps]
                for d in range(D):
                    acc = 0.0
                    for j in range(S):
                        acc += probs[j] * float(v[b, h, j, d])
                    out[b, h, i, d] = acc
    return out


@pytest.mark.parametrize("causal", [False, True])
def test_matches_brute_force_python_loops(causal: bool) -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=5, head_dim=3, seed=DEFAULT_SEED)
    expected = _brute_force_attention(q, k, v, causal)
    actual = reference_attention(q, k, v, causal=causal)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_output_shape_matches_input_shape() -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=17, head_dim=64, seed=DEFAULT_SEED)
    out = reference_attention(q, k, v)
    assert out.shape == q.shape


def test_causal_output_does_not_depend_on_future_keys() -> None:
    """The mathematical invariant a causal mask must satisfy: perturbing
    k[..., j, :] or v[..., j, :] for j > i must leave out[..., i, :]
    unchanged. This is checked directly (not just "does softmax look
    triangular") because it's the actual causal-attention contract.
    """
    q, k, v = make_qkv(batch=1, heads=2, seq_len=8, head_dim=16, seed=DEFAULT_SEED)
    out_before = reference_attention(q, k, v, causal=True)

    k_perturbed = k.clone()
    v_perturbed = v.clone()
    future_pos = 5  # perturb keys/values for position 5; positions 0-4 must be unaffected
    k_perturbed[:, :, future_pos:, :] += 1000.0
    v_perturbed[:, :, future_pos:, :] += 1000.0
    out_after = reference_attention(q, k_perturbed, v_perturbed, causal=True)

    torch.testing.assert_close(out_before[:, :, :future_pos, :], out_after[:, :, :future_pos, :])
    # Sanity: the perturbation actually changes later positions (otherwise this
    # test would trivially pass regardless of masking correctness).
    assert not torch.allclose(out_before[:, :, future_pos:, :], out_after[:, :, future_pos:, :])


def test_causal_row_zero_only_attends_to_itself() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED)
    out, probs = reference_attention(q, k, v, causal=True, return_probs=True)
    # Row 0 may only attend to key 0 -> probs[..., 0, 0] == 1, rest == 0.
    torch.testing.assert_close(probs[0, 0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(probs[0, 0, 0, 1:], torch.zeros(3))
    # And its output must equal v[..., 0, :] exactly.
    torch.testing.assert_close(out[0, 0, 0, :], v[0, 0, 0, :])


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 17, 128])  # includes awkward, non-tile-multiple lengths
@pytest.mark.parametrize("causal", [False, True])
def test_shape_and_dtype_sweep_produces_finite_normalized_output(
    head_dim: int, seq_len: int, causal: bool
) -> None:
    q, k, v = make_qkv(batch=2, heads=3, seq_len=seq_len, head_dim=head_dim, seed=DEFAULT_SEED)
    out, probs = reference_attention(q, k, v, causal=causal, return_probs=True)

    assert out.shape == (2, 3, seq_len, head_dim)
    assert torch.isfinite(out).all()
    # Every row of P must sum to 1 (a valid probability distribution).
    row_sums = probs.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_extreme_logits_do_not_overflow() -> None:
    """Spec SS1.7: 'extreme logits to stress softmax stability.' V0 uses
    torch.softmax, which subtracts the row max internally, so this must
    stay finite even with large-magnitude scores (this is the trusted
    ground truth every later, hand-written softmax kernel is checked
    against -- including at extreme values).
    """
    q, k, v = make_extreme_qkv(batch=1, heads=2, seq_len=32, head_dim=64, seed=DEFAULT_SEED, magnitude=80.0)
    out, probs = reference_attention(q, k, v, causal=False, return_probs=True)
    assert torch.isfinite(out).all()
    assert torch.isfinite(probs).all()
    row_sums = probs.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)


def test_deterministic_seed_reproduces_identical_output() -> None:
    q1, k1, v1 = make_qkv(batch=2, heads=2, seq_len=16, head_dim=32, seed=DEFAULT_SEED)
    q2, k2, v2 = make_qkv(batch=2, heads=2, seq_len=16, head_dim=32, seed=DEFAULT_SEED)
    torch.testing.assert_close(q1, q2, atol=0, rtol=0)
    torch.testing.assert_close(k1, k2, atol=0, rtol=0)
    torch.testing.assert_close(v1, v2, atol=0, rtol=0)

    out1 = reference_attention(q1, k1, v1, causal=True)
    out2 = reference_attention(q2, k2, v2, causal=True)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_cross_check_against_torch_sdpa(causal: bool) -> None:
    """ADR 0006 (D7): torch.nn.functional.scaled_dot_product_attention is
    the chosen platform-optimized comparison backend for Phase 7. Used here
    as an extra, independently-implemented cross-check of the V0 reference
    itself -- NOT as a replacement for it (the brute-force loop test above
    remains the from-first-principles ground truth per hard constraint 2:
    "derive algorithms from documented math").
    """
    q, k, v = make_qkv(batch=2, heads=4, seq_len=64, head_dim=32, seed=DEFAULT_SEED)
    ours = reference_attention(q, k, v, causal=causal)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(ours, sdpa, atol=1e-5, rtol=1e-4)


def test_unsupported_shape_fails_clearly() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED)
    with pytest.raises(ValueError, match="4-D"):
        reference_attention(q[0], k, v)  # q is now 3-D


def test_mismatched_qkv_shapes_fail_clearly() -> None:
    q, _k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED)
    k_wrong, _, _ = make_qkv(batch=1, heads=1, seq_len=5, head_dim=8, seed=DEFAULT_SEED + 1)
    with pytest.raises(ValueError, match="must share one shape"):
        reference_attention(q, k_wrong, v)


def test_fp16_input_raises_not_implemented() -> None:
    q, k, v = make_qkv(batch=1, heads=1, seq_len=4, head_dim=8, seed=DEFAULT_SEED, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="deferred"):
        reference_attention(q, k, v)
