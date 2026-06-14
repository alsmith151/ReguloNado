"""Unit tests for regulonado.training.losses."""
from __future__ import annotations

import pytest
import torch

from regulonado.training.losses import (
    log1p_huber_loss,
    paired_binwise_log2fc_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    topk_additive_loss,
    topk_reweight_loss,
    transfer_calibration_loss,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, T, L = 2, 4, 128


def _rand_pos(*shape: int) -> torch.Tensor:
    t = torch.rand(*shape) + 0.1
    t.requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# poisson_multinomial_loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rescale", [False, True])
def test_poisson_multinomial_shape(rescale: bool) -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    loss = poisson_multinomial_loss(pred, tgt, rescale=rescale)
    assert loss.shape == ()


def test_poisson_multinomial_non_negative() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert poisson_multinomial_loss(pred, tgt).item() >= 0.0


def test_poisson_multinomial_zero_target_no_nan() -> None:
    pred = _rand_pos(B, T, L)
    tgt = torch.zeros(B, T, L)
    loss = poisson_multinomial_loss(pred, tgt)
    assert torch.isfinite(loss)


def test_poisson_multinomial_perfect_pred_low_loss() -> None:
    tgt = _rand_pos(B, T, L)
    loss_perfect = poisson_multinomial_loss(tgt.detach().clone(), tgt)
    loss_random = poisson_multinomial_loss(_rand_pos(B, T, L), tgt)
    assert loss_perfect.item() < loss_random.item()


def test_poisson_multinomial_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    poisson_multinomial_loss(pred, tgt).backward()
    assert pred.grad is not None


def test_poisson_multinomial_zero_weight_collapses_to_multinomial() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    with torch.no_grad():
        loss_w0 = poisson_multinomial_loss(pred, tgt, poisson_weight=0.0)
        # Manually compute multinomial term
        eps = 1e-6
        y_p = pred.float() + eps
        y_t = tgt.float() + eps
        p_pred = y_p / y_p.sum(-1, keepdim=True)
        manual = -(y_t * torch.log(p_pred)).sum(-1) / L
        expected = manual.mean()
    assert abs(loss_w0.item() - expected.item()) < 1e-4


# ---------------------------------------------------------------------------
# scaled_poisson_multinomial_loss
# ---------------------------------------------------------------------------


def test_scaled_poisson_multinomial_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    sf = torch.ones(T)
    loss = scaled_poisson_multinomial_loss(pred, tgt, scale_factors=sf)
    assert loss.shape == ()


def test_scaled_poisson_multinomial_non_negative() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert scaled_poisson_multinomial_loss(pred, tgt).item() >= 0.0


def test_scaled_poisson_multinomial_unit_scale_matches_base() -> None:
    torch.manual_seed(0)
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    with torch.no_grad():
        base = poisson_multinomial_loss(pred, tgt)
        scaled = scaled_poisson_multinomial_loss(pred, tgt, scale_factors=torch.ones(T))
    assert abs(base.item() - scaled.item()) < 1e-4


def test_scaled_poisson_multinomial_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    scaled_poisson_multinomial_loss(pred, tgt).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# poisson_nll_loss
# ---------------------------------------------------------------------------


def test_poisson_nll_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert poisson_nll_loss(pred, tgt).shape == ()


def test_poisson_nll_non_negative() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert poisson_nll_loss(pred, tgt).item() >= 0.0


def test_poisson_nll_zero_target_no_nan() -> None:
    pred = _rand_pos(B, T, L)
    tgt = torch.zeros(B, T, L)
    assert torch.isfinite(poisson_nll_loss(pred, tgt))


def test_poisson_nll_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    poisson_nll_loss(pred, tgt).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# log1p_huber_loss
# ---------------------------------------------------------------------------


def test_log1p_huber_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert log1p_huber_loss(pred, tgt).shape == ()


def test_log1p_huber_non_negative() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert log1p_huber_loss(pred, tgt).item() >= 0.0


def test_log1p_huber_perfect_pred_zero() -> None:
    tgt = _rand_pos(B, T, L)
    loss = log1p_huber_loss(tgt.detach().clone(), tgt)
    assert loss.item() < 1e-6


def test_log1p_huber_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    log1p_huber_loss(pred, tgt).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# topk_additive_loss
# ---------------------------------------------------------------------------


def test_topk_additive_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert topk_additive_loss(pred, tgt).shape == ()


def test_topk_additive_geq_base() -> None:
    torch.manual_seed(1)
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    with torch.no_grad():
        base = poisson_multinomial_loss(pred, tgt)
        total = topk_additive_loss(pred, tgt, topk_weight=1.0)
    assert total.item() >= base.item() - 1e-5


def test_topk_additive_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    topk_additive_loss(pred, tgt).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# topk_reweight_loss
# ---------------------------------------------------------------------------


def test_topk_reweight_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert topk_reweight_loss(pred, tgt).shape == ()


def test_topk_reweight_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    topk_reweight_loss(pred, tgt).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# transfer_calibration_loss
# ---------------------------------------------------------------------------


def test_transfer_calibration_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    assert transfer_calibration_loss(pred, tgt).shape == ()


def test_transfer_calibration_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    transfer_calibration_loss(pred, tgt).backward()
    assert pred.grad is not None


def test_transfer_calibration_with_topk() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    loss = transfer_calibration_loss(pred, tgt, topk_bin_weight=1.0, topk_bin_count=10)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# paired_binwise_log2fc_loss
# ---------------------------------------------------------------------------


def test_paired_log2fc_shape() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    # condition_ids is over tracks (T=4): first 2 baseline, last 2 perturbed
    cond = torch.tensor([0, 0, 1, 1])
    loss = paired_binwise_log2fc_loss(pred, tgt, cond)
    assert loss.shape == ()


def test_paired_log2fc_no_pair_returns_zero() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    # All tracks in condition 0 — no perturbed condition, so no pairs.
    cond = torch.tensor([0, 0, 0, 0])
    loss = paired_binwise_log2fc_loss(pred, tgt, cond)
    assert loss.item() == pytest.approx(0.0)


def test_paired_log2fc_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    cond = torch.tensor([0, 0, 1, 1])
    paired_binwise_log2fc_loss(pred, tgt, cond).backward()
    assert pred.grad is not None


def test_paired_log2fc_perfect_pred_near_zero() -> None:
    tgt = _rand_pos(B, T, L)
    cond = torch.tensor([0, 0, 1, 1])
    loss = paired_binwise_log2fc_loss(tgt.detach().clone(), tgt, cond)
    assert loss.item() < 1e-5
