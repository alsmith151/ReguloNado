"""Unit tests for regulonado.regions.loss.

Expected numeric values were computed by running the same toy inputs through UEF's
``unique_enhancer_finding.modelling.loss.CountLikelihoodLoss`` directly (via its own
venv, ``sys.path``-imported) and hard-coded here -- see the plan for the scratch
script used. Any drift from those values means this port has diverged from UEF.
"""

from __future__ import annotations

import math

import pytest
import torch
from regulonado.regions.loss import COUNT_NOISE_MODELS, CountLikelihoodLoss

# ---------------------------------------------------------------------------
# Shared toy inputs (3 regions, 3 tracks, 2 groups: tracks 0/1 -> group 0, track 2 -> group 1)
# ---------------------------------------------------------------------------

TRACK_GROUPS = [0, 0, 1]
LOG_SIZE_FACTORS = torch.tensor([0.1, -0.1, 0.0])

LOG_MU = torch.tensor(
    [
        [0.5, 0.3, 0.2],
        [1.0, 0.9, -0.2],
        [0.1, 0.2, 0.3],
    ]
)
LOG_NOISE = torch.tensor([-1.0, -0.5, 0.0])
Y_TRUE = torch.tensor(
    [
        [1.0, float("nan"), 3.0],
        [2.0, 2.0, 0.0],
        [5.0, 1.0, 4.0],
    ]
)
ETA = torch.tensor(
    [
        [0.4, 0.25],
        [0.95, -0.2],
        [0.15, 0.3],
    ]
)


def test_nb_loss_matches_uef() -> None:
    loss_fn = CountLikelihoodLoss(noise="nb")
    value = loss_fn(y_true=Y_TRUE, log_mu=LOG_MU, log_noise=LOG_NOISE)
    assert value.item() == pytest.approx(2.0752148628234863, abs=1e-6)


def test_nb_loss_with_contrast_matches_uef() -> None:
    loss_fn = CountLikelihoodLoss(noise="nb", contrast_weight=0.5, contrast_multiplier=1.0)
    track_groups = torch.as_tensor(TRACK_GROUPS, dtype=torch.long)
    value = loss_fn(
        y_true=Y_TRUE,
        log_mu=LOG_MU,
        log_noise=LOG_NOISE,
        eta=ETA,
        track_groups=track_groups,
        log_size_factors=LOG_SIZE_FACTORS,
    )
    assert value.item() == pytest.approx(2.1119329929351807, abs=1e-6)


def test_lognormal_loss_matches_uef() -> None:
    loss_fn = CountLikelihoodLoss(noise="lognormal")
    value = loss_fn(y_true=Y_TRUE, log_mu=LOG_MU, log_noise=LOG_NOISE)
    assert value.item() == pytest.approx(0.14926299452781677, abs=1e-6)


def test_nan_labels_are_ignored_not_zero_filled() -> None:
    """A NaN label must be excluded from the mean, not scored as if the count were 0."""
    loss_fn = CountLikelihoodLoss(noise="nb")
    with_nan = loss_fn(y_true=Y_TRUE, log_mu=LOG_MU, log_noise=LOG_NOISE)

    y_zero_filled = torch.nan_to_num(Y_TRUE, nan=0.0)
    without_nan = loss_fn(y_true=y_zero_filled, log_mu=LOG_MU, log_noise=LOG_NOISE)

    assert with_nan.item() != pytest.approx(without_nan.item())

    # Masking every label should raise via a zero-division guard rather than NaN/inf.
    all_nan = torch.full_like(Y_TRUE, float("nan"))
    value = loss_fn(y_true=all_nan, log_mu=LOG_MU, log_noise=LOG_NOISE)
    assert math.isfinite(value.item())
    assert value.item() == pytest.approx(0.0)


def test_noise_clamped_to_bounds() -> None:
    loss_fn = CountLikelihoodLoss(noise="nb")
    low, high = COUNT_NOISE_MODELS["nb"]
    unclamped = torch.tensor([low - 5.0, high + 5.0, 0.0])
    clamped = loss_fn.clamp_noise(unclamped)
    assert clamped[0].item() == pytest.approx(low)
    assert clamped[1].item() == pytest.approx(high)
    assert clamped[2].item() == pytest.approx(0.0)


def test_invalid_noise_name_rejected() -> None:
    with pytest.raises(ValueError, match="noise must be one of"):
        CountLikelihoodLoss(noise="not-a-noise-model")


def test_contrast_term_requires_group_arguments() -> None:
    loss_fn = CountLikelihoodLoss(noise="nb", contrast_weight=1.0)
    with pytest.raises(ValueError, match="contrast term needs"):
        loss_fn(y_true=Y_TRUE, log_mu=LOG_MU, log_noise=LOG_NOISE)


def test_loss_is_differentiable_wrt_log_mu_and_log_noise() -> None:
    log_mu = LOG_MU.clone().requires_grad_(True)
    log_noise = LOG_NOISE.clone().requires_grad_(True)
    loss_fn = CountLikelihoodLoss(noise="nb")
    value = loss_fn(y_true=Y_TRUE, log_mu=log_mu, log_noise=log_noise)
    value.backward()
    assert log_mu.grad is not None
    assert torch.isfinite(log_mu.grad).all()
    assert log_noise.grad is not None
    assert torch.isfinite(log_noise.grad).all()
