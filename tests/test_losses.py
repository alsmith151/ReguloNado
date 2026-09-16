"""Unit tests for regulonado.training.losses."""

from __future__ import annotations

import pytest
import torch
from regulonado.training.losses import (
    contrast_group_weights,
    kendall_track_weighted_loss,
    log1p_huber_loss,
    paired_binwise_log2fc_loss,
    poisson_multinomial_binwise_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    specificity_stats,
    topk_additive_loss,
    topk_reweight_loss,
    track_contrast_correlation_loss,
    track_contrast_magnitude_loss,
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


def test_poisson_multinomial_reduction_none_shape_and_matches_mean() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    per_track = poisson_multinomial_loss(pred, tgt, reduction="none")
    assert per_track.shape == (T,)
    scalar = poisson_multinomial_loss(pred, tgt)
    assert abs(per_track.mean().item() - scalar.item()) < 1e-4


def test_poisson_multinomial_binwise_reduction_none_shape_and_matches_mean() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    per_track = poisson_multinomial_binwise_loss(pred, tgt, reduction="none")
    assert per_track.shape == (T,)
    scalar = poisson_multinomial_binwise_loss(pred, tgt)
    assert abs(per_track.mean().item() - scalar.item()) < 1e-4


def test_kendall_track_weighted_loss_zero_log_var_matches_mean() -> None:
    per_track = torch.rand(T) + 0.1
    log_var = torch.zeros(T)
    weighted = kendall_track_weighted_loss(per_track, log_var)
    assert abs(weighted.item() - per_track.mean().item()) < 1e-6


def test_kendall_track_weighted_loss_gradient_flows_to_log_var() -> None:
    per_track = (torch.rand(T) + 0.1).detach()
    log_var = torch.zeros(T, requires_grad=True)
    kendall_track_weighted_loss(per_track, log_var).backward()
    assert log_var.grad is not None
    assert torch.all(torch.isfinite(log_var.grad))


def test_kendall_track_weighted_loss_discourages_trivial_weight_collapse() -> None:
    # Driving log_var very negative (huge precision) blows up the per-track loss term
    # unless the true per-track loss is ~0; the log_var regulariser prevents a track
    # from freely shrinking its weight to escape a real loss.
    per_track = torch.full((T,), 1.0)
    low_var = kendall_track_weighted_loss(per_track, torch.full((T,), -5.0))
    zero_var = kendall_track_weighted_loss(per_track, torch.zeros(T))
    assert low_var.item() > zero_var.item()


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


# ---------------------------------------------------------------------------
# contrast_group_weights / track_contrast_correlation_loss
# ---------------------------------------------------------------------------


def test_contrast_group_weights_average_replicates_and_drop_single_group_families() -> None:
    weights = contrast_group_weights(["A", "A", "A", "B", None], ["x", "x", "y", "z", "w"])
    assert weights.shape == (1, 2, 5)
    torch.testing.assert_close(weights[0, 0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(weights[0, 1], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


def test_contrast_group_weights_pad_smaller_families() -> None:
    weights = contrast_group_weights(["A", "A", "A", "B", "B"], ["x", "y", "z", "x", "y"])
    assert weights.shape == (2, 3, 5)
    assert weights[1, 2].eq(0).all()


def test_contrast_group_weights_empty_without_contrast() -> None:
    assert contrast_group_weights([None, None], ["x", "y"]).shape == (0, 0, 2)


def _contrast_weights() -> torch.Tensor:
    return contrast_group_weights(["A", "A", "A", "A"], ["g1", "g2", "g3", "g3"])


def _family_mean_weights() -> torch.Tensor:
    """Group-balanced family-mean weights ``[T]`` for the single ``_contrast_weights`` family."""
    groups = _contrast_weights()[0]
    return groups.sum(0) / groups.shape[0]


def _region_expand(region_values: torch.Tensor, region_bins: int) -> torch.Tensor:
    """Expand per-(batch, track) region totals into [B, T, region_bins] with a uniform
    distribution across bins, so summing bins recovers ``region_values`` exactly."""
    expanded = region_values / region_bins
    return expanded[..., None].expand(*region_values.shape, region_bins).contiguous()


def _log_linear_batch(
    k: float, *, n_batches: int = 6, region_bins: int = 8, pseudocount: float = 0.1, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (pred, target) with predicted specificity an exact ``k``-scaling of observed
    specificity, using region_bins == length so each example is a single active region.
    The offset is negligible at this magnitude, so the log-linear relationship built here
    survives the implementation's own log/exp round trip almost exactly."""
    torch.manual_seed(seed)
    weights = _family_mean_weights()
    offset = pseudocount * region_bins
    t_values = torch.rand(n_batches, T) * 50 + 50
    log_t = torch.log(t_values + offset)
    mean_log_t = (weights * log_t).sum(dim=-1, keepdim=True)
    c_t = log_t - mean_log_t
    log_p = k * c_t + mean_log_t
    p_values = torch.exp(log_p) - offset
    target = _region_expand(t_values, region_bins)
    pred = _region_expand(p_values, region_bins).requires_grad_(True)
    return pred, target


def test_track_contrast_correlation_perfect_pred_near_zero() -> None:
    tgt = _rand_pos(B, T, L)
    loss = track_contrast_correlation_loss(
        tgt.detach().clone(), tgt, _contrast_weights(), region_bins=8, active_fraction=1.0
    )
    assert loss.shape == ()
    assert loss.item() < 1e-4


def test_track_contrast_correlation_ignores_shared_scale() -> None:
    tgt = torch.rand(B, T, L) * 50 + 50
    loss = track_contrast_correlation_loss(
        3.0 * tgt.detach(), tgt, _contrast_weights(), region_bins=8, active_fraction=1.0
    )
    assert loss.item() < 1e-3


def test_track_contrast_correlation_amplitude_compressed_by_half_still_near_zero() -> None:
    """A correlation objective doesn't constrain amplitude: halving the predicted
    specificity spread relative to the observed one still gives a near-perfect loss."""
    pred, target = _log_linear_batch(k=0.5, region_bins=8)
    loss = track_contrast_correlation_loss(pred, target, _contrast_weights(), region_bins=8)
    assert loss.item() < 1e-3


def test_track_contrast_correlation_negated_deviations_gives_loss_two() -> None:
    pred, target = _log_linear_batch(k=-1.0, region_bins=8)
    loss = track_contrast_correlation_loss(pred, target, _contrast_weights(), region_bins=8)
    assert loss.item() == pytest.approx(2.0, abs=1e-2)


def test_track_contrast_correlation_single_example_masked_finite_zero() -> None:
    # With one example and one region, every track's per-track sample count is 1, so
    # variance (and the correlation denominator) is identically zero for every track.
    pred = _rand_pos(1, T, L)
    tgt = _rand_pos(1, T, L)
    loss = track_contrast_correlation_loss(pred, tgt, _contrast_weights(), region_bins=L)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_track_contrast_correlation_gradient_finite_and_nonzero() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    loss = track_contrast_correlation_loss(pred, tgt, _contrast_weights(), region_bins=8)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).any()


def test_track_contrast_correlation_without_families_is_zero_with_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    loss = track_contrast_correlation_loss(pred, tgt, torch.zeros(0, 0, T), region_bins=8)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert pred.grad is not None


def test_specificity_stats_active_selection_uses_observed_signal_only() -> None:
    """Scrambling predictions outside the observed-active region set must not change the
    stats: active-region selection is driven by target signal, not predictions."""
    torch.manual_seed(3)
    weights = _contrast_weights()
    region_bins = 8
    pred = _rand_pos(B, T, region_bins * 4)
    tgt = _rand_pos(B, T, region_bins * 4)
    kwargs = dict(region_bins=region_bins, pseudocount=0.1, active_fraction=0.25)
    baseline = specificity_stats(pred, tgt, weights, **kwargs)

    # Identify inactive regions per (batch, family) the same way the implementation does,
    # then scramble pred there and confirm the stats are unchanged.
    with torch.no_grad():
        scrambled = pred.detach().clone()
        n_regions = scrambled.shape[-1] // region_bins
        usable_t = tgt.detach()[..., : n_regions * region_bins]
        region_t = usable_t.reshape(B, T, n_regions, region_bins).sum(-1)
        group_signal = torch.einsum("gt,btr->bgr", weights[0], region_t.clamp_min(0.0))
        family_signal = group_signal.max(dim=1).values
        n_active = max(1, round(0.25 * n_regions))
        threshold = family_signal.topk(n_active, dim=-1).values[:, -1:]
        active = (family_signal >= threshold) & (family_signal > 0)
        inactive_regions = (~active).nonzero(as_tuple=False)  # [(b, r), ...]
        for b, r in inactive_regions.tolist():
            start = r * region_bins
            scrambled[b, :, start : start + region_bins] = torch.rand(T, region_bins) + 10.0

    scrambled_stats = specificity_stats(scrambled, tgt, weights, **kwargs)
    torch.testing.assert_close(baseline, scrambled_stats)


def test_specificity_stats_ranks_regions_by_strongest_group() -> None:
    """A region open in one cell type outranks a shared peak with a higher family mean."""
    region_bins, n_groups = 4, 4
    weights = contrast_group_weights(["A"] * n_groups, [f"g{i}" for i in range(n_groups)])
    # Columns are regions. Region 0: shared peak, 10 in every group (mean 10, max 10).
    # Region 1: specific to g0, 30 there and 0 elsewhere (mean 7.5, max 30).
    # Region 2: background, 1 everywhere.
    region_t = torch.tensor(
        [[10.0, 30.0, 1.0], [10.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 0.0, 1.0]]
    )
    target = _region_expand(region_t[None], region_bins).reshape(1, n_groups, -1)
    stats = specificity_stats(
        target, target, weights, region_bins=region_bins, pseudocount=0.1, active_fraction=1 / 3
    )
    # One active region, and it is the specific one: g0 sits above the family mean.
    assert stats[0, :, 5].eq(1.0).all()
    assert stats[0, 0, 1] > 0
    assert (stats[0, 1:, 1] < 0).all()


# ---------------------------------------------------------------------------
# track_contrast_magnitude_loss
# ---------------------------------------------------------------------------


def test_contrast_magnitude_zero_for_perfect_prediction() -> None:
    tgt = _rand_pos(B, T, L)
    loss = track_contrast_magnitude_loss(
        tgt.detach().clone(), tgt, _contrast_weights(), region_bins=8, active_fraction=1.0
    )
    assert loss.shape == ()
    assert loss.item() < 1e-6


def test_contrast_magnitude_penalises_compression_that_correlation_ignores() -> None:
    """Halving the predicted specificity spread: correlation ~perfect, magnitude positive."""
    pred, target = _log_linear_batch(k=0.5, region_bins=8)
    kwargs = dict(region_bins=8, active_fraction=1.0)
    correlation = track_contrast_correlation_loss(pred, target, _contrast_weights(), **kwargs)
    magnitude = track_contrast_magnitude_loss(pred, target, _contrast_weights(), **kwargs)
    assert correlation.item() < 1e-3
    assert magnitude.item() > 1e-3


def test_contrast_magnitude_grows_with_compression() -> None:
    kwargs = dict(region_bins=8, active_fraction=1.0)
    losses = [
        track_contrast_magnitude_loss(
            *_log_linear_batch(k=k, region_bins=8), _contrast_weights(), **kwargs
        ).item()
        for k in (0.9, 0.5, 0.0)
    ]
    assert losses[0] < losses[1] < losses[2]


def test_contrast_magnitude_gradient_finite_and_nonzero() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    track_contrast_magnitude_loss(pred, tgt, _contrast_weights(), region_bins=8).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).any()


def test_contrast_magnitude_without_families_is_zero_with_gradient() -> None:
    pred = _rand_pos(B, T, L)
    tgt = _rand_pos(B, T, L)
    loss = track_contrast_magnitude_loss(pred, tgt, torch.zeros(0, 0, T), region_bins=8)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert pred.grad is not None
