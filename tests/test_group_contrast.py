from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from regulonado.target_specificity import region_statistic, target_specificity_score
from regulonado.training.group_contrast import (
    group_contrast_labels,
    group_replicate_weights,
    reduce_target_score,
    smooth_bins,
)

# ---------------------------------------------------------------------------
# 1. Decisive cross-check against target_specificity_score
# ---------------------------------------------------------------------------


def test_group_contrast_label_max_matches_target_specificity_score():
    """max_l reduce_target_score(labels)[l] == target_specificity_score(..., "rolling-max").

    Constructed so the check is exact, not approximate:

    - Every track is a flat "plateau" of its own amplitude over bins [60, 140) and a flat,
      shared low baseline everywhere else. The plateau (80 bins) is wider than the
      smoothing/rolling-max window (31 bins == round(1000 bp / 32 bp)), so there is a
      50-bin interior band (bins ~[75, 125)) where a rolling window of width 31 sits
      entirely inside the plateau, for *both* this module's same-length (edge-padded)
      :func:`smooth_bins` and target_specificity's unpadded (mode="valid") rolling mean.
      On that band both give exactly the track's plateau amplitude, so the two pipelines
      agree bin-for-bin there, and the low, shared baseline elsewhere keeps the true
      maximum of both curves inside that band (never at a padded edge, where the two
      smoothings could disagree).
    - scale_factor=1, background=None ("scale-only") makes anchor units equal the raw
      profile values, so the same numpy array feeds both pipelines directly.
    - background=None on both sides also means the gauge subtraction in
      ``group_contrast_labels`` is the only extra step versus ``target_specificity_score``,
      and it exactly cancels inside ``reduce_target_score`` (see test 2), so no gauge
      choice can perturb this comparison.
    """
    length = 200
    amplitudes = [50.0, 5.0, 2.0, 8.0]  # target, ref1, ref2, ref3
    baseline = 0.05
    bin_size = 32
    window_stat_bp = 1000
    width = max(1, round(window_stat_bp / bin_size))
    assert width == 31

    profiles = np.full((len(amplitudes), length), baseline, dtype=np.float32)
    for t, amp in enumerate(amplitudes):
        profiles[t, 60:140] = amp

    # --- numpy reference ---
    stats = region_statistic(
        profiles[np.newaxis], np.array([length]), bin_size=bin_size,
        window_stat_bp=window_stat_bp, stat="rolling-max",
    )
    group_index_np = np.array([0, 1, 2, 3])
    group_names = ["target", "ref1", "ref2", "ref3"]
    reference = target_specificity_score(
        stats,
        scale_factor=np.ones(len(amplitudes)),
        background=np.zeros(len(amplitudes)),
        group_index=group_index_np,
        group_names=group_names,
        target_group="target",
        background_mode="scale-only",
        pseudocount=0.1,
        reduction="quantile",
        quantile=0.9,
    )

    # --- torch pipeline under test ---
    normalised_target = torch.tensor(profiles, dtype=torch.float32)[None]  # [1, T, L]
    group_weights = group_replicate_weights(torch.tensor(group_index_np), n_groups=4)
    labels, _mask = group_contrast_labels(
        normalised_target,
        group_weights=group_weights,
        background=None,
        smoothing_bins=width,
        pseudocount=0.1,
        gauge="median",
        floor=0.139,
        clamp=(-100.0, 100.0),  # wide enough that clamping never engages here
    )
    score = reduce_target_score(labels, target_index=0, quantile=0.9)

    np.testing.assert_allclose(score.max().item(), reference.score[0], rtol=0, atol=2e-6)


# ---------------------------------------------------------------------------
# 2. Gauge cancellation
# ---------------------------------------------------------------------------


def test_gauge_choice_cancels_in_channel_differences():
    torch.manual_seed(1)
    normalised_target = torch.rand(2, 6, 40) * 5.0
    group_index = torch.tensor([0, 0, 1, 2, 2, 3])
    group_weights = group_replicate_weights(group_index, n_groups=4)

    labels_median, _ = group_contrast_labels(
        normalised_target, group_weights=group_weights, gauge="median", clamp=(-100.0, 100.0)
    )
    labels_mean, _ = group_contrast_labels(
        normalised_target, group_weights=group_weights, gauge="mean", clamp=(-100.0, 100.0)
    )

    diff_median = labels_median[:, 0] - labels_median[:, 2]
    diff_mean = labels_mean[:, 0] - labels_mean[:, 2]
    torch.testing.assert_close(diff_median, diff_mean)


# ---------------------------------------------------------------------------
# 3. Replicate-duplication invariance
# ---------------------------------------------------------------------------


def test_duplicating_a_replicate_track_leaves_labels_unchanged():
    torch.manual_seed(2)
    normalised_target = torch.rand(1, 3, 20) * 3.0  # tracks: target(0), refA(1), refB(2)
    group_index = torch.tensor([0, 1, 2])
    group_weights = group_replicate_weights(group_index, n_groups=3)
    labels, mask = group_contrast_labels(
        normalised_target, group_weights=group_weights, clamp=(-100.0, 100.0)
    )

    # Duplicate track 1 (refA) as a second replicate of the same group.
    dup_target = torch.cat([normalised_target, normalised_target[:, 1:2]], dim=1)
    dup_group_index = torch.tensor([0, 1, 2, 1])
    dup_weights = group_replicate_weights(dup_group_index, n_groups=3)
    dup_labels, dup_mask = group_contrast_labels(
        dup_target, group_weights=dup_weights, clamp=(-100.0, 100.0)
    )

    torch.testing.assert_close(dup_labels, labels)
    torch.testing.assert_close(dup_mask, mask)


# ---------------------------------------------------------------------------
# 4. Linear-space averaging (hand-computed)
# ---------------------------------------------------------------------------


def test_group_averaging_is_linear_not_log_space():
    # Group 0: two tracks, constant values 6.0 and 10.0 -> linear mean 8.0.
    # Group 1: one track, constant value 4.0 (arbitrary reference so there are >=2 groups).
    normalised_target = torch.tensor(
        [[[6.0, 6.0], [10.0, 10.0], [4.0, 4.0]]], dtype=torch.float32
    )  # [1, T=3, L=2]
    group_index = torch.tensor([0, 0, 1])
    group_weights = group_replicate_weights(group_index, n_groups=2)

    labels, _mask = group_contrast_labels(
        normalised_target,
        group_weights=group_weights,
        smoothing_bins=1,
        pseudocount=0.1,
        gauge="mean",
        clamp=(-100.0, 100.0),
    )

    linear_mean_log = math.log2(8.0 + 0.1)
    other_log = math.log2(4.0 + 0.1)
    geometric_mean_log = 0.5 * (math.log2(6.0 + 0.1) + math.log2(10.0 + 0.1))
    # The averaging is linear (8.0), not geometric/log-space (would give a different value).
    assert not math.isclose(linear_mean_log, geometric_mean_log, rel_tol=1e-6)

    gauge = 0.5 * (linear_mean_log + other_log)
    expected_group0 = linear_mean_log - gauge
    np.testing.assert_allclose(labels[0, 0].numpy(), [expected_group0, expected_group0], atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Mask
# ---------------------------------------------------------------------------


def test_mask_all_groups_at_floor_is_zero_one_group_above_is_one():
    floor = 0.139
    # Bin 0: every group exactly at the floor (not above it) -> mask 0.
    # Bin 1: group 0 well above the floor, group 1 still exactly at the floor -> mask 1,
    #        and group 1's own floor value is not itself masked (see labels[0,1,1]).
    normalised_target = torch.tensor(
        [[[floor, 5.0], [floor, floor]]], dtype=torch.float32
    )  # [1, T=2, L=2], track t == group t
    group_index = torch.tensor([0, 1])
    group_weights = group_replicate_weights(group_index, n_groups=2)

    labels, mask = group_contrast_labels(
        normalised_target,
        group_weights=group_weights,
        smoothing_bins=1,
        floor=floor,
        clamp=(-100.0, 100.0),
    )

    assert mask[0, 0, 0].item() == 0.0
    assert mask[0, 0, 1].item() == 1.0
    # Group 1 (index 1) sits at its own floor at bin 1 but is not masked at the channel
    # level -- the mask is a single [B, 1, L] channel, and its label there still carries
    # the (very informative) contrast against group 0's high signal.
    assert torch.isfinite(labels[0, 1, 1])


# ---------------------------------------------------------------------------
# 6. reduce_target_score
# ---------------------------------------------------------------------------


def test_reduce_target_score_quantile_one_is_max_over_others():
    channels = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.5, 5.0], [2.9, 4.0]]])  # [1, C=4, L=2]
    score = reduce_target_score(channels, target_index=0, quantile=1.0)
    expected = channels[0, 0] - channels[0, 1:].max(dim=0).values
    torch.testing.assert_close(score[0], expected)


def test_reduce_target_score_single_channel_raises():
    channels = torch.zeros(1, 1, 5)
    with pytest.raises(ValueError):
        reduce_target_score(channels, target_index=0, quantile=0.9)


# ---------------------------------------------------------------------------
# 7. Differentiability
# ---------------------------------------------------------------------------


def test_reduce_target_score_is_differentiable():
    channels = torch.rand(2, 5, 10, requires_grad=True)
    score = reduce_target_score(channels, target_index=0, quantile=0.9)
    score.sum().backward()
    assert channels.grad is not None
    assert torch.isfinite(channels.grad).all()
    assert (channels.grad != 0).any()


def test_group_contrast_labels_is_differentiable():
    normalised_target = (torch.rand(1, 4, 20) * 2.0).requires_grad_(True)
    group_index = torch.tensor([0, 0, 1, 2])
    group_weights = group_replicate_weights(group_index, n_groups=3)
    labels, _mask = group_contrast_labels(normalised_target, group_weights=group_weights)
    labels.sum().backward()
    assert normalised_target.grad is not None
    assert torch.isfinite(normalised_target.grad).all()
    assert (normalised_target.grad != 0).any()


# ---------------------------------------------------------------------------
# 8. smooth_bins
# ---------------------------------------------------------------------------


def test_smooth_bins_preserves_length_and_width_one_is_identity():
    signal = torch.rand(2, 3, 17)
    smoothed = smooth_bins(signal, width=1)
    torch.testing.assert_close(smoothed, signal)

    smoothed5 = smooth_bins(signal, width=5)
    assert smoothed5.shape == signal.shape


def test_smooth_bins_is_a_rolling_mean_on_a_ramp():
    # A linear ramp's centred rolling mean at an interior point equals the ramp's own value
    # there (the window is symmetric around a linear function), so this pins the "mean, not
    # max" requirement independently of the cross-check in test 1.
    signal = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20)
    smoothed = smooth_bins(signal, width=5)
    # Interior points (away from the replicate-padded edges) equal the ramp value exactly.
    torch.testing.assert_close(smoothed[0, 0, 5:15], signal[0, 0, 5:15])


# ---------------------------------------------------------------------------
# 9. Clamp
# ---------------------------------------------------------------------------


def test_clamp_bounds_are_respected():
    torch.manual_seed(3)
    normalised_target = torch.rand(2, 10, 30) * 1000.0  # wide dynamic range
    group_index = torch.arange(10)
    group_weights = group_replicate_weights(group_index, n_groups=10)
    labels, _mask = group_contrast_labels(
        normalised_target, group_weights=group_weights, clamp=(-1.0, 1.0)
    )
    assert labels.min().item() >= -1.0
    assert labels.max().item() <= 1.0


# ---------------------------------------------------------------------------
# gauge / smoothing_bins validation
# ---------------------------------------------------------------------------


def test_invalid_gauge_raises():
    normalised_target = torch.rand(1, 2, 5)
    group_weights = group_replicate_weights(torch.tensor([0, 1]), n_groups=2)
    with pytest.raises(ValueError):
        group_contrast_labels(normalised_target, group_weights=group_weights, gauge="mode")
