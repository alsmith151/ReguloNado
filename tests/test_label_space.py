"""Count label space (raw counts + exposure offset) and missing-bin masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from regulonado.dataset.build import make_transform, transform_signal
from regulonado.training.label_space import resolve_count_label_space, validate_label_space
from regulonado.training.losses import (
    contrast_group_weights,
    poisson_multinomial_binwise_loss,
    track_contrast_correlation_loss,
)
from regulonado.training.metrics import make_preprocess_logits_for_metrics
from regulonado.training.runner import _build_loss_fn, _empirical_track_output_bias

BIN_SIZE = 32


def _records(**overrides) -> list[dict]:
    base = [
        {"track_name": "a", "anchor_reference": 10.0, "background": 2.0, "fragment_length": 160.0},
        {"track_name": "b", "anchor_reference": 20.0, "background": 4.0, "fragment_length": 80.0},
    ]
    for field, values in overrides.items():
        for record, value in zip(base, values):
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
    return base


# ---------------------------------------------------------------------------
# resolve_count_label_space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count_unit", "expected"),
    [
        ("fragments", [BIN_SIZE / 160.0, BIN_SIZE / 80.0]),
        ("coverage_sum", [BIN_SIZE, BIN_SIZE]),
        ("mean_coverage", [1.0, 1.0]),
    ],
)
def test_count_factors_per_unit(count_unit, expected):
    space = resolve_count_label_space(
        _records(), count_unit=count_unit, exposure="none", bin_size=BIN_SIZE
    )
    np.testing.assert_allclose(space.count_factors, expected)


def test_fragment_length_derived_from_genome_sum_and_library_size():
    records = _records(
        fragment_length=[None, None], genome_sum=[1.6e9, 8e8], library_size=[1e7, 1e7]
    )
    space = resolve_count_label_space(
        records, count_unit="fragments", exposure="none", bin_size=BIN_SIZE
    )
    np.testing.assert_allclose(space.count_factors, [BIN_SIZE / 160.0, BIN_SIZE / 80.0])


def test_fragments_without_length_names_the_track():
    with pytest.raises(ValueError, match="fragment_length.*missing for: b"):
        resolve_count_label_space(
            _records(fragment_length=[160.0, None]),
            count_unit="fragments",
            exposure="none",
            bin_size=BIN_SIZE,
        )


@pytest.mark.parametrize(
    ("exposure", "expected_in_mean_coverage"),
    [
        ("anchor", [10.0, 20.0]),
        ("anchor_minus_background", [8.0, 16.0]),
        ("library_size", [20.0, 40.0]),
    ],
)
def test_exposure_is_in_count_units(exposure, expected_in_mean_coverage):
    records = _records(library_size=[2e7, 4e7])
    space = resolve_count_label_space(
        records, count_unit="fragments", exposure=exposure, bin_size=BIN_SIZE
    )
    np.testing.assert_allclose(
        space.exposure, np.asarray(expected_in_mean_coverage) * space.count_factors, rtol=1e-6
    )


def test_exposure_none_predicts_count_units():
    space = resolve_count_label_space(
        _records(), count_unit="coverage_sum", exposure="none", bin_size=BIN_SIZE
    )
    np.testing.assert_array_equal(space.exposure, [1.0, 1.0])


def test_non_positive_exposure_is_rejected():
    with pytest.raises(ValueError, match="not positive for tracks: a"):
        resolve_count_label_space(
            _records(background=[10.0, 4.0]),
            count_unit="mean_coverage",
            exposure="anchor_minus_background",
            bin_size=BIN_SIZE,
        )


def test_unknown_label_space_option_is_rejected():
    with pytest.raises(ValueError, match="data.exposure"):
        validate_label_space("counts", "fragments", "cpm")


# ---------------------------------------------------------------------------
# Read-time transform
# ---------------------------------------------------------------------------


def _example(signal: np.ndarray) -> dict:
    return {"sequence_tokens": np.zeros(8, dtype=np.uint8), "signal": signal}


def _counts_transform(**kwargs):
    return make_transform(
        np.array([0.5, 0.5], dtype=np.float32),  # scale/clip/background must be ignored
        1.0,
        1.0,
        np.array([100.0, 100.0], dtype=np.float32),
        context_length=8,
        n_pred_bins=4,
        bin_size=2,
        label_space="counts",
        count_factors=np.array([2.0, 3.0], dtype=np.float32),
        **kwargs,
    )


def test_counts_transform_only_converts_units_and_keeps_missing():
    signal = np.array([[1.0, np.nan, 50.0, -1.0], [0.0, 2.0, 4.0, np.nan]], dtype=np.float32)
    labels = _counts_transform()(_example(signal))["labels"]
    expected = np.array([[2.0, np.nan, 100.0, 0.0], [0.0, 6.0, 12.0, np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(labels, expected)


def test_counts_transform_zeroes_missing_when_masking_is_off():
    signal = np.array([[1.0, np.nan, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    labels = _counts_transform(mask_missing=False)(_example(signal))["labels"]
    assert not np.isnan(labels).any()
    assert labels[0, 1] == 0.0


def test_counts_transform_requires_count_factors():
    with pytest.raises(ValueError, match="count_factors"):
        make_transform(np.ones(2), 1.0, 1.0, label_space="counts")


@pytest.mark.parametrize("keep_missing", [False, True])
def test_transformed_signal_missing_bins(keep_missing):
    signal = np.array([[1.0, np.nan]], dtype=np.float32)
    out = transform_signal(signal, np.ones(1), 10.0, 20.0, keep_missing=keep_missing)
    assert np.isnan(out[0, 1]) == keep_missing
    assert np.isfinite(out[0, 0])


# ---------------------------------------------------------------------------
# Loss: exposure offset and masking
# ---------------------------------------------------------------------------

LOSS_CFG = {"name": "poisson_multinomial_binwise", "poisson_weight": 0.5}


def _pred_target(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, 2, 16, generator=generator) + 0.1
    target = torch.poisson(torch.full((2, 2, 16), 3.0), generator=generator)
    exposure = torch.tensor([4.0, 0.5])
    return pred, target, exposure


def _loss_fn(**kwargs):
    return _build_loss_fn(
        {**LOSS_CFG, **kwargs.pop("loss_cfg", {})},
        scale_factors=np.ones(2),
        clip_hard=np.ones(2),
        labels_already_scaled=True,
        **kwargs,
    )


def test_base_loss_sees_rate_times_exposure_against_counts():
    pred, target, exposure = _pred_target()
    loss = _loss_fn(exposure=exposure)(pred, target)
    expected = poisson_multinomial_binwise_loss(
        pred * exposure[:, None], target, poisson_weight=0.5
    )
    torch.testing.assert_close(loss, expected)


def test_contrast_term_compares_rates_with_exposure_normalised_targets():
    pred, target, exposure = _pred_target()
    weights = contrast_group_weights(["ATAC", "ATAC"], ["g1", "g2"])
    geometry = {"contrast_region_bins": 4, "contrast_active_fraction": 1.0}
    loss_fn = _loss_fn(
        loss_cfg={"contrast_weight": 1.0}, exposure=exposure, contrast_weights=weights, **geometry
    )
    expected = poisson_multinomial_binwise_loss(
        pred * exposure[:, None], target, poisson_weight=0.5
    ) + track_contrast_correlation_loss(
        pred,
        target / exposure[:, None],
        weights,
        region_bins=4,
        pseudocount=0.1,
        active_fraction=1.0,
    )
    torch.testing.assert_close(loss_fn(pred, target), expected)


def test_missing_bins_drop_out_of_loss_and_gradient():
    pred, target, exposure = _pred_target()
    target[0, 1, 3:7] = float("nan")
    pred.requires_grad_(True)
    loss = _loss_fn(exposure=exposure)(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(pred.grad[0, 1, 3:7] == 0)
    assert torch.all(pred.grad[0, 1, :3] != 0)


def test_scaled_poisson_multinomial_is_rejected_in_counts_space():
    with pytest.raises(ValueError, match="applies its own scaling"):
        _loss_fn(loss_cfg={"name": "scaled_poisson_multinomial"}, exposure=torch.ones(2))


# ---------------------------------------------------------------------------
# Metrics and output-bias init work in output units without missing bins
# ---------------------------------------------------------------------------


def test_metrics_divide_labels_and_exclude_missing_bins():
    pred, target, exposure = _pred_target()
    target[1, 0, :4] = float("nan")
    stats = make_preprocess_logits_for_metrics(topk_bins=4, label_divisor=exposure)(pred, target)

    normalised = target / exposure[:, None]
    valid = ~torch.isnan(normalised)
    assert stats[1, 0, 5] == 12  # n: 16 bins minus 4 missing
    assert stats[0, 0, 5] == 16
    torch.testing.assert_close(stats[..., 1], torch.nansum(normalised, dim=-1))  # sum_t
    torch.testing.assert_close(stats[..., 0], (pred * valid).sum(-1))  # sum_p


def test_empirical_output_bias_uses_output_units_and_skips_missing():
    labels = np.array([[8.0, np.nan, 8.0, 8.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    biases = _empirical_track_output_bias(
        [{"labels": labels}],
        n_tracks=2,
        activation_type="identity",
        max_samples=1,
        label_divisor=np.array([4.0, 0.5]),
    )
    np.testing.assert_allclose(biases, [2.0, 2.0])
