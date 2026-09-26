"""Unit tests for regulonado.training.cached.metrics.

Expected numeric values were computed by running the same toy ``(predictions, labels)``
pair through UEF's ``unique_enhancer_finding.modelling.metrics.GroupedCountMetrics``
directly (via its own venv) and hard-coded here -- see the plan for the scratch script
used, including the exact RNG-generated toy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from regulonado.training.cached.metrics import GroupedCountMetrics, group_count_rates

TRACK_GROUPS = [0, 0, 1]
LOG_SIZE_FACTORS = torch.tensor([0.1, -0.1, 0.0])


def _toy_predictions_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """Reproduces the exact toy data the UEF reference values were computed from."""
    n_regions = 40
    n_groups = 2
    rng = np.random.default_rng(42)
    predictions = rng.gamma(shape=2.0, scale=1.0, size=(n_regions, n_groups)).astype(np.float32)
    labels = np.zeros((n_regions, 3), dtype=np.float32)
    size = np.exp(np.asarray(LOG_SIZE_FACTORS))
    for i in range(n_regions):
        rate0 = predictions[i, 0] * (1 + 0.2 * rng.standard_normal())
        rate1 = predictions[i, 1] * (1 + 0.2 * rng.standard_normal())
        labels[i, 0] = max(0.0, rng.poisson(rate0 * size[0]))
        labels[i, 1] = max(0.0, rng.poisson(rate0 * size[1]))
        labels[i, 2] = max(0.0, rng.poisson(rate1 * size[2]))
    labels[3, 1] = np.nan
    labels[7, 2] = np.nan
    return predictions, labels


def test_grouped_count_metrics_matches_uef() -> None:
    predictions, labels = _toy_predictions_and_labels()
    metric = GroupedCountMetrics(
        group_names=["A", "B"],
        track_groups=TRACK_GROUPS,
        log_size_factors=LOG_SIZE_FACTORS,
        top_decile_task="A",
    )
    result = metric((predictions, labels))

    expected = {
        "contrast_pearson_A": 0.6028838958500513,
        "contrast_pearson_B": 0.6028838958500512,
        "contrast_pearson_mean": 0.6028838958500513,
        "contrast_pearson_top_decile_A": 0.885126971910325,
        "pearson_A": 0.7924898343484009,
        "pearson_B": 0.46272687745492613,
        "pearson_mean": 0.6276083559016635,
        "spearman_A": 0.7451815605163574,
        "spearman_B": 0.4731352925300598,
        "spearman_mean": 0.6091583967208862,
    }
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=1e-6), key


def test_metric_names_follow_uef_convention() -> None:
    """Names must match UEF exactly (no prefix/suffix drift) so runs compare directly."""
    predictions, labels = _toy_predictions_and_labels()
    metric = GroupedCountMetrics(
        group_names=["HL-60", "K562"],
        track_groups=TRACK_GROUPS,
        log_size_factors=LOG_SIZE_FACTORS,
        top_decile_task="HL-60",
    )
    result = metric((predictions, labels))
    for key in (
        "contrast_pearson_mean",
        "contrast_pearson_HL-60",
        "contrast_pearson_K562",
        "contrast_pearson_top_decile_HL-60",
    ):
        assert key in result


def test_metric_reset_between_calls_does_not_accumulate_state() -> None:
    """A single GroupedCountMetrics instance is reused by Trainer across evals."""
    predictions, labels = _toy_predictions_and_labels()
    metric = GroupedCountMetrics(
        group_names=["A", "B"], track_groups=TRACK_GROUPS, log_size_factors=LOG_SIZE_FACTORS
    )
    first = metric((predictions, labels))
    second = metric((predictions, labels))
    assert first["pearson_mean"] == pytest.approx(second["pearson_mean"])


def test_group_count_rates_ignores_nan_replicates() -> None:
    counts = np.array([[1.0, float("nan"), 4.0]])
    log_size_factors = LOG_SIZE_FACTORS.numpy()
    rates = group_count_rates(
        counts, track_groups=np.array(TRACK_GROUPS), log_size_factors=log_size_factors, n_groups=2
    )
    size0 = np.exp(log_size_factors[0])
    assert rates[0, 0] == pytest.approx(1.0 / size0)
    size2 = np.exp(log_size_factors[2])
    assert rates[0, 1] == pytest.approx(4.0 / size2)


def test_group_count_rates_all_masked_is_nan() -> None:
    counts = np.array([[float("nan"), float("nan"), 4.0]])
    log_size_factors = LOG_SIZE_FACTORS.numpy()
    rates = group_count_rates(
        counts, track_groups=np.array(TRACK_GROUPS), log_size_factors=log_size_factors, n_groups=2
    )
    assert np.isnan(rates[0, 0])
    assert np.isfinite(rates[0, 1])
