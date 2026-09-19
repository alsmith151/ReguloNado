"""Tests for Stage 4 of the group-contrast-head plan: group-space evaluation metrics.

Covers:
  - the tuple-return mechanism `make_preprocess_logits_for_metrics` uses to carry group
    channels through the eval loop, pinned directly against the installed transformers
    version's `EvalLoopContainer` (nested_concat/nested_numpify) rather than assumed;
  - each new metric's exact value on hand-constructed synthetic data;
  - masked-bin exclusion from every statistic;
  - byte-identical behaviour with the group head disabled;
  - sufficient-stat accumulation over two batches equalling a single-pass computation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from regulonado.training.group_contrast import group_contrast_labels, group_replicate_weights
from regulonado.training.metrics import make_compute_metrics, make_preprocess_logits_for_metrics
from transformers import EvalPrediction
from transformers.trainer_pt_utils import EvalLoopContainer

# ---------------------------------------------------------------------------
# 0. Tuple-nesting mechanism, verified against the installed transformers version
# ---------------------------------------------------------------------------


class TestTupleNestingMechanism:
    """Pins the exact machinery `prediction_step`'s output tuple flows through.

    `RegulonadoTrainer.prediction_step` (not HF's own `evaluation_loop` call site) invokes
    `preprocess_logits_for_metrics` directly and returns its result as `logits`; that
    return value then flows through the base `Trainer.evaluation_loop`'s
    `accelerator.pad_across_processes` -> `gather_function` -> `EvalLoopContainer.add`
    (`nested_concat`) -> `EvalLoopContainer.get_arrays` (`nested_numpify`) pipeline
    regardless. This test exercises that exact pipeline (minus distributed gather, which is
    a no-op single-process) with a tuple of two differently-shaped tensors, standing in for
    (track_stats [B,T,31], group_stats [B,G,17]).
    """

    def test_tuple_of_two_tensors_round_trips_through_eval_loop_container(self) -> None:
        container = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        batch1 = (torch.randn(2, 5, 31), torch.randn(2, 7, 17))
        batch2 = (torch.randn(3, 5, 31), torch.randn(3, 7, 17))
        container.add(batch1)
        container.add(batch2)

        arrays = container.get_arrays()

        assert isinstance(arrays, tuple)
        assert len(arrays) == 2
        assert isinstance(arrays[0], np.ndarray)
        assert isinstance(arrays[1], np.ndarray)
        # Each leaf is concatenated independently along the batch dim; shapes stay
        # [N, T, 31] and [N, G, 17] respectively, never mixed.
        assert arrays[0].shape == (5, 5, 31)
        assert arrays[1].shape == (5, 7, 17)
        np.testing.assert_allclose(arrays[0][:2], batch1[0].numpy())
        np.testing.assert_allclose(arrays[1][2:], batch2[1].numpy())


# ---------------------------------------------------------------------------
# Synthetic scenario shared by the metric-value tests
# ---------------------------------------------------------------------------


def _synthetic_group_scenario(
    *, B: int = 2, L: int = 60, seed: int = 0, floor: float = -1.0
) -> dict[str, torch.Tensor]:
    """Three tracks, one per group (T == G == 3); group 0 is the target.

    Anchor-unit signals are random-but-deterministic and strictly positive so
    `group_contrast_labels` (smoothing_bins=1, i.e. no smoothing; pseudocount=0.1;
    gauge="median"; scale-only, no background subtraction) produces well-defined,
    non-degenerate per-group log2 labels. `floor=-1.0` keeps every bin unmasked
    (`group_linear > floor` is always true for strictly positive signal) unless a test
    overrides it.
    """
    rng = np.random.default_rng(seed)
    T = G = 3
    # Strictly positive, varied per group so every group has nonzero variance.
    signal = rng.uniform(0.2, 10.0, size=(B, T, L)).astype(np.float32)
    normalised_target = torch.tensor(signal)

    group_index = torch.tensor([0, 1, 2])
    group_weights = group_replicate_weights(group_index, n_groups=G)
    labels, mask = group_contrast_labels(
        normalised_target,
        group_weights=group_weights,
        background=None,
        smoothing_bins=1,
        pseudocount=0.1,
        gauge="median",
        floor=floor,
        clamp=(-6.0, 6.0),
    )
    return {
        "normalised_target": normalised_target,
        "labels": labels,
        "mask": mask,
        "group_weights": group_weights,
        "unit_k": torch.ones(T),
    }


def _run_group_preprocess(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_logits: torch.Tensor,
    group_weights: torch.Tensor,
    unit_k: torch.Tensor,
    target_index: int | None = 0,
    floor: float = -1.0,
) -> torch.Tensor:
    preprocess = make_preprocess_logits_for_metrics(
        topk_bins=8,
        group_contrast_group_weights=group_weights,
        group_contrast_unit_k=unit_k,
        group_contrast_unit_b=None,
        group_contrast_target_index=target_index,
        group_contrast_smoothing_bins=1,
        group_contrast_pseudocount=0.1,
        group_contrast_gauge="median",
        group_contrast_floor=floor,
        group_contrast_clamp=(-6.0, 6.0),
        group_contrast_quantile=0.9,
    )
    out = preprocess(logits, labels, group_logits=group_logits)
    assert isinstance(out, tuple) and len(out) == 2
    return out[1]


def _compute_group_metrics(group_stats: torch.Tensor, n_tracks: int = 3) -> dict[str, float]:
    compute_metrics = make_compute_metrics(n_tracks)
    track_stats = torch.zeros(group_stats.shape[0], n_tracks, 31)
    predictions = (track_stats.numpy(), group_stats.numpy())
    eval_pred = EvalPrediction(predictions=predictions, label_ids=None)
    return compute_metrics(eval_pred)


# ---------------------------------------------------------------------------
# 1. Preprocess shape and tuple/no-tuple gating
# ---------------------------------------------------------------------------


class TestPreprocessGating:
    def test_returns_plain_tensor_when_group_weights_absent(self) -> None:
        preprocess = make_preprocess_logits_for_metrics(topk_bins=4)
        logits = torch.rand(2, 3, 16)
        labels = torch.rand(2, 3, 16)
        out = preprocess(logits, labels, group_logits=torch.rand(2, 3, 16))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3, 31)

    def test_returns_tuple_when_group_weights_present(self) -> None:
        scenario = _synthetic_group_scenario()
        labels = scenario["normalised_target"]
        logits = torch.rand_like(labels)
        group_logits = scenario["labels"]
        group_stats = _run_group_preprocess(
            logits=logits,
            labels=labels,
            group_logits=group_logits,
            group_weights=scenario["group_weights"],
            unit_k=scenario["unit_k"],
        )
        B, G = labels.shape[0], 3
        assert group_stats.shape == (B, G, 17)


class TestGroupHeadDisabledIsByteIdentical:
    def test_preprocess_output_matches_build_without_any_group_kwargs(self) -> None:
        logits = torch.rand(3, 5, 20)
        labels = torch.rand(3, 5, 20)

        baseline = make_preprocess_logits_for_metrics(topk_bins=8)(logits, labels)
        disabled = make_preprocess_logits_for_metrics(
            topk_bins=8,
            group_contrast_group_weights=None,
        )(logits, labels, group_logits=torch.rand(3, 2, 20))

        torch.testing.assert_close(baseline, disabled, rtol=0.0, atol=0.0)

    def test_compute_metrics_dict_matches_build_without_group_sink(self) -> None:
        logits = torch.rand(3, 5, 20)
        labels = torch.rand(3, 5, 20)
        stats = make_preprocess_logits_for_metrics(topk_bins=8)(logits, labels)
        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)

        baseline = make_compute_metrics(n_tracks=5)(eval_pred)
        disabled = make_compute_metrics(n_tracks=5, per_group_sink=None)(eval_pred)

        assert baseline.keys() == disabled.keys()
        for key in baseline:
            if np.isnan(baseline[key]):
                assert np.isnan(disabled[key])
            else:
                assert baseline[key] == pytest.approx(disabled[key])
        assert not any(key.startswith("group_") for key in baseline)


# ---------------------------------------------------------------------------
# 2. Perfect prediction: Pearson 1.0, sd ratio 1.0, rank1 fraction matches construction
# ---------------------------------------------------------------------------


class TestPerfectPrediction:
    def test_perfect_prediction_metrics(self) -> None:
        scenario = _synthetic_group_scenario(seed=1)
        labels = scenario["normalised_target"]
        group_logits = scenario["labels"].clone()  # perfect prediction
        track_logits = torch.rand_like(labels)

        group_stats = _run_group_preprocess(
            logits=track_logits,
            labels=labels,
            group_logits=group_logits,
            group_weights=scenario["group_weights"],
            unit_k=scenario["unit_k"],
        )
        metrics = _compute_group_metrics(group_stats)

        assert metrics["group_contrast_pearson_median"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["group_score_pearson"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["group_score_sd_ratio"] == pytest.approx(1.0, abs=1e-5)

        # Independently compute the expected rank1 fraction from the constructed labels.
        observed = scenario["labels"]  # [B, G, L]
        mask = scenario["mask"][:, 0, :].bool()  # [B, L]
        argmax = observed.argmax(dim=1)  # [B, L]
        is_rank1 = (argmax == 0) & mask
        expected_fraction = float(is_rank1.sum()) / float(mask.sum())

        assert metrics["group_target_rank1_fraction"] == pytest.approx(expected_fraction, abs=1e-6)
        assert metrics["group_target_rank1_fraction_observed"] == pytest.approx(
            expected_fraction, abs=1e-6
        )
        # Perfect prediction: predicted and observed rank1 fractions must agree exactly.
        assert metrics["group_target_rank1_fraction"] == pytest.approx(
            metrics["group_target_rank1_fraction_observed"], abs=1e-12
        )

    def test_gauge_residual_near_zero_for_labels_as_predictions(self) -> None:
        """The label itself is gauge-centred (median over groups subtracted per bin), so
        using it as the "prediction" should show a near-zero gauge residual."""
        scenario = _synthetic_group_scenario(seed=2)
        labels = scenario["normalised_target"]
        group_logits = scenario["labels"].clone()
        track_logits = torch.rand_like(labels)

        group_stats = _run_group_preprocess(
            logits=track_logits,
            labels=labels,
            group_logits=group_logits,
            group_weights=scenario["group_weights"],
            unit_k=scenario["unit_k"],
        )
        metrics = _compute_group_metrics(group_stats)
        assert metrics["group_gauge_residual_median"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Masked bins excluded from every statistic
# ---------------------------------------------------------------------------


class TestMaskedBinsExcluded:
    def test_corrupting_predictions_only_at_masked_bins_does_not_change_metrics(self) -> None:
        B, L = 2, 60
        rng = np.random.default_rng(3)
        T = G = 3
        signal = rng.uniform(0.2, 10.0, size=(B, T, L)).astype(np.float32)
        # Force a background region [40, 60) where every group sits at a low, near-constant
        # value below the floor, so those bins get mask=0.
        signal[:, :, 40:60] = 0.02
        normalised_target = torch.tensor(signal)
        group_index = torch.tensor([0, 1, 2])
        group_weights = group_replicate_weights(group_index, n_groups=G)
        floor = 0.05
        labels, mask = group_contrast_labels(
            normalised_target,
            group_weights=group_weights,
            background=None,
            smoothing_bins=1,
            pseudocount=0.1,
            gauge="median",
            floor=floor,
            clamp=(-6.0, 6.0),
        )
        assert bool((mask[:, 0, 40:60] == 0).all()), "background region must be masked out"
        assert bool((mask[:, 0, :40] == 1).all()), "signal region must remain unmasked"

        clean_pred = labels.clone()  # perfect prediction everywhere
        corrupted_pred = clean_pred.clone()
        corrupted_pred[:, :, 40:60] = torch.randn(B, G, 20) * 1000.0  # garbage in masked region

        unit_k = torch.ones(T)
        stats_clean = _run_group_preprocess(
            logits=torch.rand_like(normalised_target),
            labels=normalised_target,
            group_logits=clean_pred,
            group_weights=group_weights,
            unit_k=unit_k,
            floor=floor,
        )
        stats_corrupted = _run_group_preprocess(
            logits=torch.rand_like(normalised_target),
            labels=normalised_target,
            group_logits=corrupted_pred,
            group_weights=group_weights,
            unit_k=unit_k,
            floor=floor,
        )

        metrics_clean = _compute_group_metrics(stats_clean)
        metrics_corrupted = _compute_group_metrics(stats_corrupted)

        for key in (
            "group_contrast_pearson_median",
            "group_score_pearson",
            "group_score_sd_ratio",
            "group_target_rank1_fraction",
            "group_target_rank1_fraction_observed",
        ):
            assert metrics_clean[key] == pytest.approx(metrics_corrupted[key], abs=1e-5), key


# ---------------------------------------------------------------------------
# 4. Sufficient-stat accumulation over two batches == single-pass over the concatenation
# ---------------------------------------------------------------------------


class TestAccumulationMatchesSinglePass:
    def test_two_batch_accumulation_equals_concatenated_single_pass(self) -> None:
        B1, B2, L = 2, 3, 40
        T = G = 3
        group_index = torch.tensor([0, 1, 2])
        group_weights = group_replicate_weights(group_index, n_groups=G)
        unit_k = torch.ones(T)

        def make_example_set(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            r = np.random.default_rng(seed)
            signal = r.uniform(0.2, 10.0, size=(n, T, L)).astype(np.float32)
            pred = r.normal(0.0, 1.0, size=(n, G, L)).astype(np.float32)
            return torch.tensor(signal), torch.tensor(pred)

        target1, pred1 = make_example_set(B1, seed=10)
        target2, pred2 = make_example_set(B2, seed=11)

        stats1 = _run_group_preprocess(
            logits=torch.rand_like(target1),
            labels=target1,
            group_logits=pred1,
            group_weights=group_weights,
            unit_k=unit_k,
        )
        stats2 = _run_group_preprocess(
            logits=torch.rand_like(target2),
            labels=target2,
            group_logits=pred2,
            group_weights=group_weights,
            unit_k=unit_k,
        )
        stats_two_batch = torch.cat([stats1, stats2], dim=0)

        target_cat = torch.cat([target1, target2], dim=0)
        pred_cat = torch.cat([pred1, pred2], dim=0)
        stats_single_pass = _run_group_preprocess(
            logits=torch.rand_like(target_cat),
            labels=target_cat,
            group_logits=pred_cat,
            group_weights=group_weights,
            unit_k=unit_k,
        )

        metrics_two_batch = _compute_group_metrics(stats_two_batch)
        metrics_single_pass = _compute_group_metrics(stats_single_pass)

        for key in metrics_two_batch:
            a, b = metrics_two_batch[key], metrics_single_pass[key]
            if np.isnan(a):
                assert np.isnan(b), key
            else:
                assert a == pytest.approx(b, abs=1e-5), key
