"""CPU tests for the preprocess_logits_for_metrics / compute_metrics pipeline.

These tests exercise the sufficient-stats accumulation and Pearson formulas
without requiring a GPU or a real model.
"""

from __future__ import annotations

import unittest.mock

import numpy as np
import pytest
import torch

# Import the metrics helpers directly so the test is self-contained.
from regulonado.training.runner import (
    RegulonadoTrainer,
    make_compute_metrics,
    make_preprocess_logits_for_metrics,
)
from scipy import stats as scipy_stats
from transformers import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 4,
    T: int = 8,
    L: int = 32,
    *,
    labels_transposed: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits [B,T,L], labels) where labels orientation is configurable."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    logits = torch.rand(B, T, L, generator=rng)
    labels_btl = torch.rand(B, T, L, generator=rng)
    labels = labels_btl.transpose(1, 2) if labels_transposed else labels_btl
    return logits, labels


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_shape_btl_labels(self):
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels = _make_batch(B=3, T=5, L=20, labels_transposed=False)
        out = preprocess(logits, labels)
        assert out.shape == (3, 5, 31), out.shape

    def test_output_shape_transposed_labels(self):
        """HF datasets loads labels as [B, L, T]; preprocess must handle it."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels_lt = _make_batch(B=3, T=5, L=20, labels_transposed=True)
        out = preprocess(logits, labels_lt)
        assert out.shape == (3, 5, 31), out.shape

    def test_transposed_labels_same_stats(self):
        """Stats must be identical regardless of whether labels are transposed."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels_btl = _make_batch(B=3, T=5, L=20, labels_transposed=False)
        labels_blt = labels_btl.transpose(1, 2)

        out_btl = preprocess(logits, labels_btl)
        out_blt = preprocess(logits, labels_blt)
        torch.testing.assert_close(out_btl, out_blt)

    def test_tuple_logits_unpacked(self):
        preprocess = make_preprocess_logits_for_metrics(topk_bins=4)
        logits, labels = _make_batch(B=2, T=3, L=16)
        out_plain = preprocess(logits, labels)
        out_tuple = preprocess((logits, torch.zeros(1)), labels)
        torch.testing.assert_close(out_plain, out_tuple)

    def test_topk_capped_at_L(self):
        """topk_bins > L should not raise."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=1000)
        logits, labels = _make_batch(B=2, T=3, L=16)
        out = preprocess(logits, labels)
        assert out.shape == (2, 3, 31)

    def test_n_column_values(self):
        """Col 5 (n for all bins) must equal L; col 11 (n for topk) must equal min(k, L)."""
        L, topk = 20, 8
        preprocess = make_preprocess_logits_for_metrics(topk_bins=topk)
        logits, labels = _make_batch(B=2, T=3, L=L)
        out = preprocess(logits, labels)
        assert out[..., 5].unique().item() == pytest.approx(L)
        assert out[..., 11].unique().item() == pytest.approx(topk)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _run(self, n_batches: int = 5, B: int = 4, T: int = 6, L: int = 24, topk: int = 8):
        """Accumulate stats over n_batches and return metrics dict."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=topk)
        compute_metrics = make_compute_metrics(n_tracks=T)

        all_stats = []
        for seed in range(n_batches):
            logits, labels = _make_batch(B=B, T=T, L=L, seed=seed)
            stats = preprocess(logits, labels)
            all_stats.append(stats.numpy())

        stacked = np.stack(all_stats, axis=0)  # [n_batches, B, T, 31] — mimic HF accumulation

        from transformers import EvalPrediction

        eval_pred = EvalPrediction(predictions=stacked.reshape(-1, T, 31), label_ids=None)
        return compute_metrics(eval_pred)

    def test_returns_expected_keys(self):
        metrics = self._run()
        assert "pearson_bin_median" in metrics
        assert "pearson_total_median" in metrics
        assert "abs_log_ratio_total_median" in metrics
        assert "calibration_shape_objective" in metrics
        assert any("pearson_top" in k for k in metrics)

    def test_pearson_in_range(self):
        metrics = self._run()
        assert -1.0 <= metrics["pearson_bin_median"] <= 1.0
        assert -1.0 <= metrics["pearson_total_median"] <= 1.0
        topk_key = next(k for k in metrics if "pearson_top" in k)
        assert -1.0 <= metrics[topk_key] <= 1.0

    def test_perfect_prediction(self):
        """When logits == labels, all Pearson values should be 1.0."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        compute_metrics = make_compute_metrics(n_tracks=4)

        rng = torch.Generator()
        rng.manual_seed(42)
        signal = torch.rand(3, 4, 20, generator=rng)
        stats = preprocess(signal, signal)

        from transformers import EvalPrediction

        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)
        assert m["pearson_bin_median"] == pytest.approx(1.0, abs=1e-4)
        assert m["abs_log_ratio_total_median"] == pytest.approx(0.0, abs=1e-6)
        assert m["calibration_shape_objective"] == pytest.approx(-0.1, abs=1e-4)

    def test_flat_prediction_uses_zero_pearson_in_balanced_objective(self):
        """A calibrated but flat profile must not receive a shape reward."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        compute_metrics = make_compute_metrics(n_tracks=2, calibration_shape_pearson_weight=0.25)
        signal = torch.ones(3, 2, 20)

        from transformers import EvalPrediction

        eval_pred = EvalPrediction(predictions=preprocess(signal, signal).numpy(), label_ids=None)
        m = compute_metrics(eval_pred)
        assert np.isnan(m["pearson_bin_median"])
        assert m["abs_log_ratio_total_median"] == pytest.approx(0.0, abs=1e-6)
        assert m["calibration_shape_objective"] == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def _specificity_metrics(compression: float) -> dict[str, float]:
        """Three cell types share one family; track 3 is alone in its family.

        Region-constant signals make region sums exact, and predictions keep each family
        member's centred log signal multiplied by ``compression``.
        """
        from regulonado.training.losses import contrast_family_weights
        from transformers import EvalPrediction

        B, T, R, region_bins = 6, 4, 16, 4
        weights = contrast_family_weights(["A", "A", "A", "B"], ["g1", "g2", "g3", "g4"])
        rng = torch.Generator()
        rng.manual_seed(7)
        log_t = torch.randn(B, T, R, generator=rng) + 6.0
        log_p = log_t.clone()
        family_mean = log_t[:, :3].mean(dim=1, keepdim=True)
        log_p[:, :3] = family_mean + compression * (log_t[:, :3] - family_mean)
        target = log_t.exp().repeat_interleave(region_bins, dim=-1)
        pred = log_p.exp().repeat_interleave(region_bins, dim=-1)

        preprocess = make_preprocess_logits_for_metrics(
            topk_bins=8,
            contrast_family_weights=weights,
            contrast_region_bins=region_bins,
            contrast_active_fraction=1.0,
        )
        compute_metrics = make_compute_metrics(n_tracks=T)
        stats = preprocess(pred, target).numpy()
        return compute_metrics(EvalPrediction(predictions=stats, label_ids=None))

    def test_specificity_perfect_prediction(self):
        m = self._specificity_metrics(compression=1.0)
        assert m["contrast_pearson_median"] == pytest.approx(1.0, abs=1e-4)
        assert m["contrast_slope_median"] == pytest.approx(1.0, abs=1e-3)
        assert m["contrast_sd_ratio_median"] == pytest.approx(1.0, abs=1e-3)

    def test_specificity_compressed_differences_keep_rank_but_shrink_spread(self):
        m = self._specificity_metrics(compression=0.5)
        assert m["contrast_pearson_median"] == pytest.approx(1.0, abs=1e-3)
        assert m["contrast_slope_median"] == pytest.approx(0.5, abs=0.01)
        assert m["contrast_sd_ratio_median"] == pytest.approx(0.5, abs=0.01)
        assert np.isfinite(m["contrast_objective"])

    def test_specificity_absent_without_families(self):
        metrics = self._run()
        assert np.isnan(metrics["contrast_pearson_median"])
        assert np.isnan(metrics["contrast_objective"])

    def test_against_scipy_pearson(self):
        """Sufficient-stats Pearson must match scipy on the same flat data."""
        B, T, L, topk = 6, 3, 30, 10
        preprocess = make_preprocess_logits_for_metrics(topk_bins=topk)
        compute_metrics = make_compute_metrics(n_tracks=T)

        rng = torch.Generator()
        rng.manual_seed(7)
        logits = torch.rand(B, T, L, generator=rng)
        labels = torch.rand(B, T, L, generator=rng)

        stats = preprocess(logits, labels)

        from transformers import EvalPrediction

        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)

        # Compute expected per-track Pearson using scipy (flatten B and L together).
        p_np = logits.numpy()  # [B, T, L]
        t_np = labels.numpy()

        scipy_rs = []
        for track in range(T):
            p_flat = p_np[:, track, :].ravel()
            t_flat = t_np[:, track, :].ravel()
            r, _ = scipy_stats.pearsonr(p_flat, t_flat)
            scipy_rs.append(r)

        expected_median = float(np.median(scipy_rs))
        assert m["pearson_bin_median"] == pytest.approx(expected_median, abs=1e-4)

    def test_constant_signal_track_is_nan(self):
        """A flat track produces undefined Pearson; compute_metrics must survive it."""
        preprocess = make_preprocess_logits_for_metrics(topk_bins=8)
        compute_metrics = make_compute_metrics(n_tracks=2)

        logits = torch.zeros(3, 2, 20)
        labels = torch.zeros(3, 2, 20)
        stats = preprocess(logits, labels)

        from transformers import EvalPrediction

        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)
        # With all-zero inputs, nan is expected; just check no exception raised
        assert "pearson_bin_median" in m


# ---------------------------------------------------------------------------
# loss / metric agreement — both consume regulonado.training.losses.specificity_stats
# ---------------------------------------------------------------------------


class TestContrastLossMetricAgreement:
    @staticmethod
    def _generate(compression: float, region_bins: int, seed: int = 7):
        from regulonado.training.losses import contrast_family_weights

        B, T, R = 6, 4, 16
        weights = contrast_family_weights(["A", "A", "A", "B"], ["g1", "g2", "g3", "g4"])
        rng = torch.Generator()
        rng.manual_seed(seed)
        log_t = torch.randn(B, T, R, generator=rng) + 6.0
        log_p = log_t.clone()
        family_mean = log_t[:, :3].mean(dim=1, keepdim=True)
        log_p[:, :3] = family_mean + compression * (log_t[:, :3] - family_mean)
        target = log_t.exp().repeat_interleave(region_bins, dim=-1)
        pred = log_p.exp().repeat_interleave(region_bins, dim=-1)
        return pred, target, weights

    @pytest.mark.parametrize("compression", [1.0, 0.5, 0.0])
    @pytest.mark.parametrize("region_bins", [4, 8])
    @pytest.mark.parametrize("active_fraction", [1.0, 0.1])
    def test_loss_and_metric_agree_on_the_same_batch(
        self, compression: float, region_bins: int, active_fraction: float
    ) -> None:
        from regulonado.training.losses import (
            _pearson_from_stats_torch,
            specificity_stats,
            track_contrast_correlation_loss,
        )
        from transformers import EvalPrediction

        pred, target, weights = self._generate(compression, region_bins)
        kwargs = dict(region_bins=region_bins, pseudocount=0.1, active_fraction=active_fraction)

        stats = specificity_stats(pred, target, weights, **kwargs).sum(dim=0)  # [T, 6]
        r, valid = _pearson_from_stats_torch(stats)
        loss = track_contrast_correlation_loss(pred, target, weights, **kwargs)

        preprocess = make_preprocess_logits_for_metrics(
            topk_bins=8,
            contrast_family_weights=weights,
            contrast_region_bins=region_bins,
            contrast_pseudocount=0.1,
            contrast_active_fraction=active_fraction,
        )
        compute_metrics = make_compute_metrics(n_tracks=pred.shape[1])
        eval_stats = preprocess(pred, target).numpy()
        m = compute_metrics(EvalPrediction(predictions=eval_stats, label_ids=None))

        if not bool(valid.any()):
            # compression=0.0 collapses every family member's prediction to the family
            # mean, so predicted specificity has zero variance everywhere: both the loss
            # and the metric must agree there is no information to correlate.
            assert loss.item() == pytest.approx(0.0)
            assert np.isnan(m["contrast_pearson_median"])
            return

        expected_loss = 1.0 - r[valid].mean()
        assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)

        r_valid = r[valid].detach().numpy()
        expected_median = float(np.median(r_valid))
        assert m["contrast_pearson_median"] == pytest.approx(expected_median, abs=1e-5)


# ---------------------------------------------------------------------------
# RegulonadoTrainer.prediction_step integration
# ---------------------------------------------------------------------------


class TestRegulonadoTrainerPredictionStep:
    """Verify that prediction_step calls preprocess with raw labels and then
    reduces labels to [B, T] so that full [B, T, L] arrays are never accumulated."""

    def _make_trainer(self, preprocess_fn):
        """Construct a minimal RegulonadoTrainer without a real model or dataset."""
        from transformers import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/regulonado_test", use_cpu=True)
        # Pass a dummy model; we won't call train(), only inspect prediction_step behaviour.
        dummy_model = torch.nn.Linear(1, 1)
        trainer = RegulonadoTrainer(
            model=dummy_model,
            args=args,
            preprocess_logits_for_metrics=preprocess_fn,
        )
        return trainer

    def test_preprocess_fn_intercepted(self):
        """preprocess_logits_for_metrics must be stored on _metrics_preprocess, not base attr."""
        sentinel = object()
        trainer = self._make_trainer(preprocess_fn=sentinel)
        assert trainer._metrics_preprocess is sentinel
        # Base class must NOT hold it (so evaluation_loop won't double-call it).
        assert trainer.preprocess_logits_for_metrics is None

    def test_prediction_step_reduces_labels(self):
        """After prediction_step, labels must be [B, T] not [B, T, L]."""
        B, T, L = 3, 5, 16
        captured = {}

        def spy_preprocess(logits, labels):
            captured["label_shape"] = tuple(labels.shape)
            return logits  # pass-through for shape test

        trainer = self._make_trainer(preprocess_fn=spy_preprocess)

        # Patch super().prediction_step to return known shapes.
        raw_logits = torch.rand(B, T, L)
        raw_labels = torch.rand(B, L, T)  # transposed as HF datasets returns

        with unittest.mock.patch.object(
            Trainer, "prediction_step", return_value=(None, raw_logits, raw_labels)
        ):
            _, out_logits, out_labels = trainer.prediction_step(
                model=trainer.model, inputs={}, prediction_loss_only=False
            )

        # preprocess saw the raw [B, L, T] labels
        assert captured["label_shape"] == (B, L, T)
        # returned labels are reduced to 2D [B, T] (sum over the L dim)
        assert out_labels.shape == (B, T), out_labels.shape

    def test_prediction_step_btl_labels_also_reduced(self):
        """Labels already in [B, T, L] orientation must also be reduced to [B, T]."""
        B, T, L = 3, 5, 16

        trainer = self._make_trainer(preprocess_fn=lambda logits, labels: logits)

        raw_logits = torch.rand(B, T, L)
        raw_labels = torch.rand(B, T, L)

        with unittest.mock.patch.object(
            Trainer, "prediction_step", return_value=(None, raw_logits, raw_labels)
        ):
            _, _, out_labels = trainer.prediction_step(
                model=trainer.model, inputs={}, prediction_loss_only=False
            )

        assert out_labels.shape == (B, T), out_labels.shape


def test_preprocess_logits_accumulates_in_float32_for_half_precision_outputs() -> None:
    preprocess = make_preprocess_logits_for_metrics(topk_bins=4)
    logits = (torch.rand(2, 3, 16) * 50).to(torch.bfloat16)

    stats = preprocess(logits, torch.rand(2, 3, 16))

    assert stats.dtype == torch.float32
