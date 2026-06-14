"""CPU tests for the preprocess_logits_for_metrics / compute_metrics pipeline.

These tests exercise the sufficient-stats accumulation and Pearson formulas
without requiring a GPU or a real model.
"""
from __future__ import annotations

import unittest.mock

import numpy as np
import pytest
import torch
from scipy import stats as scipy_stats
from transformers import Trainer

# Import the private helpers directly so the test is self-contained.
from regulonado.train import (
    RegulonadoTrainer,
    _make_compute_metrics,
    _make_preprocess_logits_for_metrics,
)


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
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels = _make_batch(B=3, T=5, L=20, labels_transposed=False)
        out = preprocess(logits, labels)
        assert out.shape == (3, 5, 12), out.shape

    def test_output_shape_transposed_labels(self):
        """HF datasets loads labels as [B, L, T]; preprocess must handle it."""
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels_lt = _make_batch(B=3, T=5, L=20, labels_transposed=True)
        out = preprocess(logits, labels_lt)
        assert out.shape == (3, 5, 12), out.shape

    def test_transposed_labels_same_stats(self):
        """Stats must be identical regardless of whether labels are transposed."""
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=8)
        logits, labels_btl = _make_batch(B=3, T=5, L=20, labels_transposed=False)
        labels_blt = labels_btl.transpose(1, 2)

        out_btl = preprocess(logits, labels_btl)
        out_blt = preprocess(logits, labels_blt)
        torch.testing.assert_close(out_btl, out_blt)

    def test_tuple_logits_unpacked(self):
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=4)
        logits, labels = _make_batch(B=2, T=3, L=16)
        out_plain = preprocess(logits, labels)
        out_tuple = preprocess((logits, torch.zeros(1)), labels)
        torch.testing.assert_close(out_plain, out_tuple)

    def test_topk_capped_at_L(self):
        """topk_bins > L should not raise."""
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=1000)
        logits, labels = _make_batch(B=2, T=3, L=16)
        out = preprocess(logits, labels)
        assert out.shape == (2, 3, 12)

    def test_n_column_values(self):
        """Col 5 (n for all bins) must equal L; col 11 (n for topk) must equal min(k, L)."""
        L, topk = 20, 8
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=topk)
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
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=topk)
        compute_metrics = _make_compute_metrics(n_tracks=T)

        all_stats = []
        for seed in range(n_batches):
            logits, labels = _make_batch(B=B, T=T, L=L, seed=seed)
            stats = preprocess(logits, labels)
            all_stats.append(stats.numpy())

        stacked = np.stack(all_stats, axis=0)  # [n_batches, B, T, 12] — mimic HF accumulation

        from transformers import EvalPrediction
        eval_pred = EvalPrediction(predictions=stacked.reshape(-1, T, 12), label_ids=None)
        return compute_metrics(eval_pred)

    def test_returns_expected_keys(self):
        metrics = self._run()
        assert "pearson_bin_median" in metrics
        assert any("pearson_top" in k for k in metrics)

    def test_pearson_in_range(self):
        metrics = self._run()
        assert -1.0 <= metrics["pearson_bin_median"] <= 1.0
        topk_key = next(k for k in metrics if "pearson_top" in k)
        assert -1.0 <= metrics[topk_key] <= 1.0

    def test_perfect_prediction(self):
        """When logits == labels, all Pearson values should be 1.0."""
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=8)
        compute_metrics = _make_compute_metrics(n_tracks=4)

        rng = torch.Generator()
        rng.manual_seed(42)
        signal = torch.rand(3, 4, 20, generator=rng)
        stats = preprocess(signal, signal)

        from transformers import EvalPrediction
        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)
        assert m["pearson_bin_median"] == pytest.approx(1.0, abs=1e-4)

    def test_against_scipy_pearson(self):
        """Sufficient-stats Pearson must match scipy on the same flat data."""
        B, T, L, topk = 6, 3, 30, 10
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=topk)
        compute_metrics = _make_compute_metrics(n_tracks=T)

        rng = torch.Generator()
        rng.manual_seed(7)
        logits = torch.rand(B, T, L, generator=rng)
        labels = torch.rand(B, T, L, generator=rng)

        stats = preprocess(logits, labels)

        from transformers import EvalPrediction
        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)

        # Compute expected per-track Pearson using scipy (flatten B and L together).
        p_np = logits.numpy()   # [B, T, L]
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
        preprocess = _make_preprocess_logits_for_metrics(topk_bins=8)
        compute_metrics = _make_compute_metrics(n_tracks=2)

        logits = torch.zeros(3, 2, 20)
        labels = torch.zeros(3, 2, 20)
        stats = preprocess(logits, labels)

        from transformers import EvalPrediction
        eval_pred = EvalPrediction(predictions=stats.numpy(), label_ids=None)
        m = compute_metrics(eval_pred)
        # With all-zero inputs, nan is expected; just check no exception raised
        assert "pearson_bin_median" in m


# ---------------------------------------------------------------------------
# RegulonadoTrainer.prediction_step integration
# ---------------------------------------------------------------------------

class TestRegulonadoTrainerPredictionStep:
    """Verify that prediction_step calls preprocess with raw labels and then
    reduces labels to [B, T] so that full [B, T, L] arrays are never accumulated."""

    def _make_trainer(self, preprocess_fn):
        """Construct a minimal RegulonadoTrainer without a real model or dataset."""
        from transformers import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/regulonado_test", no_cuda=True)
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
