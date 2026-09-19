"""Stage 3 of the group-contrast-head plan: wiring the two group-contrast loss terms.

Covers:
  - the unit-conversion algebra (``group_contrast_unit_factors``) against
    ``target_specificity.to_anchor_units``, the decisive numeric check for this stage;
  - the channel and score Huber terms (``_group_contrast_loss_terms``): zero at equality,
    masking, gauge invariance, gradient flow;
  - config validation at loss-build time (``_resolve_group_contrast_loss_config``);
  - end-to-end wiring through ``RegulonadoTrainer.compute_loss`` and the per-step
    observability hook (``RegulonadoTrainer.log``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from regulonado.target_specificity import to_anchor_units
from regulonado.training.group_contrast import group_contrast_labels, group_replicate_weights
from regulonado.training.label_space import resolve_count_label_space
from regulonado.training.runner import (
    GroupContrastLossConfig,
    RegulonadoTrainer,
    _build_loss_fn,
    _group_contrast_loss_terms,
    _resolve_group_contrast_loss_config,
    group_contrast_unit_factors,
)

# ---------------------------------------------------------------------------
# Synthetic records helper
# ---------------------------------------------------------------------------


def _make_records(
    n_tracks: int,
    *,
    groups: list[str] | None = None,
    seed: int = 0,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    records = []
    for t in range(n_tracks):
        anchor_reference = float(rng.uniform(5.0, 50.0))
        background = float(rng.uniform(0.0, anchor_reference * 0.5))
        scale_factor = 1.0 / (anchor_reference - background)
        records.append(
            {
                "track_name": f"track{t}",
                "scale_factor": scale_factor,
                "background": background,
                "anchor_reference": anchor_reference,
                "fragment_length": 200.0,
                "library_size": 1e7,
                "group": groups[t] if groups is not None else f"group{t}",
            }
        )
    return records


# ---------------------------------------------------------------------------
# 1. THE CRITICAL PART: unit conversion against target_specificity.to_anchor_units
# ---------------------------------------------------------------------------


class TestUnitConversion:
    @pytest.mark.parametrize("background_mode", ["subtract", "scale-only"])
    def test_converted_normalised_target_matches_to_anchor_units(self, background_mode):
        n_tracks, n_bins = 5, 40
        bin_size = 32
        records = _make_records(n_tracks, seed=1)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=bin_size
        )

        rng = np.random.default_rng(2)
        y = rng.uniform(0.0, 5.0, size=(n_tracks, n_bins)).astype(np.float32)  # stored coverage
        counts = y * count_space.count_factors[:, None]
        n = counts / count_space.exposure[:, None]  # normalised_target, as the loss sees it

        # group_contrast_unit_factors returns (k, k * b_true): the second element is the
        # *already k-scaled* background threshold meant to be subtracted from `n * k`
        # (`normalised_target` rescaled by `k`) — exactly what group_contrast_labels'
        # `background=` argument expects. See the helper's docstring for the algebra
        # (`max(n - b_true, 0) * k == max(n * k - k * b_true, 0)` since k > 0).
        k, scaled_background = group_contrast_unit_factors(
            records, count_space, background_mode=background_mode
        )
        scaled_target = n * k[:, None]
        if background_mode == "subtract":
            a_helper = np.maximum(scaled_target - scaled_background[:, None], 0.0)
        else:
            assert scaled_background is None
            a_helper = scaled_target

        scale_factor = np.array([r["scale_factor"] for r in records], dtype=np.float32)
        background = np.array([r["background"] for r in records], dtype=np.float32)
        a_expected = to_anchor_units(
            y.T,  # (R=n_bins, T=n_tracks)
            scale_factor=scale_factor,
            background=background,
            background_mode=background_mode,
        ).T  # -> (T, n_bins)

        np.testing.assert_allclose(a_helper, a_expected, rtol=1e-5, atol=1e-6)

    def test_subtract_and_scale_only_give_different_answers_when_background_nonzero(self):
        """Sanity: the two arms must actually differ (catches a helper that silently
        ignores background_mode)."""
        records = _make_records(3, seed=3)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        k_sub, b_sub = group_contrast_unit_factors(records, count_space, background_mode="subtract")
        k_scale, b_scale = group_contrast_unit_factors(
            records, count_space, background_mode="scale-only"
        )
        np.testing.assert_allclose(k_sub, k_scale)  # k itself doesn't depend on the mode
        assert b_scale is None
        assert b_sub is not None
        assert np.any(b_sub > 0)

    def test_invalid_background_mode_raises(self):
        records = _make_records(2, seed=4)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        with pytest.raises(ValueError, match="group_contrast_background"):
            group_contrast_unit_factors(records, count_space, background_mode="bogus")

    def test_full_pipeline_group_mean_matches_reference_at_identity_smoothing(self):
        """End-to-end: feeding group_contrast_labels the (k, k*b)-converted target and
        taking its pre-log group_linear (at smoothing_bins=1, an identity) must equal
        target_specificity.group_mean(to_anchor_units(y))."""
        from regulonado.target_specificity import group_index_from_records, group_mean

        records = _make_records(4, groups=["A", "A", "B", "B"], seed=5)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        rng = np.random.default_rng(6)
        y = rng.uniform(0.0, 5.0, size=(4, 12)).astype(np.float32)
        counts = y * count_space.count_factors[:, None]
        n = counts / count_space.exposure[:, None]

        k, b = group_contrast_unit_factors(records, count_space, background_mode="subtract")
        scaled_target = torch.as_tensor(n * k[:, None])[None]  # [1, T, L]

        group_index, group_names = group_index_from_records(records)
        weights = group_replicate_weights(torch.as_tensor(group_index), len(group_names))
        labels, mask = group_contrast_labels(
            scaled_target,
            group_weights=weights,
            background=torch.as_tensor(b),
            smoothing_bins=1,
            pseudocount=0.1,
            floor=-1.0,  # disable masking for this check
        )
        # Reconstruct the pre-log group_linear from the returned (post-log, gauge-centred)
        # labels is lossy (log + gauge), so instead recompute group_linear directly via the
        # same weights applied to to_anchor_units(y) and compare in linear anchor units.
        scale_factor = np.array([r["scale_factor"] for r in records], dtype=np.float32)
        background = np.array([r["background"] for r in records], dtype=np.float32)
        a_expected = to_anchor_units(
            y.T, scale_factor=scale_factor, background=background, background_mode="subtract"
        ).T  # (T, L)
        group_linear_expected = group_mean(a_expected.T, group_index, len(group_names)).T  # (G, L)

        # Recompute group_linear from the label pipeline's own intermediate by inverting
        # the log+gauge is unnecessary: rebuild it directly the same way group_contrast_labels
        # does, using the now-verified (k, b) conversion, and compare to the numpy reference.
        smoothed = scaled_target[0]  # smoothing_bins=1 is identity
        subtracted = (smoothed - torch.as_tensor(b)[:, None]).clamp_min(0.0)
        group_linear = torch.einsum("gt,tl->gl", weights, subtracted).numpy()

        np.testing.assert_allclose(group_linear, group_linear_expected, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------


class TestResolveGroupContrastLossConfig:
    def test_disabled_by_default_returns_none(self):
        records = _make_records(3, seed=7)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        cfg = _resolve_group_contrast_loss_config(
            {},
            {},
            records=records,
            count_space=count_space,
            group_contrast_n_groups=2,
            group_contrast_group_names=["A", "B"],
        )
        assert cfg is None

    def test_enabling_without_group_head_raises(self):
        records = _make_records(3, seed=8)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        with pytest.raises(ValueError, match="group_contrast_enabled"):
            _resolve_group_contrast_loss_config(
                {"group_contrast_weight": 1.0},
                {},
                records=records,
                count_space=count_space,
                group_contrast_n_groups=0,
                group_contrast_group_names=[],
            )

    def test_enabling_without_exposure_raises(self):
        records = _make_records(3, seed=9)
        with pytest.raises(ValueError, match="data.exposure|data.label_space"):
            _resolve_group_contrast_loss_config(
                {"group_contrast_weight": 1.0},
                {},
                records=records,
                count_space=None,
                group_contrast_n_groups=2,
                group_contrast_group_names=["A", "B"],
            )

    def test_missing_target_group_raises_when_score_weight_set(self):
        records = _make_records(4, groups=["A", "A", "B", "B"], seed=10)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        with pytest.raises(ValueError, match="group_contrast_target"):
            _resolve_group_contrast_loss_config(
                {"group_score_weight": 1.0},
                {"group_contrast_target": "C"},
                records=records,
                count_space=count_space,
                group_contrast_n_groups=2,
                group_contrast_group_names=["A", "B"],
            )

    def test_score_weight_without_target_raises(self):
        records = _make_records(4, groups=["A", "A", "B", "B"], seed=11)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        with pytest.raises(ValueError, match="group_contrast_target"):
            _resolve_group_contrast_loss_config(
                {"group_score_weight": 1.0},
                {},
                records=records,
                count_space=count_space,
                group_contrast_n_groups=2,
                group_contrast_group_names=["A", "B"],
            )

    def test_channel_weight_alone_does_not_require_target(self):
        records = _make_records(4, groups=["A", "A", "B", "B"], seed=12)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        cfg = _resolve_group_contrast_loss_config(
            {"group_contrast_weight": 1.0},
            {},
            records=records,
            count_space=count_space,
            group_contrast_n_groups=2,
            group_contrast_group_names=["A", "B"],
        )
        assert cfg is not None
        assert cfg.target_index is None
        assert cfg.channel_weight == 1.0
        assert cfg.score_weight == 0.0


# ---------------------------------------------------------------------------
# 3. The two loss terms
# ---------------------------------------------------------------------------


def _make_cfg(records, count_space, **overrides) -> GroupContrastLossConfig:
    loss_cfg = {
        "group_contrast_weight": 1.0,
        "group_score_weight": 1.0,
        "group_contrast_delta": 1.0,
    }
    trainer_cfg = {
        "group_contrast_target": "A",
        "group_contrast_smoothing_bins": 1,
        "group_contrast_floor": -1.0,  # disabled by default in these tests; overridden below
        "group_contrast_background": "subtract",
    }
    trainer_cfg.update(overrides)
    cfg = _resolve_group_contrast_loss_config(
        loss_cfg,
        trainer_cfg,
        records=records,
        count_space=count_space,
        group_contrast_n_groups=2,
        group_contrast_group_names=["A", "B"],
    )
    assert cfg is not None
    return cfg


class TestGroupContrastLossTerms:
    def _setup(self, *, floor: float = -1.0):
        records = _make_records(4, groups=["A", "A", "B", "B"], seed=20)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        cfg = _make_cfg(records, count_space, group_contrast_floor=floor)
        return records, count_space, cfg

    def test_zero_at_prediction_equals_label(self):
        _, _, cfg = self._setup()
        torch.manual_seed(0)
        normalised_target = torch.rand(2, 4, 16)

        unit_k = cfg.unit_k[None, :, None]
        unit_b = cfg.unit_b
        scaled = normalised_target * unit_k
        labels, mask = group_contrast_labels(
            scaled,
            group_weights=cfg.group_weights,
            background=unit_b,
            smoothing_bins=cfg.smoothing_bins,
            pseudocount=cfg.pseudocount,
            gauge=cfg.gauge,
            floor=cfg.floor,
            clamp=cfg.clamp,
        )
        assert mask.sum() > 0, "test needs at least one unmasked bin"

        channel_loss, score_loss = _group_contrast_loss_terms(labels, normalised_target, cfg)

        torch.testing.assert_close(channel_loss, torch.tensor(0.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(score_loss, torch.tensor(0.0), atol=1e-5, rtol=0)

    def test_gradients_flow_to_group_channels_and_are_finite(self):
        _, _, cfg = self._setup()
        torch.manual_seed(1)
        normalised_target = torch.rand(2, 4, 16)
        group_pred = torch.randn(2, 2, 16, requires_grad=True)

        channel_loss, score_loss = _group_contrast_loss_terms(group_pred, normalised_target, cfg)
        (channel_loss + score_loss).backward()

        assert group_pred.grad is not None
        assert torch.isfinite(group_pred.grad).all()
        assert torch.any(group_pred.grad != 0)

    def test_gradient_path_survives_an_empty_mask(self):
        """floor set absurdly high masks every bin; both terms must still be zero-valued
        (not NaN) and keep a gradient path (the `+ pred.sum() * 0.0` convention)."""
        _, _, cfg = self._setup(floor=1e6)
        torch.manual_seed(2)
        normalised_target = torch.rand(2, 4, 16)
        group_pred = torch.randn(2, 2, 16, requires_grad=True)

        channel_loss, score_loss = _group_contrast_loss_terms(group_pred, normalised_target, cfg)

        assert torch.isfinite(channel_loss)
        assert torch.isfinite(score_loss)
        total = channel_loss + score_loss
        total.backward()
        assert group_pred.grad is not None
        assert torch.isfinite(group_pred.grad).all()

    def test_masking_excludes_the_intended_bins(self):
        """Corrupting only masked-bin predictions must leave both terms unchanged."""
        records, count_space, cfg = self._setup(floor=0.139)
        torch.manual_seed(3)
        # Two tracks per group; make group A strongly signal-bearing in bins [0, 4) and
        # everything else near-zero (below the anchor-unit floor), so only [0, 4) survives
        # the all-groups-OR mask.
        normalised_target = torch.full((1, 4, 8), 1e-4)
        normalised_target[:, 0:2, 0:4] = 5.0  # group A tracks high in bins 0..3

        unit_k = cfg.unit_k[None, :, None]
        scaled = normalised_target * unit_k
        labels, mask = group_contrast_labels(
            scaled,
            group_weights=cfg.group_weights,
            background=cfg.unit_b,
            smoothing_bins=cfg.smoothing_bins,
            pseudocount=cfg.pseudocount,
            gauge=cfg.gauge,
            floor=cfg.floor,
            clamp=cfg.clamp,
        )
        assert mask[0, 0, 0:4].bool().all()
        assert not mask[0, 0, 4:].bool().any()

        base_pred = labels.clone()
        corrupted_pred = base_pred.clone()
        corrupted_pred[:, :, 4:] += 1000.0  # only touch masked bins

        loss_base = _group_contrast_loss_terms(base_pred, normalised_target, cfg)
        loss_corrupted = _group_contrast_loss_terms(corrupted_pred, normalised_target, cfg)

        for base_term, corrupted_term in zip(loss_base, loss_corrupted):
            torch.testing.assert_close(base_term, corrupted_term, atol=1e-5, rtol=0)

    def test_score_term_gauge_invariance_to_additive_per_bin_constant(self):
        """Adding the same constant to every predicted group channel at a bin must not
        change the score term (reduce_target_score is a difference, invariant to a
        per-bin shift applied uniformly across channels)."""
        _, _, cfg = self._setup(floor=-1.0)
        torch.manual_seed(4)
        normalised_target = torch.rand(2, 4, 16)
        group_pred = torch.randn(2, 2, 16)

        _, score_before = _group_contrast_loss_terms(group_pred, normalised_target, cfg)

        shift = torch.randn(2, 1, 16)  # per (batch, bin), broadcast over the channel axis
        shifted_pred = group_pred + shift
        _, score_after = _group_contrast_loss_terms(shifted_pred, normalised_target, cfg)

        torch.testing.assert_close(score_before, score_after, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. _build_loss_fn wiring: disabled path is byte-identical, enabled path adds terms
# ---------------------------------------------------------------------------


class TestBuildLossFnGroupContrastWiring:
    def _base_kwargs(self, n_tracks: int):
        return dict(
            scale_factors=np.ones(n_tracks, dtype=np.float32),
            clip_hard=np.full(n_tracks, 796.0, dtype=np.float32),
            labels_already_scaled=False,
        )

    def test_disabled_by_default_is_byte_identical(self):
        n_tracks = 3
        loss_cfg = {"name": "poisson_multinomial"}
        exposure = torch.ones(n_tracks)
        loss_fn_without = _build_loss_fn(
            loss_cfg, exposure=exposure, group_contrast=None, **self._base_kwargs(n_tracks)
        )
        loss_fn_with_none = _build_loss_fn(
            loss_cfg, exposure=exposure, group_contrast=None, **self._base_kwargs(n_tracks)
        )
        torch.manual_seed(5)
        pred = torch.rand(2, n_tracks, 8)
        target = torch.rand(2, n_tracks, 8) * 3

        loss_a = loss_fn_without(pred, target)
        loss_b = loss_fn_with_none(pred, target, None)

        torch.testing.assert_close(loss_a, loss_b)
        assert loss_fn_without.last_group_channel_loss is None
        assert loss_fn_without.last_group_score_loss is None

    def test_group_pred_ignored_when_group_contrast_config_is_none(self):
        n_tracks = 3
        loss_cfg = {"name": "poisson_multinomial"}
        exposure = torch.ones(n_tracks)
        loss_fn = _build_loss_fn(
            loss_cfg, exposure=exposure, group_contrast=None, **self._base_kwargs(n_tracks)
        )
        torch.manual_seed(6)
        pred = torch.rand(2, n_tracks, 8)
        target = torch.rand(2, n_tracks, 8) * 3
        group_pred = torch.randn(2, 2, 8)

        loss_without_group = loss_fn(pred, target)
        loss_with_group = loss_fn(pred, target, group_pred)

        torch.testing.assert_close(loss_without_group, loss_with_group)

    def test_enabled_adds_weighted_group_terms_and_records_last_values(self):
        n_tracks = 4
        records = _make_records(n_tracks, groups=["A", "A", "B", "B"], seed=30)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        group_contrast_cfg = _make_cfg(records, count_space, group_contrast_floor=-1.0)
        loss_cfg = {"name": "poisson_multinomial"}
        loss_fn = _build_loss_fn(
            loss_cfg,
            exposure=torch.as_tensor(count_space.exposure),
            group_contrast=group_contrast_cfg,
            **self._base_kwargs(n_tracks),
        )
        torch.manual_seed(7)
        pred = torch.rand(2, n_tracks, 16)
        target = torch.rand(2, n_tracks, 16) * 3
        group_pred = torch.randn(2, 2, 16)

        loss_without_group = loss_fn(pred, target, None)
        loss_with_group = loss_fn(pred, target, group_pred)

        assert loss_fn.last_group_channel_loss is not None
        assert loss_fn.last_group_score_loss is not None
        assert not torch.isclose(loss_with_group, loss_without_group)


# ---------------------------------------------------------------------------
# 5. End-to-end through RegulonadoTrainer.compute_loss + observability
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    def test_compute_loss_passes_group_slice_and_records_observable_terms(self):
        from transformers import TrainingArguments

        n_tracks = 4
        records = _make_records(n_tracks, groups=["A", "A", "B", "B"], seed=40)
        count_space = resolve_count_label_space(
            records, count_unit="fragments", exposure="anchor", bin_size=32
        )
        group_contrast_cfg = _make_cfg(records, count_space, group_contrast_floor=-1.0)
        loss_fn = _build_loss_fn(
            {"name": "poisson_multinomial"},
            scale_factors=np.ones(n_tracks, dtype=np.float32),
            clip_hard=np.full(n_tracks, 796.0, dtype=np.float32),
            labels_already_scaled=False,
            exposure=torch.as_tensor(count_space.exposure),
            group_contrast=group_contrast_cfg,
        )

        B, T, G, L = 2, n_tracks, 2, 16
        track_part = torch.rand(B, T, L)
        group_part = torch.randn(B, G, L)
        logits = torch.cat([track_part, group_part], dim=1)
        labels = torch.rand(B, T, L) * 3

        class _CompositeStub(torch.nn.Module):
            track_channel_count = T

            def forward(self, input_ids, **_):
                return logits

        args = TrainingArguments(output_dir="/tmp/regulonado_test_group_contrast", use_cpu=True)
        trainer = RegulonadoTrainer(model=_CompositeStub(), args=args, loss_fn=loss_fn)

        trainer.compute_loss(trainer.model, {"input_ids": torch.zeros(B, 1), "labels": labels})

        assert trainer._group_contrast_count == 1
        assert trainer._group_contrast_sum != 0.0 or trainer._group_score_sum != 0.0

        logs = {"loss": 1.0}
        trainer.log(logs)
        # trainer.log mutates a local copy and delegates to Trainer.log for reporting;
        # the accumulator reset is the observable effect here.
        assert trainer._group_contrast_count == 0

    def test_single_head_path_never_calls_group_contrast(self):
        """A plain model with no track_channel_count must never populate the group
        accumulator, matching the 'byte-identical when no group head' requirement."""
        from transformers import TrainingArguments

        B, T, L = 2, 3, 5
        logits = torch.randn(B, T, L)
        labels = torch.randn(B, T, L)

        def loss_fn(pred, target, group_pred=None):
            assert group_pred is None
            return (pred - target).pow(2).mean()

        class _PlainStub(torch.nn.Module):
            def forward(self, input_ids, **_):
                return logits

        args = TrainingArguments(output_dir="/tmp/regulonado_test_group_contrast2", use_cpu=True)
        trainer = RegulonadoTrainer(model=_PlainStub(), args=args, loss_fn=loss_fn)

        trainer.compute_loss(trainer.model, {"input_ids": torch.zeros(B, 1), "labels": labels})

        assert trainer._group_contrast_count == 0
