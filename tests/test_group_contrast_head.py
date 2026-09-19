"""Tests for the group-contrast head, its composite-output plumbing, and config threading.

Covers Stage 2 of the group-contrast-head plan: the ``GroupContrastHead`` subclass, the
``CompositeTrackGroupHead`` that concatenates it with the existing per-track head, and the
``RegulonadoConfig`` / ``RegulonadoModel`` wiring that keeps the single-head path unchanged
when the group head is not configured.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from regulonado.model import (
    CompositeTrackGroupHead,
    GroupContrastHead,
    RegulonadoConfig,
    RegulonadoModel,
)
from regulonado.model.heads import build_transfer_learning_head


class TestGroupContrastHead:
    def test_output_shape_is_batch_by_n_groups_by_length(self) -> None:
        head = GroupContrastHead(in_ch=8, hidden=8, n_groups=5, activation_type="identity")
        x = torch.randn(2, 8, 16)

        out = head(x)

        assert out.shape == (2, 5, 16)

    def test_non_identity_activation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="identity"):
            GroupContrastHead(in_ch=8, hidden=8, n_groups=5, activation_type="softplus")

    def test_default_activation_is_identity_and_signed_output_survives(self) -> None:
        # Zeroed output weights + a negative bias should pass straight through untouched
        # by any squashing activation (softplus/exp would clamp this at ~0).
        head = GroupContrastHead(
            in_ch=4,
            hidden=4,
            n_groups=3,
            output_bias_init=-5.0,
            zero_output_weights=True,
        )
        x = torch.randn(2, 4, 8)

        out = head(x)

        assert torch.all(out < 0)
        torch.testing.assert_close(out, torch.full_like(out, -5.0))

    def test_registered_in_head_type_constructors(self) -> None:
        head = build_transfer_learning_head(
            head_type="group_contrast", activation_type="identity", in_ch=4, hidden=4, n_groups=3
        )
        assert isinstance(head, GroupContrastHead)


class TestCompositeTrackGroupHead:
    def _make(self, *, n_tracks: int = 3, n_groups: int = 4, in_ch: int = 8, hidden: int = 8):
        track_head = build_transfer_learning_head(
            head_type="transfer_mlp",
            activation_type="softplus",
            in_ch=in_ch,
            hidden=hidden,
            n_tracks=n_tracks,
        )
        group_head = build_transfer_learning_head(
            head_type="group_contrast",
            activation_type="identity",
            in_ch=in_ch,
            hidden=hidden,
            n_groups=n_groups,
        )
        return CompositeTrackGroupHead(
            track_head=track_head, group_head=group_head, track_channel_count=n_tracks
        )

    def test_concatenates_track_then_group_channels(self) -> None:
        composite = self._make(n_tracks=3, n_groups=4)
        x = torch.randn(2, 8, 16)

        out = composite(x)

        assert out.shape == (2, 7, 16)
        assert composite.track_channel_count == 3

    def test_matches_separate_head_calls(self) -> None:
        composite = self._make(n_tracks=3, n_groups=4)
        x = torch.randn(2, 8, 16)

        out = composite(x)
        expected_track = composite.track_head(x)
        expected_group = composite.group_head(x)

        torch.testing.assert_close(out[:, : composite.track_channel_count, :], expected_track)
        torch.testing.assert_close(out[:, composite.track_channel_count :, :], expected_group)

    def test_group_head_does_not_receive_track_metadata_kwargs(self) -> None:
        calls: list[dict] = []

        class RecordingGroupHead(nn.Module):
            def forward(self, x, **kwargs):
                calls.append(kwargs)
                return torch.zeros(x.shape[0], 2, x.shape[-1])

        track_head = build_transfer_learning_head(
            head_type="transfer_mlp", activation_type="softplus", in_ch=4, hidden=4, n_tracks=3
        )
        composite = CompositeTrackGroupHead(
            track_head=track_head, group_head=RecordingGroupHead(), track_channel_count=3
        )
        x = torch.randn(1, 4, 8)

        composite(x, track_condition_ids=torch.tensor([0, 1, -1]))

        assert calls == [{}]


class _DummyAdapter(nn.Module):
    """Minimal backbone stand-in with a deterministic, differentiable forward."""

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Conv1d(4, feature_dim, kernel_size=1)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.conv(input_ids)

    def iter_named_blocks(self):
        yield "conv", self.conv


class TestRegulonadoModelCompositeWiring:
    def test_single_head_path_unchanged_when_group_head_not_configured(self) -> None:
        config = RegulonadoConfig(
            feature_dim=8, n_tracks=3, head_type="transfer_mlp", head_hidden=8
        )
        model = RegulonadoModel(config=config, backbone=_DummyAdapter(feature_dim=8))

        assert not isinstance(model.head, CompositeTrackGroupHead)
        assert model.track_channel_count == 3

        tokens = torch.zeros(2, 32, dtype=torch.uint8)
        out = model(tokens)
        assert out.shape == (2, 3, 32)

    def test_group_head_is_strictly_opt_in(self) -> None:
        config = RegulonadoConfig(
            feature_dim=8,
            n_tracks=3,
            head_type="transfer_mlp",
            head_hidden=8,
            group_contrast_n_groups=0,
        )
        model = RegulonadoModel(config=config, backbone=_DummyAdapter(feature_dim=8))

        assert not isinstance(model.head, CompositeTrackGroupHead)

    def test_composite_head_built_when_group_contrast_configured(self) -> None:
        config = RegulonadoConfig(
            feature_dim=8,
            n_tracks=3,
            head_type="transfer_mlp",
            head_hidden=8,
            group_contrast_n_groups=5,
            group_contrast_hidden=8,
        )
        model = RegulonadoModel(config=config, backbone=_DummyAdapter(feature_dim=8))

        assert isinstance(model.head, CompositeTrackGroupHead)
        assert model.track_channel_count == 3

        tokens = torch.zeros(2, 32, dtype=torch.uint8)
        out = model(tokens)
        assert out.shape == (2, 3 + 5, 32)

    def test_composite_head_group_channels_use_identity_activation(self) -> None:
        # Deliberately configure the per-track head with a squashing activation; the
        # group head must still be identity-activated regardless.
        config = RegulonadoConfig(
            feature_dim=8,
            n_tracks=2,
            head_type="transfer_mlp",
            head_hidden=8,
            activation_type="softplus",
            group_contrast_n_groups=3,
            group_contrast_hidden=8,
        )
        model = RegulonadoModel(config=config, backbone=_DummyAdapter(feature_dim=8))

        with torch.no_grad():
            for param in model.head.group_head.proj[-1].parameters():
                if param.ndim == 1:
                    param.fill_(-10.0)

        tokens = torch.zeros(1, 32, dtype=torch.uint8)
        out = model(tokens)
        group_out = out[:, 2:, :]
        # softplus(-10) ~= 4.5e-5 > 0; identity keeps it strongly negative.
        assert torch.all(group_out < -1.0)
