"""Unit tests for regulonado.regions.model."""

from __future__ import annotations

import pytest
import torch
from regulonado.regions.model import (
    AttentionPool,
    CountHead,
    RegionCountConfig,
    RegionCountModel,
)

TRACK_GROUPS = [0, 0, 1, 1, 1]
LOG_SIZE_FACTORS = [0.1, -0.1, 0.0, 0.2, -0.2]
N_GROUPS = 2


def _toy_config(**overrides: object) -> RegionCountConfig:
    kwargs: dict[str, object] = dict(
        backbone_name="stub",
        d=8,
        k=4,
        track_groups=TRACK_GROUPS,
        log_size_factors=LOG_SIZE_FACTORS,
        hidden=16,
        dropout=0.0,
        eta_max=6.0,
    )
    kwargs.update(overrides)
    return RegionCountConfig(**kwargs)


# ---------------------------------------------------------------------------
# CountHead
# ---------------------------------------------------------------------------


def test_count_head_offsets_centred_per_group() -> None:
    head = CountHead(
        track_groups=TRACK_GROUPS, log_size_factors=LOG_SIZE_FACTORS, n_groups=N_GROUPS
    )
    with torch.no_grad():
        head.replicate_offsets.copy_(torch.tensor([1.0, 3.0, -2.0, 5.0, 9.0]))
    offsets = head.centred_offsets()

    group0 = torch.as_tensor(TRACK_GROUPS) == 0
    group1 = torch.as_tensor(TRACK_GROUPS) == 1
    assert offsets[group0].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert offsets[group1].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_count_head_single_track_group_gets_zero_offset() -> None:
    head = CountHead(track_groups=[0, 1, 1], log_size_factors=[0.0, 0.1, -0.1], n_groups=2)
    with torch.no_grad():
        head.replicate_offsets.copy_(torch.tensor([7.0, 2.0, -2.0]))
    offsets = head.centred_offsets()
    assert offsets[0].item() == pytest.approx(0.0)


def test_count_head_no_offsets_returns_zeros() -> None:
    head = CountHead(
        track_groups=TRACK_GROUPS,
        log_size_factors=LOG_SIZE_FACTORS,
        n_groups=N_GROUPS,
        replicate_offsets=False,
    )
    assert head.replicate_offsets is None
    offsets = head.centred_offsets()
    assert torch.allclose(offsets, torch.zeros_like(offsets))


def test_count_head_forward_shape_and_formula() -> None:
    head = CountHead(
        track_groups=TRACK_GROUPS, log_size_factors=LOG_SIZE_FACTORS, n_groups=N_GROUPS
    )
    eta = torch.tensor([[0.5, -0.3], [1.0, 0.0]])
    log_mu = head(eta)
    assert log_mu.shape == (2, len(TRACK_GROUPS))
    expected_00 = eta[0, TRACK_GROUPS[0]] + head.log_size_factors[0] + head.centred_offsets()[0]
    assert log_mu[0, 0].item() == pytest.approx(expected_00.item())


def test_count_head_rejects_out_of_range_groups() -> None:
    with pytest.raises(ValueError, match="track_groups must index"):
        CountHead(track_groups=[0, 2], log_size_factors=[0.0, 0.0], n_groups=2)


def test_count_head_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="one entry per track"):
        CountHead(track_groups=[0, 1], log_size_factors=[0.0], n_groups=2)


# ---------------------------------------------------------------------------
# AttentionPool
# ---------------------------------------------------------------------------


def test_attention_pool_output_shape() -> None:
    pool = AttentionPool(feature_dim=8)
    features = torch.randn(3, 5, 8)
    pooled = pool(features)
    assert pooled.shape == (3, 8)


def test_attention_pool_is_convex_combination_of_bins() -> None:
    """Pooled output must lie in the convex hull of the K bin vectors (softmax weights)."""
    pool = AttentionPool(feature_dim=4)
    features = torch.eye(4).unsqueeze(0)  # [1, 4, 4]: bin i is the i-th standard basis vector
    pooled = pool(features)[0]
    assert pooled.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (pooled >= 0).all()


# ---------------------------------------------------------------------------
# RegionCountConfig validation
# ---------------------------------------------------------------------------


def test_config_infers_n_groups_from_track_groups() -> None:
    config = _toy_config()
    assert config.n_groups == N_GROUPS
    assert config.n_tracks == len(TRACK_GROUPS)


def test_config_rejects_mismatched_n_tracks() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        _toy_config(n_tracks=99)


def test_config_rejects_unknown_noise() -> None:
    with pytest.raises(ValueError, match="count_noise must be one of"):
        _toy_config(count_noise="bogus")


def test_config_rejects_mismatched_track_names() -> None:
    with pytest.raises(ValueError, match="track_names"):
        _toy_config(track_names=["only_one"])


# ---------------------------------------------------------------------------
# RegionCountModel: forward/backward, save/load round-trip
# ---------------------------------------------------------------------------


def test_forward_without_labels_returns_no_loss() -> None:
    model = RegionCountModel(_toy_config())
    features = torch.randn(2, 4, 8)
    output = model(features)
    assert output.loss is None
    assert output.logits.shape == (2, N_GROUPS)
    assert (output.logits >= 0).all()  # exp(eta) is non-negative


def test_forward_backward_with_random_features() -> None:
    model = RegionCountModel(_toy_config())
    features = torch.randn(4, 4, 8, dtype=torch.float16)  # cache dtype
    labels = torch.tensor(
        [
            [1.0, 2.0, 0.0, 3.0, 1.0],
            [0.0, 1.0, 2.0, 0.0, 4.0],
            [float("nan"), 1.0, 1.0, 2.0, 0.0],
            [2.0, 2.0, 3.0, 1.0, 1.0],
        ]
    )
    output = model(features, labels=labels)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert any(norm > 0 for norm in grad_norms)


def test_eta_cap_bounds_logits_below_exp_eta_max() -> None:
    config = _toy_config(eta_max=1.0, hidden=4)
    model = RegionCountModel(config)
    # Drive the MLP output far above the cap by scaling up the input features.
    features = torch.randn(8, 4, 8) * 1000.0
    output = model(features)
    assert (output.logits <= math_exp(1.0) + 1e-3).all()


def math_exp(x: float) -> float:
    import math

    return math.exp(x)


def test_count_freeze_noise_stops_log_noise_gradient() -> None:
    config = _toy_config(count_freeze_noise=True)
    model = RegionCountModel(config)
    assert not model.count_head.log_noise.requires_grad


def test_count_head_parameter_names() -> None:
    model = RegionCountModel(_toy_config())
    names = model.count_head_parameter_names()
    assert set(names) == {"count_head.replicate_offsets", "count_head.log_noise"}
    named = dict(model.named_parameters())
    for name in names:
        assert name in named


def test_save_and_load_round_trip(tmp_path) -> None:
    model = RegionCountModel(_toy_config())
    model.eval()
    features = torch.randn(2, 4, 8)
    with torch.no_grad():
        before = model(features).logits

    save_dir = tmp_path / "region_count_model"
    model.save_pretrained(save_dir)

    reloaded = RegionCountModel.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(features).logits

    assert torch.allclose(before, after, atol=1e-6)
    assert reloaded.config.track_groups == TRACK_GROUPS
    assert reloaded.config.log_size_factors == pytest.approx(LOG_SIZE_FACTORS)
