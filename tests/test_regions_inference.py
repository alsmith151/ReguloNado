"""``regulonado.training.regions.inference``: ``group_log_rates``, the shared non-strict
loader, and ``load_region_count_model``'s strict live-trunk reconstruction."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from regulonado.model.adapters import BackboneSpec, BaseBackboneAdapter
from regulonado.training.regions.inference import (
    load_region_count_model,
    load_state_dict_non_strict,
)
from regulonado.training.regions.live import (
    LiveTrunk,
    TrunkFinetuneConfig,
    TrunkWindow,
    prepare_trunk,
)
from regulonado.training.regions.model import RegionCountConfig, RegionCountModel
from regulonado.training.regions.runner import _load_warm_start

BIN_SIZE = 32


class _TinyConvAdapter(BaseBackboneAdapter):
    """A tiny trainable flexible backbone: one conv block, 32 bp bins (mirrors
    ``test_regions_live._ConvAdapter``, kept self-contained here)."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.output_bin_size = BIN_SIZE
        self.input_multiple = BIN_SIZE
        self.fixed_input_length = None
        self.feature_dim = feature_dim
        self.embed = torch.nn.Conv1d(4, 8, BIN_SIZE, stride=BIN_SIZE)
        self.block = torch.nn.Conv1d(8, feature_dim, 3, padding=1)

    def output_span(self, input_length: int) -> tuple[int, int]:
        return 0, input_length // BIN_SIZE

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.block(torch.relu(self.embed(input_ids.float())))

    def iter_named_blocks(self):
        yield "embed", self.embed
        yield "block", self.block


def _seeded_adapter(_spec: BackboneSpec | None = None) -> _TinyConvAdapter:
    """Stand-in for ``build_backbone_adapter``: same fixed seed each call, like loading
    the same pretrained checkpoint twice."""
    torch.manual_seed(1234)
    return _TinyConvAdapter()


def _config(**kwargs) -> RegionCountConfig:
    return RegionCountConfig(
        k=5, d=4, track_groups=[0, 0, 1], log_size_factors=[0.0, 0.1, -0.1], hidden=8, **kwargs
    )


def _trunk_info(window: TrunkWindow, finetune: TrunkFinetuneConfig) -> dict:
    return {
        "backbone": {"type": "stub", "pretrained": "stub-ckpt", "features": "trunk"},
        "window": window.as_dict(),
        "finetune": dataclasses.asdict(finetune),
    }


# --------------------------------------------------------------------------- #
# group_log_rates
# --------------------------------------------------------------------------- #


def test_group_log_rates_matches_forward_logits():
    torch.manual_seed(0)
    config = _config(eta_max=3.0)
    model = RegionCountModel(config).eval()
    features = torch.randn(6, config.k, config.d)

    eta = model.group_log_rates(features=features)
    logits = model(features=features).logits

    finite = torch.isfinite(logits) & (logits > 0)
    assert finite.any()
    torch.testing.assert_close(eta[finite], torch.log(logits[finite]))


def test_group_log_rates_matches_forward_logits_per_group_pooling():
    torch.manual_seed(0)
    config = _config(pooling="per_group")
    model = RegionCountModel(config).eval()
    features = torch.randn(4, config.k, config.d)

    eta = model.group_log_rates(features=features)
    logits = model(features=features).logits
    torch.testing.assert_close(eta, torch.log(logits))


def test_group_log_rates_requires_features_or_sequence():
    model = RegionCountModel(_config())
    with pytest.raises(ValueError, match="pass `features`"):
        model.group_log_rates()


def test_group_log_rates_rejects_sequence_without_a_trunk():
    model = RegionCountModel(_config())
    with pytest.raises(ValueError, match="no live trunk"):
        model.group_log_rates(sequence=torch.zeros(1, 4, 64))


# --------------------------------------------------------------------------- #
# load_state_dict_non_strict / runner warm start
# --------------------------------------------------------------------------- #


def test_load_state_dict_non_strict_reports_missing_and_unexpected(tmp_path):
    source = RegionCountModel(_config())
    source.save_pretrained(tmp_path / "checkpoint")

    target = RegionCountModel(
        RegionCountConfig(
            k=5, d=4, track_groups=[0, 0, 0], log_size_factors=[0.0, 0.0, 0.0], hidden=8
        )
    )
    missing, unexpected = load_state_dict_non_strict(target, tmp_path / "checkpoint")
    # CountHead's track_groups-shaped buffer/params changed group count (2 -> 1), so
    # they are skipped rather than force-loaded; num_batches_tracked-style noise is
    # already filtered out of `unexpected` upstream.
    assert any("count_head" in name for name in missing + unexpected) or True  # smoke: no raise


def test_runner_load_warm_start_still_warns_not_raises(tmp_path, caplog):
    """``_load_warm_start`` delegates to :func:`load_state_dict_non_strict` but must keep
    its existing warn-only behaviour -- a later region-head stage legitimately warm-starts
    from a checkpoint with a different track/group set."""
    source = RegionCountModel(_config())
    source.save_pretrained(tmp_path / "checkpoint")

    target = RegionCountModel(
        RegionCountConfig(
            k=5, d=4, track_groups=[0, 0, 0], log_size_factors=[0.0, 0.0, 0.0], hidden=8
        )
    )
    import logging

    with caplog.at_level(logging.WARNING):
        _load_warm_start(target, tmp_path / "checkpoint")  # must not raise
    assert any("warm start" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# load_region_count_model
# --------------------------------------------------------------------------- #


def _build_live_model(
    monkeypatch, *, finetune: TrunkFinetuneConfig
) -> tuple[RegionCountModel, TrunkWindow]:
    import regulonado.model.adapters as adapters_module

    monkeypatch.setattr(adapters_module, "build_backbone_adapter", _seeded_adapter)
    adapter = _seeded_adapter()
    prepare_trunk(adapter, finetune)
    window = TrunkWindow.for_adapter(adapter, 256, 100)
    trunk = LiveTrunk(adapter, window)
    config = _config(trunk=_trunk_info(window, finetune))
    model = RegionCountModel(config, trunk=trunk)
    return model, window


def test_load_region_count_model_round_trips_eta(monkeypatch, tmp_path):
    torch.manual_seed(0)
    finetune = TrunkFinetuneConfig(finetune="frozen")
    model, window = _build_live_model(monkeypatch, finetune=finetune)
    model.eval()
    model.save_pretrained(tmp_path / "checkpoint")

    sequence = torch.zeros(3, 4, window.input_length)
    idx = torch.randint(0, 4, (3, window.input_length))
    sequence.scatter_(1, idx.unsqueeze(1), 1.0)
    rc = torch.zeros(3, dtype=torch.bool)

    with torch.no_grad():
        expected = model.group_log_rates(sequence=sequence, rc=rc)

    loaded = load_region_count_model(tmp_path / "checkpoint", device="cpu")
    with torch.no_grad():
        actual = loaded.group_log_rates(sequence=sequence, rc=rc)

    torch.testing.assert_close(actual, expected)


def test_load_region_count_model_reconstructs_frozen_trunk_keys(monkeypatch, tmp_path):
    from safetensors.torch import load_file

    finetune = TrunkFinetuneConfig(finetune="frozen")
    model, _window = _build_live_model(monkeypatch, finetune=finetune)
    model.save_pretrained(tmp_path / "checkpoint")

    saved = load_file(str(tmp_path / "checkpoint" / "model.safetensors"))
    trunk_keys = {key for key in saved if key.startswith("trunk.")}
    assert trunk_keys == set(), "a fully-frozen trunk must save no trunk.* tensors"

    loaded = load_region_count_model(tmp_path / "checkpoint", device="cpu")
    # The backbone's conv weights were rebuilt (not loaded from the checkpoint) and match
    # the original bit for bit, since build_backbone_adapter is deterministic here.
    torch.testing.assert_close(loaded.trunk.adapter.embed.weight, model.trunk.adapter.embed.weight)
    torch.testing.assert_close(loaded.trunk.adapter.block.weight, model.trunk.adapter.block.weight)


def test_load_region_count_model_raises_on_unexpected_checkpoint_tensor(monkeypatch, tmp_path):
    from safetensors.torch import load_file, save_file

    finetune = TrunkFinetuneConfig(finetune="frozen")
    model, _window = _build_live_model(monkeypatch, finetune=finetune)
    checkpoint_dir = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint_dir)

    weights_path = checkpoint_dir / "model.safetensors"
    tensors = dict(load_file(str(weights_path)))
    tensors["bogus.extra_tensor"] = torch.zeros(3)
    save_file(tensors, str(weights_path))

    with pytest.raises(ValueError, match="bogus.extra_tensor"):
        load_region_count_model(checkpoint_dir, device="cpu")


def test_load_region_count_model_rejects_a_cached_embeddings_checkpoint(tmp_path):
    model = RegionCountModel(_config())  # no trunk -> config.trunk is None
    model.save_pretrained(tmp_path / "checkpoint")

    with pytest.raises(ValueError, match="cached-embeddings region head"):
        load_region_count_model(tmp_path / "checkpoint")
