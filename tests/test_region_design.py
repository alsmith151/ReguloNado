"""Tests for regulonado.design.region_predictor and the region-model energy factories in
regulonado.design.objective.

Follows tests/test_design.py's and tests/test_attribution.py's convention of duck-typed stub
ensembles (`.predict()` only) rather than a real checkpoint -- region predictors/ensembles are
plumbing over an already-loaded model, so nothing here needs `load_region_count_model` to run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from regulonado.design.attribution import ism_scan
from regulonado.design.objective import (
    SpecificityEnergy,
    contrast_energy,
    group_track_groups,
    max_offtarget_energy,
    worst_offtarget_energy,
)
from regulonado.design.region_predictor import RegionContrastReadout
from regulonado.design.sequence import Seed
from regulonado.genomics import Window, one_hot
from regulonado.model.adapters import BaseBackboneAdapter

CONTEXT = 400
MOTIF = slice(200, 240)


class _StubRegionEnsemble:
    """Duck-typed `.predict()` stand-in: returns a fixed `(F, B, G, 1)` tensor regardless of
    input. Used wherever a test only needs to pin the arithmetic of the objective/readout
    against known numbers, not simulate a sequence-dependent model."""

    def __init__(self, preds: torch.Tensor) -> None:
        self.preds = preds

    def predict(self, one_hot_batch):
        return self.preds


class _FakeRegionEnsemble:
    """2 folds x 3 groups; only group 1 (the target) responds, and only to 'A' content inside
    MOTIF -- the region-model analogue of test_attribution.py's `_MotifEnsemble`."""

    group_names = ["g0", "g1", "g2"]
    track_names = group_names

    def __init__(self, n_folds: int = 2) -> None:
        self.n_folds = n_folds

    def predict(self, one_hot_batch):
        x = torch.as_tensor(np.asarray(one_hot_batch)).float()
        batch = x.shape[0]
        signal = x[:, 0, MOTIF].sum(dim=1)  # A-count inside the motif
        out = torch.zeros(self.n_folds, batch, 3, 1)
        out[:, :, 1, 0] = signal.view(1, batch)
        return out


def _window() -> Window:
    return Window(chrom="chr1", pred_start=180, pred_end=280, ctx_start=0, ctx_end=CONTEXT)


def _seed(name: str = "c1", start: int = 180, end: int = 280) -> Seed:
    return Seed(
        name=name,
        chrom="chr1",
        cand_start=start,
        cand_end=end,
        window=_window(),
        fold_label="test",
        editable=slice(start, end),
        bins=slice(0, 1),
    )


def _motif_context(seed_value: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed_value)
    context = one_hot("".join(rng.choice(list("ACGT"), CONTEXT)))
    context[:, MOTIF] = 0
    context[0, MOTIF] = 1  # a pure poly-A block
    return context


# --------------------------------------------------------------------------- #
# 1. The algebraic pin test                                                  #
# --------------------------------------------------------------------------- #
def test_contrast_energy_matches_region_contrast_readout_algebraically():
    """Pins objective.contrast_energy's `specificity` against RegionContrastReadout's centred
    score so the two cannot silently drift apart.

    For offtarget_reduction="mean", target_alpha=1.0:
        s[t] - mean_all(s) == (G-1)/G * (s[t] - mean_{g!=t}(s)) == -(G-1)/G * specificity
    i.e. RegionContrastReadout's centred score (LHS) equals -(G-1)/G * SpecificityEnergy's
    `specificity` -- equivalently `specificity == -(G / (G - 1)) * readout`. This is verified
    both algebraically and numerically against the real implementation (see the batch-C task
    notes); it is the reverse coefficient direction from a literal reading of "specificity ==
    -(G-1)/G * readout", which does not hold for G != a degenerate case and was checked not to
    hold numerically here.
    """
    torch.manual_seed(0)
    n_folds, batch, n_groups = 3, 4, 5
    preds = torch.rand(n_folds, batch, n_groups, 1) * 5.0
    ensemble = _StubRegionEnsemble(preds)
    group_names = [f"g{i}" for i in range(n_groups)]
    target_index = 2
    groups = group_track_groups(group_names, target=group_names[target_index])

    energy_fn = contrast_energy(ensemble, groups, slice(0, 1))
    dummy_batch = np.zeros((batch, 4, 10))
    result = energy_fn.forward(dummy_batch)

    readout = RegionContrastReadout(ensemble, group_index=target_index)
    scores, _ = readout(dummy_batch)

    scores = torch.as_tensor(scores, dtype=torch.float32)
    expected_readout = -(n_groups - 1) / n_groups * result.specificity
    assert torch.allclose(scores, expected_readout, atol=1e-6)

    expected_specificity = -(n_groups / (n_groups - 1)) * scores
    assert torch.allclose(result.specificity, expected_specificity, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. RegionContrastReadout                                                   #
# --------------------------------------------------------------------------- #
def test_region_contrast_readout_centring_arithmetic():
    # F=1, B=2, G=3: batch0 = [1,2,3], batch1 = [4,5,6].
    preds = torch.tensor([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]])
    ensemble = _StubRegionEnsemble(preds)
    readout = RegionContrastReadout(ensemble, group_index=0, centre=True)
    scores, per_fold = readout(np.zeros((2, 4, 10)))
    # mean_all = 2 and 5; target = 1 and 4; centred = -1 and -1.
    assert scores.tolist() == pytest.approx([-1.0, -1.0])
    assert per_fold.shape == (1, 2)


def test_region_contrast_readout_centre_false_returns_raw_target():
    preds = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    ensemble = _StubRegionEnsemble(preds)
    readout = RegionContrastReadout(ensemble, group_index=2, centre=False)
    scores, _ = readout(np.zeros((1, 4, 10)))
    assert scores.tolist() == pytest.approx([3.0])


def test_region_contrast_readout_fold_reduction_mean_vs_median():
    # 3 folds, one outlier fold, G=3.
    preds = torch.zeros(3, 1, 3, 1)
    preds[0, 0, :, 0] = torch.tensor([3.0, 1.0, 1.0])
    preds[1, 0, :, 0] = torch.tensor([3.0, 1.0, 1.0])
    preds[2, 0, :, 0] = torch.tensor([30.0, 1.0, 1.0])
    ensemble = _StubRegionEnsemble(preds)

    mean_readout = RegionContrastReadout(ensemble, group_index=0, fold_reduction="mean")
    median_readout = RegionContrastReadout(ensemble, group_index=0, fold_reduction="median")

    mean_scores, _ = mean_readout(np.zeros((1, 4, 10)))
    median_scores, _ = median_readout(np.zeros((1, 4, 10)))

    # Per-fold centred target: fold0/1 -> 3 - 5/3 = 4/3; fold2 -> 30 - 32/3 = 58/3.
    assert median_scores[0] == pytest.approx(4.0 / 3.0)
    assert mean_scores[0] == pytest.approx((4.0 / 3.0 + 4.0 / 3.0 + 58.0 / 3.0) / 3.0)
    assert mean_scores[0] != pytest.approx(median_scores[0])


def test_region_contrast_readout_rejects_unknown_fold_reduction():
    preds = torch.zeros(1, 1, 2, 1)
    readout = RegionContrastReadout(
        _StubRegionEnsemble(preds), group_index=0, fold_reduction="nope"
    )
    with pytest.raises(ValueError, match="Unknown fold_reduction"):
        readout(np.zeros((1, 4, 10)))


# --------------------------------------------------------------------------- #
# 3. ism_scan end to end                                                     #
# --------------------------------------------------------------------------- #
def test_ism_scan_recovers_planted_motif_on_a_region_ensemble():
    seed, context = _seed(), _motif_context()
    readout = RegionContrastReadout(_FakeRegionEnsemble(), group_index=1)
    result = ism_scan(readout, seed, context)

    motif_columns = range(MOTIF.start - seed.editable.start, MOTIF.stop - seed.editable.start)
    peak = int(np.nanargmax(result.importance))
    assert peak in motif_columns

    outside_max = np.nanmax(
        np.delete(result.importance, list(motif_columns))[
            ~np.isnan(np.delete(result.importance, list(motif_columns)))
        ]
    )
    assert result.importance[peak] > outside_max


# --------------------------------------------------------------------------- #
# 4. group_track_groups                                                      #
# --------------------------------------------------------------------------- #
def test_group_track_groups_masks_are_one_hot():
    groups = group_track_groups(["a", "b", "c"], target="b")
    assert groups.target_idx.dtype == torch.bool
    assert groups.target_idx.tolist() == [False, True, False]
    assert set(groups.other_group_masks) == {"a", "c"}
    assert groups.other_group_masks["a"].tolist() == [True, False, False]
    assert groups.other_group_masks["c"].tolist() == [False, False, True]
    # Every column is one-hot across target + all other masks.
    stacked = torch.stack([groups.target_idx, *groups.other_group_masks.values()])
    assert (stacked.sum(dim=0) == 1).all()


def test_group_track_groups_unknown_target_error_names_available_groups():
    with pytest.raises(ValueError, match="matches no group"):
        group_track_groups(["a", "b"], target="nonexistent")


def test_group_track_groups_exclude_groups():
    groups = group_track_groups(["a", "b", "c"], target="a", exclude_groups=("c",))
    assert groups.labels == ["a", "b", None]
    assert set(groups.other_group_masks) == {"b"}


def test_group_track_groups_unknown_exclude_raises():
    with pytest.raises(ValueError, match="Excluded group"):
        group_track_groups(["a", "b"], target="a", exclude_groups=("nope",))


def test_group_track_groups_empty_or_none_group_names_raises():
    with pytest.raises(ValueError, match="group_names is empty or None"):
        group_track_groups([], target="a")
    with pytest.raises(ValueError, match="group_names is empty or None"):
        group_track_groups(None, target="a")


# --------------------------------------------------------------------------- #
# 5. SpecificityEnergy with bins=slice(0, 1) on the region stub              #
# --------------------------------------------------------------------------- #
def test_specificity_energy_region_bins_per_group_scores():
    # F=1, B=2, G=3: batch0 = [1,2,3], batch1 = [4,5,6]; target = g0.
    preds = torch.tensor([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]])
    ensemble = _StubRegionEnsemble(preds)
    groups = group_track_groups(["g0", "g1", "g2"], target="g0")
    energy_fn = SpecificityEnergy(ensemble, groups, slice(0, 1), offtarget_reduction="mean")

    result = energy_fn.forward(np.zeros((2, 4, 10)))

    assert result.target.tolist() == pytest.approx([1.0, 4.0])
    # offtarget mean: (2+3)/2 = 2.5, (5+6)/2 = 5.5
    assert result.specificity.tolist() == pytest.approx([1.5, 1.5])
    assert result.group_names == ["g0", "g1", "g2"]
    assert result.per_group[0].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert result.per_group[1].tolist() == pytest.approx([4.0, 5.0, 6.0])


# --------------------------------------------------------------------------- #
# 6. Named energy factories                                                  #
# --------------------------------------------------------------------------- #
def test_energy_factories_use_the_documented_offtarget_reduction():
    ensemble = _StubRegionEnsemble(torch.zeros(1, 1, 3, 1))
    groups = group_track_groups(["a", "b", "c"], target="a")

    assert contrast_energy(ensemble, groups, slice(0, 1)).offtarget_reduction == "mean"
    assert worst_offtarget_energy(ensemble, groups, slice(0, 1)).offtarget_reduction == "logsumexp"
    assert max_offtarget_energy(ensemble, groups, slice(0, 1)).offtarget_reduction == "max"


def test_energy_factories_reject_overriding_offtarget_reduction():
    ensemble = _StubRegionEnsemble(torch.zeros(1, 1, 3, 1))
    groups = group_track_groups(["a", "b", "c"], target="a")
    with pytest.raises(TypeError, match="offtarget_reduction is fixed"):
        contrast_energy(ensemble, groups, slice(0, 1), offtarget_reduction="max")


# --------------------------------------------------------------------------- #
# 7. checkpoint_model_kind / load_fold_ensemble dispatch (batch E1)          #
# --------------------------------------------------------------------------- #
def _profile_checkpoint(tmp_path, name: str, monkeypatch, *, context: int = 100):
    """A real, tiny profile checkpoint -- mirrors tests/test_design.py's ``_make_checkpoint``."""
    from conftest import TinyBackbone
    from regulonado.model import RegulonadoConfig as ModelConfig
    from regulonado.model import RegulonadoModel, TransferMLPHead, adapters

    monkeypatch.setattr(adapters, "build_backbone_architecture", lambda *a, **k: TinyBackbone())
    config = ModelConfig(
        backbone_type="tiny",
        head_type="transfer_mlp",
        head_hidden=4,
        mlp_hidden=4,
        feature_dim=8,
        n_tracks=2,
        context_length=context,
        n_pred_bins=4,
        bin_size=10,
        track_names=["alpha", "beta"],
    )
    model = RegulonadoModel(
        config, backbone=TinyBackbone(), head=TransferMLPHead(in_ch=8, hidden=4, n_tracks=2)
    )
    out_dir = tmp_path / name
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


class _TinyConvAdapter(BaseBackboneAdapter):
    """Minimal live-trunk backbone adapter -- mirrors
    tests/test_regions_inference.py's ``_TinyConvAdapter``, kept self-contained here."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.output_bin_size = 32
        self.input_multiple = 32
        self.fixed_input_length = None
        self.feature_dim = feature_dim
        self.embed = torch.nn.Conv1d(4, 8, 32, stride=32)
        self.block = torch.nn.Conv1d(8, feature_dim, 3, padding=1)

    def output_span(self, input_length: int) -> tuple[int, int]:
        return 0, input_length // 32

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.block(torch.relu(self.embed(input_ids.float())))

    def iter_named_blocks(self):
        yield "embed", self.embed
        yield "block", self.block


def _region_checkpoint(tmp_path, name: str, monkeypatch):
    """A real, tiny live-trunk region-count checkpoint -- mirrors
    tests/test_regions_inference.py's ``_build_live_model``."""
    import dataclasses

    import regulonado.model.adapters as adapters_module
    from regulonado.training.regions.live import (
        LiveTrunk,
        TrunkFinetuneConfig,
        TrunkWindow,
        prepare_trunk,
    )
    from regulonado.training.regions.model import RegionCountConfig, RegionCountModel

    def _seeded_adapter(_spec=None):
        torch.manual_seed(1234)
        return _TinyConvAdapter()

    monkeypatch.setattr(adapters_module, "build_backbone_adapter", _seeded_adapter)
    adapter = _seeded_adapter()
    finetune = TrunkFinetuneConfig(finetune="frozen")
    prepare_trunk(adapter, finetune)
    window = TrunkWindow.for_adapter(adapter, 256, 100)
    trunk = LiveTrunk(adapter, window)
    trunk_info = {
        "backbone": {"type": "stub", "pretrained": "stub-ckpt", "features": "trunk"},
        "window": window.as_dict(),
        "finetune": dataclasses.asdict(finetune),
    }
    config = RegionCountConfig(
        k=5,
        d=4,
        track_groups=[0, 0, 1],
        log_size_factors=[0.0, 0.1, -0.1],
        hidden=8,
        group_names=["g0", "g1"],
        trunk=trunk_info,
    )
    model = RegionCountModel(config, trunk=trunk)
    out_dir = tmp_path / name
    model.save_pretrained(out_dir)
    return out_dir


def test_checkpoint_model_kind_sniffs_profile_and_region(tmp_path, monkeypatch):
    from regulonado.design.predictor import checkpoint_model_kind

    profile_dir = _profile_checkpoint(tmp_path, "profile", monkeypatch)
    assert checkpoint_model_kind(profile_dir) == "profile"

    region_dir = _region_checkpoint(tmp_path, "region", monkeypatch)
    assert checkpoint_model_kind(region_dir) == "region_counts"


def test_checkpoint_model_kind_no_config_json_is_profile(tmp_path):
    from regulonado.design.predictor import checkpoint_model_kind

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    assert checkpoint_model_kind(legacy_dir) == "profile"


def test_load_fold_ensemble_builds_a_profile_ensemble(tmp_path, monkeypatch):
    from regulonado.design.predictor import FoldEnsemble, FoldSpec, load_fold_ensemble

    checkpoint = _profile_checkpoint(tmp_path, "profile", monkeypatch)
    ensemble = load_fold_ensemble([FoldSpec(checkpoint, name="fold_a")], device="cpu")
    assert isinstance(ensemble, FoldEnsemble)
    assert ensemble.track_names == ["alpha", "beta"]


def test_load_fold_ensemble_builds_a_region_ensemble(tmp_path, monkeypatch):
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble
    from regulonado.design.region_predictor import RegionFoldEnsemble

    checkpoint = _region_checkpoint(tmp_path, "region", monkeypatch)
    ensemble = load_fold_ensemble([FoldSpec(checkpoint, name="fold_a")], device="cpu")
    assert isinstance(ensemble, RegionFoldEnsemble)
    assert ensemble.group_names == ["g0", "g1"]


def test_load_fold_ensemble_disagreeing_folds_raise_naming_both_dirs(tmp_path, monkeypatch):
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble

    profile_dir = _profile_checkpoint(tmp_path, "profile", monkeypatch)
    region_dir = _region_checkpoint(tmp_path, "region", monkeypatch)

    with pytest.raises(ValueError, match="disagree on model kind") as excinfo:
        load_fold_ensemble(
            [FoldSpec(profile_dir, name="fold_a"), FoldSpec(region_dir, name="fold_b")],
            device="cpu",
        )
    assert str(profile_dir) in str(excinfo.value)
    assert str(region_dir) in str(excinfo.value)


def test_load_fold_ensemble_explicit_model_kind_contradiction_raises(tmp_path, monkeypatch):
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble

    profile_dir = _profile_checkpoint(tmp_path, "profile", monkeypatch)
    with pytest.raises(ValueError, match="model_kind='region_counts'"):
        load_fold_ensemble(
            [FoldSpec(profile_dir, name="fold_a")], device="cpu", model_kind="region_counts"
        )


def test_load_fold_ensemble_needs_at_least_one_fold():
    from regulonado.design.predictor import load_fold_ensemble

    with pytest.raises(ValueError, match="at least one fold"):
        load_fold_ensemble([])


# --------------------------------------------------------------------------- #
# 8. run._build_energy_fn dispatch (batch E1, Task 3)                        #
# --------------------------------------------------------------------------- #
def test_run_build_energy_fn_dispatches_to_the_named_factory_for_a_region_ensemble(
    tmp_path, monkeypatch
):
    from regulonado.config.models import DesignConfig, DesignTarget
    from regulonado.design import run as design_run
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble

    checkpoint = _region_checkpoint(tmp_path, "region", monkeypatch)
    ensemble = load_fold_ensemble([FoldSpec(checkpoint, name="fold_a")], device="cpu")
    groups = group_track_groups(ensemble.group_names, target="g0")
    config = DesignConfig(
        candidates="c.bed",
        energy="worst_offtarget",
        targets=[DesignTarget(name="t1", target="g0")],
    )

    energy_fn = design_run._build_energy_fn(config, ensemble, groups, slice(0, 1))

    assert isinstance(energy_fn, SpecificityEnergy)
    assert energy_fn.offtarget_reduction == "logsumexp"  # worst_offtarget_energy's factory


def test_run_build_energy_fn_uses_plain_specificity_energy_for_a_profile_ensemble(
    tmp_path, monkeypatch
):
    from regulonado.config.models import DesignConfig, DesignTarget
    from regulonado.design import run as design_run
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble

    checkpoint = _profile_checkpoint(tmp_path, "profile", monkeypatch)
    ensemble = load_fold_ensemble([FoldSpec(checkpoint, name="fold_a")], device="cpu")
    # DesignConfig.energy is ignored for a profile ensemble; offtarget_reduction (the "old"
    # config knob) still governs it, exactly as before batch E1.
    config = DesignConfig(
        candidates="c.bed",
        offtarget_reduction="mean",
        targets=[DesignTarget(name="t1", target="alpha")],
    )
    groups = group_track_groups(ensemble.track_names, target="alpha")

    energy_fn = design_run._build_energy_fn(config, ensemble, groups, slice(0, 1))

    assert isinstance(energy_fn, SpecificityEnergy)
    assert energy_fn.offtarget_reduction == "mean"


# --------------------------------------------------------------------------- #
# 9. Neutral-flank wiring in attribute_run._score_one_candidate (Task 4)     #
# --------------------------------------------------------------------------- #
class _FakeFastaContig:
    def __init__(self, seq: str) -> None:
        self._seq = seq

    def __getitem__(self, sl):
        return self._seq[sl]

    def __len__(self) -> int:
        return len(self._seq)


class _FakeFasta:
    def __init__(self, sequences: dict[str, str]) -> None:
        self._sequences = sequences

    def __getitem__(self, chrom: str) -> _FakeFastaContig:
        return _FakeFastaContig(self._sequences[chrom])


class _FlankProbeEnsemble:
    """Duck-typed FoldEnsemble stand-in with the attributes/`.predict()` attribute_run's
    ``_score_one_candidate`` needs, but no sequence-dependence -- only used to probe whether
    ``apply_neutral_flanks`` gets called, not to check search dynamics."""

    track_names = ["t0", "t1"]
    context_length = 200
    crop_bp = 50
    bin_size = 10

    def predict(self, one_hot_batch):
        x = torch.as_tensor(np.asarray(one_hot_batch)).float()
        return torch.zeros(1, x.shape[0], 2, 10)


def _flank_probe_attribution_config(tmp_path, *, flank_mode: str):
    from regulonado.config.models import AttributionConfig, AttributionTarget

    return AttributionConfig(
        candidates=str(tmp_path / "c.bed"),
        targets=[AttributionTarget(name="t1", track="t0")],
        flank_mode=flank_mode,
    )


def _flank_probe_seed() -> Seed:
    window = Window(chrom="chr1", pred_start=50, pred_end=150, ctx_start=0, ctx_end=200)
    return Seed(
        name="c1",
        chrom="chr1",
        cand_start=90,
        cand_end=110,
        window=window,
        fold_label=None,
        editable=slice(90, 110),
        bins=slice(4, 6),
    )


def test_flank_mode_genomic_skips_apply_neutral_flanks_entirely(tmp_path, monkeypatch):
    import regulonado.design.sequence as sequence_module
    from regulonado.design import attribute_run

    def _boom(*args, **kwargs):
        raise AssertionError("apply_neutral_flanks must not be called when flank_mode='genomic'")

    monkeypatch.setattr(sequence_module, "apply_neutral_flanks", _boom)

    config = _flank_probe_attribution_config(tmp_path, flank_mode="genomic")
    seed = _flank_probe_seed()
    fasta = _FakeFasta({"chr1": "A" * 200})

    record = attribute_run._score_one_candidate(
        config, 1, 1, seed, _FlankProbeEnsemble(), [0], fasta, {"chr1": 200}, None
    )
    assert record.seed is seed  # ran to completion without the patched apply_neutral_flanks


def test_flank_mode_shuffle_calls_apply_neutral_flanks(tmp_path, monkeypatch):
    import regulonado.design.sequence as sequence_module
    from regulonado.design import attribute_run

    calls = []
    real_apply = sequence_module.apply_neutral_flanks

    def _spy(context, keep, *, mode, rng):
        calls.append((keep, mode))
        return real_apply(context, keep, mode=mode, rng=rng)

    monkeypatch.setattr(sequence_module, "apply_neutral_flanks", _spy)

    config = _flank_probe_attribution_config(tmp_path, flank_mode="shuffle")
    seed = _flank_probe_seed()
    fasta = _FakeFasta({"chr1": "ACGT" * 50})

    attribute_run._score_one_candidate(
        config, 1, 1, seed, _FlankProbeEnsemble(), [0], fasta, {"chr1": 200}, None
    )
    assert len(calls) == 1
    assert calls[0][1] == "shuffle"


def test_flank_keep_span_candidate_vs_scored_span_and_padding():
    from regulonado.config.models import AttributionConfig, AttributionTarget
    from regulonado.design import attribute_run

    seed = _flank_probe_seed()
    ensemble = _FlankProbeEnsemble()

    candidate_config = AttributionConfig(
        candidates="c.bed",
        targets=[AttributionTarget(name="t1", track="t0")],
        flank_keep="candidate",
        flank_keep_bp=5,
    )
    span = attribute_run._flank_keep_span(candidate_config, seed, ensemble)
    assert span == slice(85, 115)  # editable [90,110) widened by 5bp each side

    scored_config = AttributionConfig(
        candidates="c.bed",
        targets=[AttributionTarget(name="t1", track="t0")],
        flank_keep="scored_span",
        flank_keep_bp=0,
    )
    span = attribute_run._flank_keep_span(scored_config, seed, ensemble)
    # bins=[4,6) * bin_size 10 + crop_bp 50 -> [90, 110)
    assert span == slice(90, 110)

    clamped_config = AttributionConfig(
        candidates="c.bed",
        targets=[AttributionTarget(name="t1", track="t0")],
        flank_keep="candidate",
        flank_keep_bp=1000,
    )
    span = attribute_run._flank_keep_span(clamped_config, seed, ensemble)
    assert span == slice(0, 200)  # clamped to the context
