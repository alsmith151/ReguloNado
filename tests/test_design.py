"""Tests for the regulonado.design module (see docs/design.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from regulonado.design.objective import (
    SpecificityEnergy,
    TrackGroups,
    resolve_track_groups,
)
from regulonado.design.predictor import FoldEnsemble, FoldSpec
from regulonado.design.search import AdaLead, AdaLeadConfig, adalead, ism_greedy
from regulonado.design.sequence import (
    DatasetWindowIndex,
    Seed,
    context_bp_to_pred_bins,
    decode,
    one_hot,
    resolve_seeds,
    reverse_complement,
)
from regulonado.inference import Window

# Tiny geometry for fast tests, matching tests/test_predict_bigwig.py's convention.
N_PRED_BINS = 4
BIN_SIZE = 10
PRED_BP = N_PRED_BINS * BIN_SIZE  # 40
CONTEXT = 100


def _write_bed(path: Path, rows: list[tuple]) -> Path:
    lines = ["\t".join(str(field) for field in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 1. context_bp_to_pred_bins                                                 #
# --------------------------------------------------------------------------- #
def test_context_bp_to_pred_bins_center_window():
    span = context_bp_to_pred_bins(
        30, 40, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )
    assert span == slice(0, 1)


def test_context_bp_to_pred_bins_out_of_crop_raises():
    with pytest.raises(ValueError, match="outside the predicted crop"):
        context_bp_to_pred_bins(
            0, 10, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
        )


# --------------------------------------------------------------------------- #
# 2. one_hot / decode / reverse_complement                                   #
# --------------------------------------------------------------------------- #
def test_one_hot_decode_roundtrip_including_n():
    seq = "ACGTN"
    encoded = one_hot(seq)
    assert encoded.shape == (4, 5)
    assert encoded.dtype == np.int8
    assert decode(encoded) == seq


def test_reverse_complement_matches_training_convention():
    encoded = one_hot("ACGT")
    rc = reverse_complement(encoded)
    assert np.array_equal(rc, np.flip(encoded, axis=(0, 1)))
    assert decode(rc) == "ACGT"  # self-complementary under reverse-complement


# --------------------------------------------------------------------------- #
# 2b. Seed resolution                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def intervals_bed(tmp_path) -> Path:
    return _write_bed(
        tmp_path / "intervals.bed",
        [
            ("chr1", 500, 540, "train"),  # pred [500,540), ctx [470,570)
            ("chr1", 510, 550, "valid"),  # pred [510,550), ctx [480,580)
            ("chr1", 600, 620, "valid"),  # pred [590,630) — tie-break row A (earlier)
            ("chr1", 620, 640, "valid"),  # pred [610,650) — tie-break row B (later)
        ],
    )


@pytest.fixture
def index(intervals_bed) -> DatasetWindowIndex:
    return DatasetWindowIndex.from_bed(
        intervals_bed, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )


def test_resolve_seeds_picks_most_centered_window_and_maps_coordinates(tmp_path, index):
    candidates = _write_bed(tmp_path / "cand1.bed", [("chr1", 510, 520, "cand1")])
    seeds = resolve_seeds(candidates, index)
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.name == "cand1"
    assert seed.fold_label == "train"  # matched the [500,540) window, not [510,550)
    assert (seed.window.pred_start, seed.window.pred_end) == (500, 540)
    assert seed.editable == slice(40, 50)  # 510-470, 520-470
    assert seed.bins == slice(1, 2)


def test_resolve_seeds_ties_break_on_bed_order(tmp_path, index):
    candidates = _write_bed(tmp_path / "tie.bed", [("chr1", 615, 625, "tie_cand")])
    seeds = resolve_seeds(candidates, index)
    assert len(seeds) == 1
    # Both tie-break windows (pred [590,630) and [610,650)) give the candidate a minimum
    # edge distance of 5; the earlier row in the intervals BED wins.
    assert (seeds[0].window.pred_start, seeds[0].window.pred_end) == (590, 630)


def test_resolve_seeds_on_missing_error_lists_offenders(tmp_path, index):
    candidates = _write_bed(
        tmp_path / "mixed.bed",
        [
            ("chr1", 510, 520, "cand1"),
            ("chr1", 480, 490, "cand2"),  # inside context, outside predicted crop
            ("chr2", 10, 20, "cand3"),  # unknown chromosome
        ],
    )
    with pytest.raises(ValueError, match="cand2"):
        resolve_seeds(candidates, index, on_missing="error")


def test_resolve_seeds_on_missing_skip_drops_unmatched(tmp_path, index):
    candidates = _write_bed(
        tmp_path / "mixed.bed",
        [("chr1", 510, 520, "cand1"), ("chr1", 480, 490, "cand2")],
    )
    seeds = resolve_seeds(candidates, index, on_missing="skip")
    assert [s.name for s in seeds] == ["cand1"]


def test_resolve_seeds_on_missing_center_synthesizes_window(tmp_path, index):
    candidates = _write_bed(tmp_path / "cand2.bed", [("chr1", 480, 490, "cand2")])
    seeds = resolve_seeds(candidates, index, on_missing="center")
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.fold_label is None
    # center=485, pred_start=465, ctx_start=435
    assert (seed.window.pred_start, seed.window.ctx_start) == (465, 435)
    assert seed.editable == slice(45, 55)


def test_resolve_seeds_pad_widens_editable_without_moving_window(tmp_path, index):
    candidates = _write_bed(tmp_path / "cand1.bed", [("chr1", 510, 520, "cand1")])
    seeds = resolve_seeds(candidates, index, pad=15)
    assert seeds[0].editable == slice(25, 65)
    assert (seeds[0].window.pred_start, seeds[0].window.pred_end) == (500, 540)


# --------------------------------------------------------------------------- #
# 3. resolve_track_groups                                                    #
# --------------------------------------------------------------------------- #
def test_resolve_track_groups_from_track_sheet_collapses_replicates(tmp_path):
    sheet = tmp_path / "tracks.csv"
    sheet.write_text(
        "sample_id,source\nt0,K562\nt1,K562\nt2,HepG2\nt3,HepG2\n"
    )
    groups = resolve_track_groups(
        ["t0", "t1", "t2", "t3"], group_by="source", target="K562", track_sheet=sheet
    )
    assert groups.target_idx.tolist() == [True, True, False, False]
    assert set(groups.other_group_masks) == {"HepG2"}
    assert groups.other_group_masks["HepG2"].tolist() == [False, False, True, True]


def test_resolve_track_groups_unknown_target_raises(tmp_path):
    sheet = tmp_path / "tracks.csv"
    sheet.write_text("sample_id,source\nt0,K562\nt1,HepG2\n")
    with pytest.raises(ValueError, match="matches no track"):
        resolve_track_groups(
            ["t0", "t1"], group_by="source", target="Nonexistent", track_sheet=sheet
        )


def test_resolve_track_groups_falls_back_to_config_metadata():
    from regulonado.model.config import RegulonadoConfig as ModelConfig

    config = ModelConfig(track_metadata={"track_cell_line_ids": [0, 0, 1, 1]})
    groups = resolve_track_groups(
        ["t0", "t1", "t2", "t3"], group_by="source", target="0", config=config
    )
    assert groups.target_idx.tolist() == [True, True, False, False]
    assert groups.other_group_masks["1"].tolist() == [False, False, True, True]


# --------------------------------------------------------------------------- #
# 4. SpecificityEnergy                                                       #
# --------------------------------------------------------------------------- #
class _DummyEnsemble:
    """2 folds x 4 tracks (target,target,off,off) x 3 bins; scores depend on batch content."""

    track_names = ["t0", "t1", "t2", "t3"]

    def predict(self, one_hot_batch):
        if isinstance(one_hot_batch, np.ndarray):
            one_hot_batch = torch.from_numpy(one_hot_batch).float()
        batch = one_hot_batch.shape[0]
        signal = one_hot_batch.reshape(batch, -1).sum(dim=1)  # (B,) — set-base count
        base = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        preds = base.unsqueeze(1).expand(2, batch, 4, 3).clone()
        preds[:, :, 0:2, :] += signal.view(1, batch, 1, 1)  # only target tracks see the signal
        return preds


def _dummy_groups() -> TrackGroups:
    return TrackGroups(
        labels=["target", "target", "off", "off"],
        target="target",
        target_idx=torch.tensor([True, True, False, False]),
        other_group_masks={"off": torch.tensor([False, False, True, True])},
    )


def test_specificity_energy_hand_computed_and_multi_track_group():
    energy_fn = SpecificityEnergy(
        _DummyEnsemble(), _dummy_groups(), slice(0, 3), offtarget_reduction="mean"
    )
    a = one_hot("ACGTA")  # 5 set bases -> signal=5
    b = one_hot("ACGTN")  # 4 set bases -> signal=4
    batch = np.stack([a, b])

    result = energy_fn(batch)

    # off_mean - target_mean = 6 - signal for both folds (see _DummyEnsemble docstring).
    assert result.energy.tolist() == pytest.approx([1.0, 2.0])
    assert result.target.tolist() == pytest.approx([13.5, 12.5])
    assert torch.allclose(result.per_fold_energy, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert result.group_names == ["target", "off"]
    assert result.per_group[:, 1].tolist() == pytest.approx([14.5, 14.5])  # off unaffected


def test_specificity_energy_batch_independence():
    energy_fn = SpecificityEnergy(
        _DummyEnsemble(), _dummy_groups(), slice(0, 3), offtarget_reduction="mean"
    )
    a = one_hot("ACGTA")
    b = one_hot("ACGTN")
    batched = energy_fn(np.stack([a, b])).energy
    individually = torch.stack(
        [energy_fn(a[None]).energy[0], energy_fn(b[None]).energy[0]]
    )
    assert torch.allclose(batched, individually)


# --------------------------------------------------------------------------- #
# 5. ism_greedy                                                              #
# --------------------------------------------------------------------------- #
def _toy_base_reward_energy(position: int, base_index: int):
    """Lower energy the more batch members have `base_index` at `position`."""

    def energy_fn(batch):
        arr = batch if isinstance(batch, np.ndarray) else batch.numpy()
        return torch.from_numpy(-arr[:, base_index, position].astype(np.float64))

    return energy_fn


def _uniform_context(base: str, length: int = CONTEXT) -> np.ndarray:
    return one_hot(base * length)


def test_ism_greedy_converges_and_respects_editable_bounds():
    window = Window(chrom="chr1", pred_start=0, pred_end=PRED_BP, ctx_start=0, ctx_end=CONTEXT)
    seed = Seed(
        name="c1",
        chrom="chr1",
        cand_start=30,
        cand_end=40,
        window=window,
        fold_label=None,
        editable=slice(30, 40),
        bins=slice(0, 1),
    )
    context = _uniform_context("C")
    energy_fn = _toy_base_reward_energy(position=35, base_index=0)  # reward 'A' at position 35

    state = ism_greedy(energy_fn, seed, context, rounds=5, top_k=1, batch_size=6)

    assert state.history[0]["round"] == 0
    assert state.history[0]["energy"] == pytest.approx(0.0)  # no 'A' anywhere initially
    assert decode(state.context[:, 35:36]) == "A"
    assert state.energy == pytest.approx(-1.0)
    # Untouched outside the editable window.
    assert decode(state.context[:, :30]) == "C" * 30
    assert decode(state.context[:, 40:]) == "C" * (CONTEXT - 40)


# --------------------------------------------------------------------------- #
# 6. adalead                                                                 #
# --------------------------------------------------------------------------- #
def _seed_and_context(editable_width: int = 10):
    window = Window(chrom="chr1", pred_start=0, pred_end=PRED_BP, ctx_start=0, ctx_end=CONTEXT)
    seed = Seed(
        name="c1",
        chrom="chr1",
        cand_start=30,
        cand_end=30 + editable_width,
        window=window,
        fold_label=None,
        editable=slice(30, 30 + editable_width),
        bins=slice(0, 1),
    )
    context = _uniform_context("A")  # target base already present everywhere
    return seed, context


def test_recombine_population_preserves_length_and_actually_mixes():
    seed, context = _seed_and_context(editable_width=8)
    adalead_search = AdaLead(
        energy_fn=lambda x: torch.zeros(x.shape[0]),
        seed=seed,
        context=context,
        config=AdaLeadConfig(population_size=2),
        rng=np.random.default_rng(0),
    )
    a = one_hot("AAAAAAAA")
    b = one_hot("CCCCCCCC")

    unchanged = adalead_search._recombine_population([a, b], recomb_rate=0.0)
    assert decode(unchanged[0]) == "AAAAAAAA"
    assert decode(unchanged[1]) == "CCCCCCCC"
    for child in unchanged:
        assert child.shape[1] == 8

    mixed = adalead_search._recombine_population([a, b], recomb_rate=1.0)
    for child in mixed:
        assert child.shape[1] == 8
    # With population_size > 1, recombination actually runs (the skeleton's hardcoded
    # batch_size=1 made this branch a no-op); at recomb_rate=1 the children differ from
    # either pure parent.
    assert {decode(c) for c in mixed} != {"AAAAAAAA", "CCCCCCCC"}


def test_adalead_seeded_from_endogenous_sequence_and_correct_length():
    seed, context = _seed_and_context(editable_width=8)
    energy_fn = _toy_base_reward_energy(position=34, base_index=0)  # 'A' already present
    config = AdaLeadConfig(rounds=0, population_size=4, mu=1.0, model_queries_per_batch=20)

    state = adalead(energy_fn, seed, context, config, rng=np.random.default_rng(0))

    assert state.editable == seed.editable
    assert state.context[:, state.editable].shape[1] == 8
    # The endogenous insert is already optimal (all 'A'); with rounds=0 the best candidate
    # in the initial population must be the endogenous sequence itself.
    assert state.energy == pytest.approx(-1.0)
    assert decode(state.context[:, seed.editable]) == "A" * 8


# --------------------------------------------------------------------------- #
# 7. FoldEnsemble                                                            #
# --------------------------------------------------------------------------- #
class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(4, 8, 1)
        self.feature_dim = 8

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(input_ids)

    def iter_named_blocks(self):
        yield "proj", self.proj


def _make_checkpoint(tmp_path, name, *, n_pred_bins=N_PRED_BINS):
    from regulonado.model import RegulonadoConfig as ModelConfig
    from regulonado.model import RegulonadoModel, TransferMLPPerturbHead

    config = ModelConfig(
        backbone_type="tiny",
        head_type="transfer_mlp",
        head_hidden=4,
        mlp_hidden=4,
        feature_dim=8,
        n_tracks=2,
        context_length=CONTEXT,
        n_pred_bins=n_pred_bins,
        bin_size=BIN_SIZE,
        track_names=["alpha", "beta"],
    )
    model = RegulonadoModel(
        config, backbone=_TinyBackbone(), head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2)
    )
    out_dir = tmp_path / name
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def test_fold_ensemble_mismatched_geometry_raises(tmp_path, monkeypatch):
    from regulonado.model import adapters

    monkeypatch.setattr(adapters, "build_backbone_architecture", lambda *a, **k: _TinyBackbone())
    good = _make_checkpoint(tmp_path, "fold_a")
    bad = _make_checkpoint(tmp_path, "fold_b", n_pred_bins=N_PRED_BINS + 1)

    with pytest.raises(ValueError, match="fold_b"):
        FoldEnsemble(
            [FoldSpec(good, name="fold_a"), FoldSpec(bad, name="fold_b")], device="cpu"
        )


def test_fold_ensemble_sequential_and_resident_agree(tmp_path, monkeypatch):
    from regulonado.model import adapters

    monkeypatch.setattr(adapters, "build_backbone_architecture", lambda *a, **k: _TinyBackbone())
    folds = [
        FoldSpec(_make_checkpoint(tmp_path, "fold_a"), name="fold_a"),
        FoldSpec(_make_checkpoint(tmp_path, "fold_b"), name="fold_b"),
    ]

    resident = FoldEnsemble(folds, device="cpu", mode="resident")
    sequential = FoldEnsemble(folds, device="cpu", mode="sequential")

    rng = np.random.default_rng(0)
    bases = rng.integers(0, 4, size=(2, CONTEXT))
    batch = np.zeros((2, 4, CONTEXT), dtype=np.float32)
    for i in range(2):
        batch[i, bases[i], np.arange(CONTEXT)] = 1

    assert torch.allclose(resident.predict(batch), sequential.predict(batch))


# --------------------------------------------------------------------------- #
# 8. DesignConfig                                                            #
# --------------------------------------------------------------------------- #
def test_design_config_parses_example_workflow_config():
    from regulonado.config.models import RegulonadoConfig as WorkflowConfig

    path = Path(__file__).parents[1] / "examples" / "enhancer_design.yaml"
    config = WorkflowConfig.from_yaml(path)
    assert config.design is not None
    assert {t.name for t in config.design.targets} == {"target_ism", "target_adalead"}
    assert config.design.holdout_run == "fold_3"
    assert config.design.design_runs == ["fold_0", "fold_1", "fold_2"]


def test_design_config_unknown_holdout_run_raises():
    import pydantic
    from regulonado.config.models import (
        BuildConfig,
        DesignConfig,
        DesignTarget,
        InputsConfig,
        RecompressConfig,
        ScalingConfig,
        TrainConfig,
        TrainPhase,
        TrainRun,
    )
    from regulonado.config.models import RegulonadoConfig as WorkflowConfig

    with pytest.raises(pydantic.ValidationError, match="holdout_run"):
        WorkflowConfig(
            results_dir="results",
            inputs=InputsConfig(intervals="i.bed", fasta="g.fa", bigwig_dir="bw"),
            build=BuildConfig(),
            recompress=RecompressConfig(),
            scaling=ScalingConfig(),
            train=TrainConfig(
                phases=[TrainPhase(name="head", preset="head_only")],
                runs=[TrainRun(name="fold_0", seed=0, pretrained_model="model/a")],
            ),
            design=DesignConfig(
                candidates="c.bed",
                holdout_run="fold_9",
                targets=[DesignTarget(name="t1", target="K562")],
            ),
        )


def test_design_config_duplicate_target_names_raise():
    import pydantic
    from regulonado.config.models import DesignConfig, DesignTarget

    with pytest.raises(pydantic.ValidationError, match="unique"):
        DesignConfig(
            candidates="c.bed",
            targets=[
                DesignTarget(name="a", target="K562"),
                DesignTarget(name="a", target="HepG2"),
            ],
        )


def test_design_config_shards_must_be_positive():
    import pydantic
    from regulonado.config.models import DesignConfig, DesignTarget

    with pytest.raises(pydantic.ValidationError):
        DesignConfig(
            candidates="c.bed", shards=0, targets=[DesignTarget(name="a", target="K562")]
        )
