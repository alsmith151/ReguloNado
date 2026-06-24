from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from regulonado.model import RegulonadoConfig, RegulonadoModel, TransferMLPPerturbHead
from regulonado.predict import (
    RegionPredictionConfig,
    RegionPredictor,
    Window,
    _model_track_metadata,
    _resolve_tracks,
    _unique_track_names,
    collapse_bins,
    iter_windows,
    one_hot_context,
    read_chrom_sizes,
)

# Small geometry for fast tests: 4 bins of 10 bp => 40 bp prediction window,
# context 100 bp.
N_PRED_BINS = 4
BIN_SIZE = 10
PRED_BP = N_PRED_BINS * BIN_SIZE
CONTEXT = 100


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(4, 8, 1)
        self.feature_dim = 8

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(input_ids)

    def iter_named_blocks(self):
        yield "proj", self.proj


class DummyPredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = RegulonadoConfig(
            n_tracks=2,
            context_length=CONTEXT,
            n_pred_bins=N_PRED_BINS,
            bin_size=BIN_SIZE,
            track_names=["alpha", "beta"],
        )

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        values = torch.arange(
            batch * 2 * N_PRED_BINS,
            dtype=torch.float32,
            device=input_ids.device,
        )
        return values.reshape(batch, 2, N_PRED_BINS) * self.weight


# --------------------------------------------------------------------------- #
# collapse_bins                                                               #
# --------------------------------------------------------------------------- #
def test_collapse_merges_equal_bins():
    values = np.array([1.0, 1.0, 1.0, 1.0])
    intervals = collapse_bins(values, "chr1", 100, BIN_SIZE, rtol=0.0, chrom_length=1000)
    assert intervals == [("chr1", 100, 140, 1.0)]


def test_collapse_zero_region_merges_via_relative_tolerance():
    # Relative tolerance with both values 0 -> always merge (flat zero regions collapse).
    values = np.zeros(4)
    intervals = collapse_bins(values, "chr1", 0, BIN_SIZE, rtol=0.01, chrom_length=1000)
    assert intervals == [("chr1", 0, 40, 0.0)]


def test_collapse_splits_on_distinct_values():
    values = np.array([1.0, 1.0, 5.0, 5.0])
    intervals = collapse_bins(values, "chr1", 0, BIN_SIZE, rtol=0.01, chrom_length=1000)
    assert intervals == [("chr1", 0, 20, 1.0), ("chr1", 20, 40, 5.0)]


def test_collapse_relative_tolerance_merges_close_values():
    # 100 and 100.5 differ by 0.5%, within rtol=0.01 -> merged to their mean.
    values = np.array([100.0, 100.5])
    intervals = collapse_bins(values, "chr1", 0, BIN_SIZE, rtol=0.01, chrom_length=1000)
    assert len(intervals) == 1
    chrom, start, end, value = intervals[0]
    assert (chrom, start, end) == ("chr1", 0, 20)
    assert value == pytest.approx(100.25)


def test_collapse_clamps_to_chromosome_length():
    # chrom_length 25 truncates the window; bins fully past the end are dropped.
    values = np.array([1.0, 1.0, 1.0, 1.0])
    intervals = collapse_bins(values, "chr1", 0, BIN_SIZE, rtol=0.0, chrom_length=25)
    assert intervals == [("chr1", 0, 25, 1.0)]


# --------------------------------------------------------------------------- #
# iter_windows                                                                #
# --------------------------------------------------------------------------- #
def _chrom_sizes() -> dict[str, int]:
    return {"chr1": 1000, "chr2": 500}


def test_iter_windows_targeted_centers_on_bed(tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t490\t510\n")  # center 500
    windows = iter_windows(
        chrom_sizes=_chrom_sizes(),
        context_length=CONTEXT,
        n_pred_bins=N_PRED_BINS,
        bin_size=BIN_SIZE,
        bed_path=bed,
    )
    assert len(windows) == 1
    w = windows[0]
    assert (w.pred_start, w.pred_end) == (500 - PRED_BP // 2, 500 + PRED_BP // 2)
    assert (w.ctx_start, w.ctx_end) == (500 - CONTEXT // 2, 500 + CONTEXT // 2)


def test_iter_windows_targeted_rejects_overlap(tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t500\t500\nchr1\t510\t510\n")  # centers 20 bp apart < 40 bp window
    with pytest.raises(ValueError, match="Overlapping"):
        iter_windows(
            chrom_sizes=_chrom_sizes(),
            context_length=CONTEXT,
            n_pred_bins=N_PRED_BINS,
            bin_size=BIN_SIZE,
            bed_path=bed,
        )


def test_iter_windows_whole_genome_tiles_and_sorts():
    windows = iter_windows(
        chrom_sizes=_chrom_sizes(),
        context_length=CONTEXT,
        n_pred_bins=N_PRED_BINS,
        bin_size=BIN_SIZE,
        whole_genome=True,
    )
    # chr1: ceil(1000/40)=25 windows, chr2: ceil(500/40)=13 -> 38 total, sorted.
    assert len(windows) == 25 + 13
    chr1 = [w for w in windows if w.chrom == "chr1"]
    assert chr1[0].pred_start == 0
    assert chr1[1].pred_start == PRED_BP
    assert [w.pred_start for w in chr1] == sorted(w.pred_start for w in chr1)


def test_iter_windows_whole_genome_only_tiles_chrom_sizes_keys():
    # Only chromosomes present in chrom_sizes are tiled (chrom.sizes drives the selection).
    windows = iter_windows(
        chrom_sizes={"chr2": 500},
        context_length=CONTEXT,
        n_pred_bins=N_PRED_BINS,
        bin_size=BIN_SIZE,
        whole_genome=True,
    )
    assert {w.chrom for w in windows} == {"chr2"}
    assert len(windows) == 13  # ceil(500/40)


def test_read_chrom_sizes_parses_and_orders(tmp_path):
    path = tmp_path / "genome.chrom.sizes"
    path.write_text("# comment\nchr1\t1000\nchr2 500\n\n")
    assert read_chrom_sizes(path) == {"chr1": 1000, "chr2": 500}
    assert list(read_chrom_sizes(path)) == ["chr1", "chr2"]


def test_iter_windows_requires_exactly_one_mode():
    with pytest.raises(ValueError, match="exactly one"):
        iter_windows(
            chrom_sizes=_chrom_sizes(),
            context_length=CONTEXT,
            n_pred_bins=N_PRED_BINS,
            bin_size=BIN_SIZE,
        )


# --------------------------------------------------------------------------- #
# one_hot_context                                                             #
# --------------------------------------------------------------------------- #
def test_one_hot_context_encodes_and_pads(tmp_path):
    pyfaidx = pytest.importorskip("pyfaidx")
    fa_path = tmp_path / "g.fa"
    fa_path.write_text(">chr1\n" + "ACGT" * 25 + "\n")  # 100 bp
    fasta = pyfaidx.Fasta(str(fa_path), as_raw=True)

    # Context that runs 10 bp off the left end -> first 10 columns zero-padded.
    window = Window("chr1", pred_start=0, pred_end=PRED_BP, ctx_start=-10, ctx_end=-10 + CONTEXT)
    out = one_hot_context(fasta, window, CONTEXT, chrom_length=100)
    assert out.shape == (4, CONTEXT)
    assert out.dtype == np.int8
    assert out[:, :10].sum() == 0  # padded region
    # Column 10 corresponds to genomic position 0 == 'A' (row 0).
    assert out[0, 10] == 1
    assert out[:, 10].sum() == 1


# --------------------------------------------------------------------------- #
# track name + selection helpers                                             #
# --------------------------------------------------------------------------- #
def test_unique_track_names_strips_extensions_and_dedupes():
    records = [
        {"path": "/a/sample.bigWig"},
        {"path": "/b/sample.bigWig"},
        {"path": "/c/other.bw"},
    ]
    assert _unique_track_names(records) == ["sample", "sample_1", "other"]


def test_resolve_tracks_by_name_and_index():
    names = ["alpha", "beta", "gamma"]
    assert _resolve_tracks(None, names) == [0, 1, 2]
    assert _resolve_tracks(["beta", "0"], names) == [1, 0]
    with pytest.raises(ValueError, match="Unknown track"):
        _resolve_tracks(["missing"], names)


def test_load_model_for_inference_prefers_hf_model_dir(tmp_path, monkeypatch):
    from regulonado.model import adapters
    from regulonado.predict import load_model_for_inference

    config = RegulonadoConfig(
        backbone_type="tiny",
        head_type="transfer_mlp",
        head_hidden=4,
        mlp_hidden=4,
        feature_dim=8,
        n_tracks=2,
        context_length=CONTEXT,
        n_pred_bins=N_PRED_BINS,
        bin_size=BIN_SIZE,
        track_names=["alpha", "beta"],
    )
    model = RegulonadoModel(
        config,
        backbone=TinyBackbone(),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    model.save_pretrained(tmp_path, safe_serialization=True)

    monkeypatch.setattr(adapters, "build_backbone_architecture", lambda *_: TinyBackbone())
    loaded = load_model_for_inference(tmp_path, device="cpu")

    assert isinstance(loaded, RegulonadoModel)
    assert loaded.config.track_names == ["alpha", "beta"]
    assert loaded.config.context_length == CONTEXT
    assert loaded(torch.randn(1, 4, CONTEXT)).shape == (1, 2, CONTEXT)


def test_load_model_for_inference_legacy_run_root_returns_regulonado_model(
    tmp_path, monkeypatch
):
    import regulonado.model as model_module
    from regulonado.predict import load_model_for_inference

    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "regulonado_metadata.json").write_text(
        json.dumps(
            {
                "context_length": CONTEXT,
                "n_pred_bins": N_PRED_BINS,
                "bin_size": BIN_SIZE,
                "final_track_records": [
                    {"track_index": 0, "path": "/x/alpha.bw"},
                    {"track_index": 1, "path": "/x/beta.bw"},
                ],
            }
        )
    )
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (run_root / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": {"path": str(data_dir)},
                "backbone": {"name": "tiny", "config_overrides": {}},
                "model": {
                    "use_track_metadata": False,
                    "activation_type": "softplus",
                },
                "head": {
                    "type": "transfer_mlp",
                    "hidden": 4,
                    "dropout": 0.0,
                    "mlp_hidden": None,
                },
            }
        )
    )
    model = RegulonadoModel(
        RegulonadoConfig(
            backbone_type="tiny",
            head_type="transfer_mlp",
            head_hidden=4,
            mlp_hidden=4,
            feature_dim=8,
            n_tracks=2,
        ),
        backbone=TinyBackbone(),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    model.save_pretrained(checkpoint, safe_serialization=True)

    monkeypatch.setattr(model_module, "build_backbone_adapter", lambda _: TinyBackbone())
    loaded = load_model_for_inference(run_root, device="cpu")

    assert isinstance(loaded, RegulonadoModel)
    assert loaded.config.track_names == ["alpha", "beta"]
    assert loaded.config.context_length == CONTEXT


def test_model_track_metadata_from_config_uses_prediction_tensors():
    model = RegulonadoModel(
        RegulonadoConfig(
            track_metadata={
                "track_condition_ids": [0, 1],
                "track_timepoint_minutes": [None, 30.0],
            }
        ),
        backbone=TinyBackbone(),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )

    metadata = _model_track_metadata(model, "cpu")

    assert metadata["track_condition_ids"].dtype == torch.long
    assert metadata["track_condition_ids"].tolist() == [0, 1]
    assert metadata["track_timepoint_minutes"].dtype == torch.float32
    assert torch.isnan(metadata["track_timepoint_minutes"][0])
    assert metadata["track_timepoint_minutes"][1].item() == pytest.approx(30.0)


def test_quick_predictor_predicts_from_configured_paths(tmp_path, monkeypatch):
    pytest.importorskip("pyfaidx")
    import regulonado.predict as predict_module

    fa_path = tmp_path / "g.fa"
    fa_path.write_text(">chr1\n" + "ACGT" * 100 + "\n")
    monkeypatch.setattr(
        predict_module,
        "load_model_for_inference",
        lambda checkpoint_dir, dataset_dir=None, device=None: DummyPredictModel(),
    )

    predictor = RegionPredictor(
        RegionPredictionConfig(
            checkpoint_dir=tmp_path / "checkpoint",
            fasta_path=fa_path,
            tracks=["beta"],
            device="cpu",
        )
    )
    result = predictor("chr1", 190, 210)

    assert result.chrom == "chr1"
    assert (result.query_start, result.query_end) == (190, 210)
    assert (result.pred_start, result.pred_end) == (180, 220)
    assert result.track_names == ["beta"]
    assert result.values.tolist() == [[4.0, 5.0, 6.0, 7.0]]
    assert result.bin_starts.tolist() == [180, 190, 200, 210]
    assert result.bin_ends.tolist() == [190, 200, 210, 220]
    assert result.as_records()[0] == {
        "chrom": "chr1",
        "start": 180,
        "end": 190,
        "track": "beta",
        "value": 4.0,
    }


def test_quick_predictor_allows_per_call_tracks_and_validates_coordinates(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyfaidx")
    import regulonado.predict as predict_module

    fa_path = tmp_path / "g.fa"
    fa_path.write_text(">chr1\n" + "ACGT" * 100 + "\n")
    monkeypatch.setattr(
        predict_module,
        "load_model_for_inference",
        lambda checkpoint_dir, dataset_dir=None, device=None: DummyPredictModel(),
    )

    predictor = RegionPredictor.from_paths(tmp_path / "checkpoint", fa_path, device="cpu")

    result = predictor.predict("chr1", 200, 200, tracks=["alpha"])
    assert result.track_names == ["alpha"]
    assert result.values.tolist() == [[0.0, 1.0, 2.0, 3.0]]

    with pytest.raises(ValueError, match="0 <= start <= end"):
        predictor.predict("chr1", 10, 9)
    with pytest.raises(ValueError, match="not present"):
        predictor.predict("chrX", 10, 10)


def test_find_config_and_weights_hf_layout(tmp_path):
    """Run root has resolved_config.json; weights live in checkpoint-NNNN/ subdirs."""
    from regulonado.predict import _find_config, _find_weights

    run_root = tmp_path / "run"
    ckpt = run_root / "checkpoint-1000"
    ckpt.mkdir(parents=True)
    config_file = run_root / "resolved_config.json"
    config_file.write_text("{}")
    weights_file = ckpt / "model.safetensors"
    weights_file.write_bytes(b"")

    # Passing the run root: config found locally, weights found in checkpoint subdir.
    assert _find_config(run_root) == config_file
    assert _find_weights(run_root) == weights_file

    # Passing the checkpoint subdir: config found in parent, weights found locally.
    assert _find_config(ckpt) == config_file
    assert _find_weights(ckpt) == weights_file


def test_find_weights_picks_best_checkpoint_from_trainer_state(tmp_path):
    import json as _json

    from regulonado.predict import _find_weights

    run_root = tmp_path / "run"
    for step in (500, 1000, 2000):
        d = run_root / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"")

    # trainer_state.json in the latest checkpoint says 1000 was best.
    best_dir = run_root / "checkpoint-1000"
    (run_root / "checkpoint-2000" / "trainer_state.json").write_text(
        _json.dumps({"best_model_checkpoint": str(best_dir)})
    )

    assert _find_weights(run_root) == best_dir / "model.safetensors"


def test_find_weights_falls_back_to_latest_when_no_trainer_state(tmp_path):
    from regulonado.predict import _find_weights

    run_root = tmp_path / "run"
    for step in (500, 1000, 2000):
        d = run_root / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"")

    assert _find_weights(run_root) == run_root / "checkpoint-2000" / "model.safetensors"


# --------------------------------------------------------------------------- #
# end-to-end write + read                                                     #
# --------------------------------------------------------------------------- #
def test_write_and_read_bigwig_roundtrip(tmp_path):
    pybigtools = pytest.importorskip("pybigtools")
    from regulonado.predict import _write_bigwigs

    chrom_sizes = {"chr1": 1000}
    values = np.concatenate([np.zeros(50), np.full(50, 3.0)])  # flat then raised
    intervals = collapse_bins(values, "chr1", 0, BIN_SIZE, rtol=0.0, chrom_length=1000)
    written = _write_bigwigs(tmp_path, ["track0"], [0], {0: intervals}, chrom_sizes)

    assert len(written) == 1 and written[0].exists()
    bw = pybigtools.open(str(written[0]))
    vals = np.asarray(bw.values("chr1", 0, 1000), dtype=np.float32)
    assert vals[0] == pytest.approx(0.0)
    assert vals[600] == pytest.approx(3.0)
