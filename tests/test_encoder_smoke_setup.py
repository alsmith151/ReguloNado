"""Test the experiment's geometry/split preservation and numeric label alignment."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
experiment = runpy.run_path(str(ROOT / "experiments/alphagenome-encoder-smoke/run.py"))


def test_prepare_preserves_target_windows_and_original_splits(tmp_path):
    source = pd.DataFrame(
        dict(
            chrom=["chr1", "chr1", "chr10", "chr10", "chr9", "chr9"],
            start=[2500, 4500] * 3,
            end=[4600, 6600] * 3,
            target_start=[3000, 5000] * 3,
            target_end=[4000, 6000] * 3,
            split=["train", "train", "val", "val", "test", "test"],
            ccre_class=["dELS"] * 6,
        )
    )
    regions = tmp_path / "regions.parquet"
    source.to_parquet(regions)
    out = tmp_path / "run"
    selected = experiment["prepare_regions"](
        regions, out, training_count=2, validation_count=1, candidate_count=1
    )
    np.testing.assert_array_equal(selected.start, selected.target_start)
    assert selected.loc[selected.split.eq("validation"), "chrom"].tolist() == ["chr10"]
    assert selected.loc[selected.split.eq("test"), "chrom"].tolist() == ["chr9"]
    candidates = pd.read_csv(out / "candidates.bed", sep="\t", header=None)
    assert len(candidates) == 1
    assert int(candidates.iloc[0, 2] - candidates.iloc[0, 1]) == 200
    training = experiment["write_configs"](
        out, tmp_path / "oracle", validation_chromosomes=["chr10"]
    )
    assert training.validation_chromosomes == ["chr10"]
    assert yaml.safe_load((out / "gradient.yaml").read_text())["model_kind"] == "region_counts"


def test_measured_labels_follow_selected_original_rows(tmp_path, monkeypatch):
    from regulonado.design import region_predictor

    class Oracle:
        group_names = ["HL-60", "K-562"]
        context_length, crop_bp, bin_size, snap_bp = 4096, 1408, 1152, 128

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, batch):
            return torch.tensor([[[1.0], [2.0]]])

    monkeypatch.setattr(region_predictor, "RegionSequencePredictor", Oracle)
    regions = tmp_path / "regions.parquet"
    pd.DataFrame(
        dict(
            chrom=["chr1", "chr1", "chr10", "chr9"],
            start=[3000, 5000, 3000, 3000],
            end=[4000, 6000, 4000, 4000],
            split=["train", "train", "val", "test"],
            ccre_class=["dELS"] * 4,
        )
    ).to_parquet(regions)
    counts = pd.DataFrame(dict(t0=[2.0, 20.0, 200.0, 2000.0], t1=[3.0, 30.0, 300.0, 3000.0]))
    counts.to_parquet(tmp_path / "counts.parquet")
    checkpoint = tmp_path / "oracle"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            dict(
                track_names=["t0", "t1"],
                track_groups=[0, 1],
                log_size_factors=[np.log(2), 0.0],
                n_groups=2,
            )
        )
    )
    fasta = tmp_path / "hg38.fa"
    fasta.write_text(
        "".join(f">{chrom}\n" + "ACGT" * 2500 + "\n" for chrom in ["chr1", "chr10", "chr9"])
    )
    out = tmp_path / "run"
    selected = experiment["prepare_regions"](
        regions, out, training_count=2, validation_count=1, candidate_count=1
    )
    selected["HL60_specificity"] = [0.25, 1.25, 0.75, 2.0]
    activities = experiment["label_regions"](selected, fasta, checkpoint, out, "cpu", regions)
    measured = pd.read_csv(out / "measured_labels.tsv", sep="\t")
    expected = np.log1p(counts.iloc[selected.dataset_row].t0.to_numpy() / 2)
    np.testing.assert_allclose(measured["HL-60"], expected)
    training = pd.read_csv(out / "diffusion-training.tsv", sep="\t")
    assert set(training.split) == {"train", "validation"}
    expected_target = measured.loc[measured.HL60_specificity.eq(1.25), "HL-60"].iloc[0]
    np.testing.assert_allclose(activities["HL-60"], expected_target)
    assert activities["HL60_specificity"] == 1.25
    assert training.HL60_specificity.max() == 1.25  # Test score 2.0 never enters training.
    profiles = json.loads((out / "specificity_profiles.json").read_text())
    values = [profile["activities"]["HL60_specificity"] for profile in profiles.values()]
    assert values == sorted(values)
    config = experiment["write_configs"](
        out, checkpoint, activities=activities, validation_chromosomes=["chr10"]
    )
    assert "HL60_specificity" in config.activity_columns
    assert (out / "diffusion-q25.yaml").exists()
    oracle = pd.read_csv(out / "oracle_labels.tsv", sep="\t")
    assert (oracle["HL-60"] == 1.0).all()


def test_curated_candidates_match_original_intervals_and_preserve_scores(tmp_path):
    source = pd.DataFrame(
        dict(
            chrom=["chr1", "chr10", "chr9", "chr9"],
            start=[2500, 2500, 2500, 4500],
            end=[4600, 4600, 4600, 6600],
            target_start=[3000, 3000, 3000, 5000],
            target_end=[4000, 4000, 4000, 6000],
            orig_chrom=["chr1", "chr10", "chr9", "chr9"],
            orig_start=[3200, 3200, 3200, 5200],
            orig_end=[3800, 3800, 3800, 5800],
            split=["train", "val", "test", "test"],
            ccre_class=["dELS", "dELS", "dELS", None],
        )
    )
    path = tmp_path / "regions.parquet"
    source.to_parquet(path)
    priority = tmp_path / "priority.bed"
    priority.write_text(
        "chr9\t5200\t5800\tpriority\t.\t.\t2\t0.5\nchr9\t5200\t5800\tpriority\t.\t.\t2\t1.5\n"
    )
    out = tmp_path / "run"
    selected = experiment["prepare_regions"](
        path,
        out,
        training_count=1,
        validation_count=1,
        candidate_count=1,
        candidate_regions=priority,
    )
    candidate = selected.loc[selected.split.eq("test")].iloc[0]
    assert candidate.orig_start == 5200
    assert candidate.start == 5000
    audit = pd.read_csv(out / "priority_regions.tsv", sep="\t")
    assert len(audit) == 1
    observations = json.loads(audit.iloc[0].specificity_observations)
    assert [row["specificity"] for row in observations] == [0.5, 1.5]
    assert int(audit.iloc[0].source_rows) == 2
    assert audit.iloc[0].HL60_specificity == 1.0
