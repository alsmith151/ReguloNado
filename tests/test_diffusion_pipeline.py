import json

import numpy as np
import pandas as pd
import pytest
from regulonado.config.models import DiffusionStageConfig, RegulonadoConfig
from regulonado.design.diffusion_data import (
    choose_regions,
    chromosome_splits,
    prepare_diffusion_data,
)


def test_chromosome_leakage_rejected():
    with pytest.raises(ValueError, match="Chromosome leakage"):
        chromosome_splits(
            pd.DataFrame({"chrom": ["chr1", "chr1"], "split": ["train", "val"]}), [], []
        )


def test_all_regions_default_and_priority_retained():
    frame = pd.DataFrame({"priority": [True, False, False]})
    assert len(choose_regions(frame, None, 1)) == 3
    assert choose_regions(frame, 1, 1).priority.all()
    with pytest.raises(ValueError):
        DiffusionStageConfig(validation_chromosomes=["chr1"], test_chromosomes=["chr1"])


def test_prepare_augment_train_only(tmp_path):
    rng = np.random.default_rng(12)
    fasta = tmp_path / "ref.fa"
    fasta.write_text(
        "".join(
            ">" + c + "\n" + "".join(rng.choice(list("ACGT"), 500)) + "\n"
            for c in ["chr1", "chr8", "chr9"]
        )
    )
    source = tmp_path / "dataset"
    source.mkdir()
    pd.DataFrame(
        dict(
            chrom=["chr1", "chr8", "chr9"],
            start=[50] * 3,
            end=[450] * 3,
            split=["train", "val", "test"],
            ccre_class=[None, "dELS", "dELS"],
        )
    ).to_parquet(source / "regions.parquet")
    pd.DataFrame({"a": [10, 20, 30], "b": [5, 5, 5]}).to_parquet(source / "counts.parquet")
    pd.DataFrame(
        dict(track_name=["a", "b"], group=["HL-60", "K562"], log_size_factor=[0.0, 0.0])
    ).to_parquet(source / "tracks.parquet")
    bed = tmp_path / "priority.bed"
    bed.write_text("chr1\t50\t450\tx\t0\t.\t1\t2\nchr1\t50\t450\tx\t0\t.\t1\t4\n")
    config = DiffusionStageConfig(
        specificity_regions=str(bed),
        core_length=40,
        priority_views=4,
        priority_jitter_bp=50,
        priority_sampling_weight=7,
    )
    manifest = prepare_diffusion_data(source, fasta, tmp_path / "prepared", config)
    table = pd.read_parquet(tmp_path / "prepared/training.parquet")
    train = table.loc[table.split.eq("train")]
    val = table.loc[table.split.eq("validation")]
    assert 1 < len(train) <= 4
    assert len(val) == 1
    assert set(table.chrom) == {"chr1", "chr8"}
    assert train.sampling_weight.eq(7).all()
    assert train.target_specificity.eq(3).all()
    assert val.target_specificity.isna().all()
    assert train.core_start.between(180, 280).all()
    assert np.allclose(train.target_activity, np.log1p(10))
    assert manifest["chromosomes"]["test"] == ["chr9"]


def test_example_config_validates():
    from pathlib import Path

    import yaml

    config = RegulonadoConfig.model_validate(
        yaml.safe_load(Path("examples/diffusion_training.yaml").read_text())
    )
    assert config.diffusion.train_regions is None
    assert config.diffusion.ccre_classes == []


def test_resume_matches_uninterrupted_training(tmp_path):
    import torch
    from regulonado.config.diffusion import DiffusionConfig
    from regulonado.design.diffusion import train_diffusion

    table = tmp_path / "train.parquet"
    pd.DataFrame(
        dict(
            chrom=["chr1", "chr1", "chr8"],
            sequence=["ACGTACGT", "AAAAACCC", "TTTGGGCC"],
            activity=[1.0, 2.0, 3.0],
            sampling_weight=[1.0, 4.0, 1.0],
        )
    ).to_parquet(table)
    config = DiffusionConfig(
        activity_columns=["activity"],
        width=8,
        heads=2,
        layers=1,
        max_length=8,
        batch_size=2,
        epochs=2,
        device="cpu",
        validation_chromosomes=["chr8"],
        sampling_weight_column="sampling_weight",
        lr_schedule="plateau",
    )
    train_diffusion(table, tmp_path / "whole", config)
    train_diffusion(table, tmp_path / "resume", config.model_copy(update={"epochs": 1}))
    train_diffusion(table, tmp_path / "resume", config, resume=tmp_path / "resume/last.pt")
    a = torch.load(tmp_path / "whole/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert a["training_state"]["history"] == b["training_state"]["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


def test_workflow_diffusion_dryrun(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    source = tmp_path / "dataset"
    source.mkdir()
    for name in ["regions", "counts", "tracks"]:
        (source / f"{name}.parquet").touch()
    fasta = tmp_path / "ref.fa"
    fasta.touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                results_dir=str(tmp_path / "results"),
                inputs={"fasta": str(fasta)},
                qc={"checks": []},
                diffusion={
                    "dataset_dir": str(source),
                    "training": {"activity_columns": ["target_activity"], "device": "cuda"},
                },
            )
        )
    )
    workflow = Path(__file__).parents[1] / "python/regulonado/workflow/Snakefile"
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "snakemake"),
            "-s",
            str(workflow),
            "--configfile",
            str(config),
            "--cores",
            "1",
            "-n",
            "-p",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "diffusion_train" in result.stdout
    assert "generate fit" in result.stdout


def test_pipeline_wandb_payload_excludes_data_and_paths(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import regulonado.design.diffusion as diffusion
    from regulonado.design.diffusion_pipeline import fit_pipeline

    payload = {}

    class Run:
        id = "test-id"
        url = "https://wandb.invalid/test"
        summary = {}

        def log(self, values):
            payload["metrics"] = values

        def finish(self):
            payload["finished"] = True

    def init(**kwargs):
        payload["init"] = kwargs
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))

    def train(table, out, config, resume, on_epoch):
        on_epoch({"epoch": 1, "train_loss": 1.3, "validation_loss": 1.4})
        (out / "training_summary.json").write_text(json.dumps({"best_epoch": 1}))
        return out / "best.pt"

    monkeypatch.setattr(diffusion, "train_diffusion", train)
    params = tmp_path / "settings.json"
    params.write_text(
        json.dumps(
            {
                "wandb_project": "regulonado-design",
                "specificity_regions": "/private/sensitive.bed",
                "training": {"activity_columns": ["target_activity"]},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "chromosomes": {"validation": ["chr8"], "test": ["chr9"]},
                "split_counts": {"train": 100, "validation": 10},
            }
        )
    )
    fit_pipeline(params, tmp_path / "data.parquet", manifest, tmp_path / "model")
    safe = payload["init"]["config"]
    assert "encoder_checkpoint" not in safe
    assert "activity_columns" not in safe
    assert "/private" not in json.dumps(safe)
    assert "chr8" not in json.dumps(safe)
    assert payload["metrics"]["validation_loss"] == 1.4
    assert payload["finished"]
