"""Keep W&B uploads restricted to the explicitly approved metric payload."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("wandb")
MODULE = runpy.run_path(str(Path(__file__).parents[1] / "experiments/hl60-full/track_wandb.py"))


def test_config_whitelist_excludes_private_paths_and_conditions(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                device="mps",
                core_length=200,
                split_counts={"train": 8191},
                reference="private-reference",
                checkpoint="private-checkpoint",
            )
        )
    )
    training = dict(
        width=128,
        layers=4,
        heads=4,
        batch_size=64,
        epochs=60,
        learning_rate=0.0003,
        condition_dropout=0.15,
        early_stopping_patience=8,
        early_stopping_min_delta=0.001,
        seed=17,
        activity_columns=["private-label"],
        private_setting="private-setting",
    )
    (tmp_path / "training.yaml").write_text(yaml.safe_dump(training))
    config = MODULE["safe_config"](tmp_path)
    assert "private" not in json.dumps(config)
    assert config["device"] == "mps"
    assert config["condition_dimensions"] == 1


def test_sync_sends_only_aggregate_scalar_values(tmp_path):
    class Run:
        def __init__(self):
            self.summary = {}
            self.logs = []

        def log(self, values):
            self.logs.append(values)

    path = tmp_path / "designs/ism"
    path.mkdir(parents=True)
    pd.DataFrame(
        dict(
            sequence=["PRIVATE_SEQUENCE"],
            chrom=["PRIVATE_CHROM"],
            start=[123],
            target_gain=[1.0],
            offtarget_boost=[0.2],
            n_edits=[3],
        )
    ).to_csv(path / "designs.tsv", sep="\t", index=False)
    run = Run()
    MODULE["sync"](run, tmp_path, {}, set())
    assert run.logs
    assert all(isinstance(value, (int, float)) for row in run.logs for value in row.values())
    assert "PRIVATE" not in json.dumps(run.logs)
    assert run.summary["design/ism/completed"] == 1
