from __future__ import annotations

import json

import pandas as pd
from regulonado.cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


def test_train_builds_a_readable_preset_command(tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            str(tmp_path / "dataset"),
            "--preset",
            "head_only",
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--set",
            "seed=3",
            "--set",
            "backbone.pretrained_name=johahi/flashzoi-replicate-3",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "+experiment=head_only" in result.output
    assert f"data.metadata_path={tmp_path / 'metadata.json'}" in result.output
    assert "seed=3" in result.output
    assert "backbone.pretrained_name=johahi/flashzoi-replicate-3" in result.output


def test_train_rejects_an_invalid_set_value(tmp_path):
    result = runner.invoke(
        app,
        ["train", str(tmp_path / "dataset"), "--set", "not-an-assignment", "--dry-run"],
    )

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_train_print_config_resolves_the_selected_model_and_seed(tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            str(tmp_path / "dataset"),
            "--preset",
            "head_only",
            "--set",
            "seed=3",
            "--set",
            "backbone.pretrained_name=johahi/flashzoi-replicate-3",
            "--print-config",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "seed: 3" in result.output
    assert "pretrained_name: johahi/flashzoi-replicate-3" in result.output


def test_train_print_config_rejects_wrong_parameter_types(tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            str(tmp_path / "dataset"),
            "--preset",
            "head_only",
            "--set",
            "trainer.batch_size=oops",
            "--print-config",
        ],
    )

    assert result.exit_code != 0
    assert "batch_size" in result.output


def test_enrich_metadata_writes_a_new_file(tmp_path):
    source = tmp_path / "regulonado_metadata.json"
    source.write_text(
        json.dumps(
            {
                "final_track_records": [
                    {"track_index": 0, "bigwig_path": "a.bw"},
                    {"track_index": 1, "bigwig_path": "b.bw"},
                ]
            }
        )
    )
    factors = tmp_path / "scale_factors.csv"
    pd.DataFrame(
        {
            "track_index": [0, 1],
            "scale_factor": [2.0, 3.0],
            "clip_soft": [4.0, 5.0],
            "clip_hard": [6.0, 7.0],
        }
    ).to_csv(factors, index=False)
    output = tmp_path / "enriched.json"

    result = runner.invoke(
        app,
        ["enrich-metadata", str(source), str(factors), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "scale_factor" not in json.loads(source.read_text())["final_track_records"][0]
    records = json.loads(output.read_text())["final_track_records"]
    assert records[0]["scale_factor"] == 2.0
    assert records[1]["clip_hard"] == 7.0


def test_enrich_metadata_refuses_in_place_output(tmp_path):
    source = tmp_path / "regulonado_metadata.json"
    source.write_text(json.dumps({"final_track_records": []}))
    factors = tmp_path / "scale_factors.csv"
    pd.DataFrame(
        columns=["track_index", "scale_factor", "clip_soft", "clip_hard"]
    ).to_csv(factors, index=False)

    result = runner.invoke(
        app,
        ["enrich-metadata", str(source), str(factors), "--output", str(source)],
    )

    assert result.exit_code != 0
    assert "must differ" in result.output
