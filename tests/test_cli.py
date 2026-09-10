from __future__ import annotations

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


def test_tracks_assemble_joins_scale_factors_by_track_name(tmp_path):
    """'tracks assemble' replaces the deleted 'enrich-metadata' hand-rolled join."""
    from regulonado.tracks_table import read_track_table, write_track_table

    discovered = tmp_path / "discovered.parquet"
    write_track_table(
        pd.DataFrame(
            [
                {"track_name": "a", "status": "included", "track_index": 0},
                {"track_name": "b", "status": "included", "track_index": 1},
            ]
        ),
        discovered,
    )
    factors = tmp_path / "scale_factors.parquet"
    pd.DataFrame(
        {
            "track_index": [0, 1],
            "track_name": ["a", "b"],
            "scale_factor": [2.0, 3.0],
            "clip_soft": [4.0, 5.0],
            "clip_hard": [6.0, 7.0],
        }
    ).to_parquet(factors, index=False)
    output = tmp_path / "tracks.parquet"

    result = runner.invoke(
        app,
        ["tracks", "assemble", str(discovered), "-o", str(output), "--scale-factors", str(factors)],
    )

    assert result.exit_code == 0, result.output
    table = read_track_table(output)
    row_b = table.set_index("track_name").loc["b"]
    assert table.set_index("track_name").loc["a", "scale_factor"] == 2.0
    assert row_b["scale_clip_hard"] == 7.0
