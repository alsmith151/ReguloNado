from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# attribute: --params -> AttributionConfig, CLI-flag override precedence      #
# (F03/S5: cli/attribute.py delegates to design/attribute_run.run_attribution #
# so run_attribution is monkeypatched rather than actually running an ISM     #
# sweep — the sweep itself is covered by tests/test_attribution.py.)          #
# --------------------------------------------------------------------------- #
def _attribute_args(tmp_path, **extra):
    args = [
        "attribute",
        "--candidates",
        str(tmp_path / "candidates.bed"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--fasta",
        str(tmp_path / "genome.fa"),
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--out",
        str(tmp_path / "out"),
    ]
    for flag, value in extra.items():
        args.extend([f"--{flag.replace('_', '-')}", str(value)])
    return args


def _patch_run_attribution(monkeypatch, captured):
    import regulonado.design.attribute_run as attribute_run

    def fake_run_attribution(config):
        captured["config"] = config
        return attribute_run.AttributionResult(
            out_dir=Path("out"),
            core_regions_bed=Path("out/core_regions.bed"),
            n_candidates=2,
            n_cores_called=1,
        )

    monkeypatch.setattr(attribute_run, "run_attribution", fake_run_attribution)


def test_attribute_params_round_trip(tmp_path, monkeypatch):
    """Values set only in --params (not passed as CLI flags) reach AttributionConfig."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("fix_width: 250\nquantile: 0.8\n")
    captured: dict = {}
    _patch_run_attribution(monkeypatch, captured)

    result = runner.invoke(app, _attribute_args(tmp_path, params=params_file, track="atac_hl60"))

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.fix_width == 250
    assert config.quantile == 0.8
    assert config.candidates == str(tmp_path / "candidates.bed")
    assert config.targets[0].track == "atac_hl60"


def test_attribute_cli_flags_override_params(tmp_path, monkeypatch):
    """A value set both in --params and as a CLI flag takes the CLI flag's value."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "candidates: from_params.bed\ntargets:\n  - {name: cli, track: from_params_track}\n"
    )
    captured: dict = {}
    _patch_run_attribution(monkeypatch, captured)

    result = runner.invoke(
        app, _attribute_args(tmp_path, params=params_file, track="from_cli_track")
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.candidates == str(tmp_path / "candidates.bed")  # CLI flag, not from_params.bed
    assert config.targets[0].track == "from_cli_track"


def test_attribute_rejects_an_invalid_option_combination(tmp_path):
    """A mode-specific AttributionConfig validator failure surfaces as a CLI error."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("topk_bins: 5\n")  # bin_reduction stays the 'mean' default

    result = runner.invoke(app, _attribute_args(tmp_path, params=params_file, track="atac_hl60"))

    assert result.exit_code != 0
    assert "topk_bins only applies" in result.output


# --------------------------------------------------------------------------- #
# design: --params -> DesignConfig, CLI-flag override precedence              #
# (F03/S5: cli/design.py delegates to design/run.run_design, so run_design is #
# monkeypatched rather than actually running a search -- the search itself is #
# covered by tests/test_design.py.)                                          #
# --------------------------------------------------------------------------- #
def _design_args(tmp_path, **extra):
    args = [
        "design",
        "--candidates",
        str(tmp_path / "candidates.bed"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--fasta",
        str(tmp_path / "genome.fa"),
        "--intervals",
        str(tmp_path / "intervals.bed"),
        "--out",
        str(tmp_path / "out"),
    ]
    for flag, value in extra.items():
        args.extend([f"--{flag.replace('_', '-')}", str(value)])
    return args


def _patch_run_design(monkeypatch, captured):
    import regulonado.design.run as design_run

    def fake_run_design(config):
        captured["config"] = config
        return design_run.DesignResult(
            out_dir=Path("out"),
            target=config.targets[0].target,
            method=config.targets[0].method,
            n_candidates=2,
        )

    monkeypatch.setattr(design_run, "run_design", fake_run_design)


def test_design_params_round_trip(tmp_path, monkeypatch):
    """Values set only in --params (not passed as CLI flags) reach DesignConfig."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("rounds: 7\nobjective: selective-activation\n")
    captured: dict = {}
    _patch_run_design(monkeypatch, captured)

    result = runner.invoke(app, _design_args(tmp_path, params=params_file, target="K562"))

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.rounds == 7
    assert config.objective == "selective-activation"
    assert config.candidates == str(tmp_path / "candidates.bed")
    assert config.targets[0].target == "K562"


def test_design_cli_flags_override_params(tmp_path, monkeypatch):
    """A value set both in --params and as a CLI flag takes the CLI flag's value."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "candidates: from_params.bed\ntargets:\n  - {name: cli, target: from_params_target}\n"
    )
    captured: dict = {}
    _patch_run_design(monkeypatch, captured)

    result = runner.invoke(
        app, _design_args(tmp_path, params=params_file, target="from_cli_target")
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.candidates == str(tmp_path / "candidates.bed")  # CLI flag, not from_params.bed
    assert config.targets[0].target == "from_cli_target"


def test_design_rejects_adalead_only_setting_with_cli_method_ism(tmp_path):
    """--method overrides --params' method, and the mismatch is then caught for the new method."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "targets:\n  - {name: cli, target: K562, method: adalead}\npopulation_size: 50\n"
    )

    result = runner.invoke(
        app, _design_args(tmp_path, params=params_file, target="K562", method="ism")
    )

    assert result.exit_code != 0
    assert "AdaLead-only" in result.output


def test_root_quiet_suppresses_info_logging():
    """``--quiet`` on the root command raises the 'regulonado' logger to WARNING."""
    result = runner.invoke(app, ["--quiet", "tracks", "--help"])

    assert result.exit_code == 0, result.output
    package_logger = logging.getLogger("regulonado")
    assert not package_logger.isEnabledFor(logging.INFO)
    assert package_logger.isEnabledFor(logging.WARNING)


def test_root_verbose_enables_debug_logging():
    """``-v``/``--verbose`` on the root command lowers the 'regulonado' logger to DEBUG."""
    result = runner.invoke(app, ["-v", "tracks", "--help"])

    assert result.exit_code == 0, result.output
    assert logging.getLogger("regulonado").isEnabledFor(logging.DEBUG)


def test_importing_dataset_build_installs_no_handlers():
    """A library module must never install its own logging handler.

    Run in a fresh subprocess so a handler installed by an earlier CLI
    invocation (in this test process) can't hide a regression here.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, regulonado.dataset.build\n"
            "assert logging.getLogger('regulonado').handlers == [], "
            "logging.getLogger('regulonado').handlers",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
