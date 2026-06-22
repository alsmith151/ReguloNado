from __future__ import annotations

from regulonado.__main__ import app
from regulonado.experiments import (
    builtin_experiment_dir,
    discover_experiments,
    suggest_experiments,
)
from typer.testing import CliRunner

runner = CliRunner()


def test_discover_finds_builtin_experiments():
    exps = discover_experiments()
    # Representative built-ins that ship with the repo.
    assert "sharp_transfer_learning" in exps
    assert "condition_agnostic_borzoi" in exps
    info = exps["sharp_transfer_learning"]
    assert info.builtin is True
    assert info.path.parent == builtin_experiment_dir()
    assert info.summary.startswith("Sharp transfer learning")
    # The @package directive must not leak into the parsed summary.
    assert "@package" not in info.description


def test_suggest_experiments_handles_typo():
    assert "sharp_transfer_learning" in suggest_experiments("sharp_transfer")


def test_general_agnostic_configs_present_and_resolve():
    from regulonado.experiments import resolve_experiment

    exps = discover_experiments()
    for name in (
        "condition_agnostic_enformer",
        "condition_agnostic_full_finetune",
        "condition_agnostic_calibrated",
    ):
        assert name in exps, f"missing general config {name}"

    cfg = resolve_experiment("condition_agnostic_full_finetune")
    # Full fine-tune means the whole backbone is trainable.
    assert cfg["trainer"]["freeze_backbone"] is False
    assert resolve_experiment("condition_agnostic_enformer")["backbone"]["name"] == "enformer"


def test_moved_configs_listed_as_local_and_resolvable():
    from regulonado.experiments import resolve_experiment

    exps = discover_experiments()
    # The older magnitude_fix sweep was moved to scripts/experiment/, so it is
    # discovered but flagged non-builtin (listed as "(local)").
    name = "magnitude_fix_poisson_nll"
    assert name in exps
    assert exps[name].builtin is False
    # It must still resolve via the scripts/ search path.
    assert resolve_experiment(name)["loss"]["name"] == "poisson_nll"


def test_experiments_list_command_lists_all():
    result = runner.invoke(app, ["experiments"])
    assert result.exit_code == 0
    assert "sharp_transfer_learning" in result.stdout
    assert "condition_agnostic_borzoi" in result.stdout


def test_train_unknown_experiment_suggests_and_exits():
    result = runner.invoke(app, ["train", "ds/", "-e", "sharp_transfer"])
    assert result.exit_code == 1
    # Did-you-mean suggestion and the full listing are both surfaced.
    assert "Unknown experiment" in result.output
    assert "sharp_transfer_learning" in result.output


def test_train_local_dry_run_builds_command(tmp_path):
    result = runner.invoke(
        app,
        ["train", str(tmp_path), "-e", "condition_agnostic_borzoi", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "regulonado.train" in result.stdout
    assert "+experiment=condition_agnostic_borzoi" in result.stdout
    assert f"data.path={tmp_path}" in result.stdout


def test_train_slurm_dry_run_emits_sbatch(tmp_path):
    result = runner.invoke(
        app,
        ["train", str(tmp_path), "-e", "condition_agnostic_borzoi", "--slurm", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "sbatch" in result.stdout
    assert "train_slurm.sh" in result.stdout
    assert "EXPERIMENT=condition_agnostic_borzoi" in result.stdout
    assert f"DATA_DIR={tmp_path}" in result.stdout
    assert "REPO_DIR=" in result.stdout
