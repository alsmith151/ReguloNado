"""Regression tests for the packaged multi-run Snakemake workflow."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).parents[1] / "python" / "regulonado" / "workflow" / "Snakefile"
)


def test_two_runs_form_independent_phase_chains(tmp_path):
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    intervals.touch()
    fasta.touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
build:
  context_length: 100
  bin_size: 10
  n_pred_bins: 4
  shift_max_bp: 0
  extract_threads: 1
  arrow_write_threads: 1
  arrow_batch_size: 4
  compression: lz4
  stage_to_scratch: false
  drop_missing: true
  dedupe_tracks: content
recompress:
  enabled: false
  zstd_level: 3
  max_batch_size: 4
  workers: 1
scaling:
  method: tmm
train:
  nproc_per_node: 1
  phases:
    - {{name: first, preset: head_only}}
    - {{name: second, preset: unfreeze_output}}
  runs:
    - {{name: run_a, seed: 10, pretrained_model: model/a}}
    - {{name: run_b, seed: 20, pretrained_model: model/b}}
"""
    )

    result = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(WORKFLOW),
            "--configfile",
            str(config),
            "--cores",
            "1",
            "--dry-run",
            "--printshellcmds",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )

    assert result.returncode == 0, result.stderr
    assert re.search(r"train_phase\s+4", result.stdout)
    assert "results/train/run_a/first/trainer_state.json" in result.stdout
    assert "results/train/run_a/second/trainer_state.json" in result.stdout
    assert "results/train/run_b/first/trainer_state.json" in result.stdout
    assert "results/train/run_b/second/trainer_state.json" in result.stdout
    assert "--metadata" in result.stdout
    assert "--set seed=10" in result.stdout
    assert "--set seed=20" in result.stdout
    assert "normalization original" in result.stdout
    assert "normalization tmm" in result.stdout
    assert "--scale-factors" in result.stdout
    assert "predict" not in result.stdout.lower()

    cli_result = subprocess.run(
        [sys.executable, "-m", "regulonado", "pipeline", str(config), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert cli_result.returncode == 0, cli_result.stdout + cli_result.stderr
    assert re.search(r"train_phase\s+4", cli_result.stdout + cli_result.stderr)

    bad_config = tmp_path / "bad-config.yaml"
    bad_config.write_text(
        config.read_text().replace(
            "train:\n  nproc_per_node:",
            "train:\n  common:\n    trainer:\n      made_up_setting: 1\n  nproc_per_node:",
        )
    )
    bad_result = subprocess.run(
        [sys.executable, "-m", "regulonado", "pipeline", str(bad_config), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert bad_result.returncode != 0
    assert "Invalid training settings" in bad_result.stdout + bad_result.stderr

    wrong_type_config = tmp_path / "wrong-type-config.yaml"
    wrong_type_config.write_text(
        config.read_text().replace(
            "train:\n  nproc_per_node:",
            "train:\n  common:\n    trainer:\n      batch_size: oops\n  nproc_per_node:",
        )
    )
    wrong_type_result = subprocess.run(
        [sys.executable, "-m", "regulonado", "pipeline", str(wrong_type_config), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert wrong_type_result.returncode != 0
    assert "batch_size" in wrong_type_result.stdout + wrong_type_result.stderr


def test_selective_activation_design_flags_reach_pipeline_command(tmp_path):
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    checkpoint_a = tmp_path / "checkpoint-a"
    checkpoint_b = tmp_path / "checkpoint-b"
    intervals.touch()
    fasta.touch()
    candidates.write_text("chr1\t10\t20\tcandidate-1\n")
    checkpoint_a.mkdir()
    checkpoint_b.mkdir()

    config = tmp_path / "design-config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
build:
  context_length: 100
  bin_size: 10
  n_pred_bins: 4
  shift_max_bp: 0
  extract_threads: 1
  arrow_write_threads: 1
  arrow_batch_size: 4
  compression: lz4
  stage_to_scratch: false
  drop_missing: true
  dedupe_tracks: content
recompress:
  enabled: false
  zstd_level: 3
  max_batch_size: 4
  workers: 1
scaling:
  method: tmm
train:
  nproc_per_node: 1
  phases:
    - {{name: head, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, pretrained_model: model/a}}
    - {{name: fold_1, seed: 20, pretrained_model: model/b}}
design:
  candidates: {candidates}
  checkpoint_dirs:
    - {checkpoint_a}
    - {checkpoint_b}
  common:
    objective: selective-activation
    gain_transform: log2-fold-change
    gain_pseudocount: 1.0
    offtarget_boost_weight: 1.5
    offtarget_boost_tolerance: 0.05
    offtarget_temperature: 0.4
  targets:
    - name: k562
      target: K562
      settings:
        offtarget_boost_weight: 2.5
"""
    )

    result = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(WORKFLOW),
            "--configfile",
            str(config),
            "--cores",
            "1",
            "--dry-run",
            "--printshellcmds",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "regulonado design" in output
    assert "--objective selective-activation" in output
    assert "--gain-transform log2-fold-change" in output
    assert "--gain-pseudocount 1.0" in output
    assert "--offtarget-boost-weight 2.5" in output
    assert "--offtarget-boost-tolerance 0.05" in output
    assert "--offtarget-temperature 0.4" in output
    assert str(tmp_path / "results" / "design" / "k562" / "designs.tsv") in output

    cli_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "regulonado",
            "pipeline",
            str(config),
            "design",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    cli_output = cli_result.stdout + cli_result.stderr
    assert cli_result.returncode == 0, cli_output
    assert "--objective selective-activation" in cli_output
    assert "--gain-transform log2-fold-change" in cli_output
