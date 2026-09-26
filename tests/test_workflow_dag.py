"""Regression tests for the packaged multi-run Snakemake workflow."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parents[1] / "python" / "regulonado" / "workflow" / "Snakefile"


def test_dataset_sentinel_is_readme_md(tmp_path):
    """The build_dataset rule outputs README.md as the completion sentinel."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    intervals.touch()
    fasta.touch()
    results = tmp_path / "results"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
scaling:
  method: tmm
"""
    )
    target_dir = results / "dataset"
    target = target_dir / "README.md"

    planned = subprocess.run(
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
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    planned_output = planned.stdout + planned.stderr
    assert planned.returncode == 0, planned_output
    assert "regulonado dataset" in planned_output
    assert "recompress" not in planned_output.lower()
    assert str(target_dir) in planned_output

    target_dir.mkdir(parents=True)
    target.touch()
    (target_dir / "tracks.parquet").touch()
    (target_dir / "data").mkdir()

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
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Nothing to be done" in output
    assert "Execute 1 jobs" not in output


def test_parameter_sweep_stage_does_not_require_train_config(tmp_path):
    """A sweep builds its dataset dependency without requiring a training matrix."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    sweep = tmp_path / "sweep.yaml"
    intervals.touch()
    fasta.touch()
    sweep.write_text("program: regulonado\nmethod: bayes\n")
    results = tmp_path / "results"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
scaling:
  method: tmm
parameter_sweep:
  enabled: true
  sweep_config: {sweep}
  agents: 2
  trials_per_agent: 1
  cpus_per_agent: 4
  mem_mb_per_agent: 64000
  runtime_minutes_per_agent: 240
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "regulonado",
            "pipeline",
            str(config),
            "parameter-sweep",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert str(results / "parameter-sweep" / "sweep.done") in output
    assert "train_phase" not in output
    assert "parameter_sweep_agent" in output
    assert "agent_0.done" in output
    assert "agent_1.done" in output
    assert "wandb agent" in output
    assert "--count 1" in output
    assert "--forward-signals" in output
    assert "mem_mb=64000" in output
    assert "runtime=240" in output

    all_result = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(WORKFLOW),
            "--configfile",
            str(config),
            "--cores",
            "1",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    all_output = all_result.stdout + all_result.stderr
    assert all_result.returncode == 0, all_output
    assert "Empty file path encountered" not in all_output


@pytest.mark.parametrize(
    ("track_config", "expected_flag"),
    [
        ("", ""),
        ("  tracks: [H3K27ac, CTCF]\n", "--tracks H3K27ac,CTCF"),
    ],
)
def test_prediction_stage_writes_all_tracks_or_one_selected_track(
    tmp_path, track_config, expected_flag
):
    """The optional prediction stage resolves the final run checkpoint once."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    intervals.touch()
    fasta.touch()
    results = tmp_path / "results"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: finetune, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
prediction:
  run: fold_0
{track_config}  whole_genome: true
  chromsizes: {tmp_path / "genome.sizes"}
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
    assert "predict_bigwigs" in output
    assert "regulonado predict" in output
    assert str(results / "predictions" / "fold_0" / ".complete") in output
    assert "--whole-genome" in output
    assert f"--chromsizes {tmp_path / 'genome.sizes'}" in output
    if expected_flag:
        assert expected_flag in output
    else:
        assert "--tracks" not in output


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
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
      - {{name: second, preset: unfreeze_output}}
  runs:
    - {{name: run_a, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: run_b, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
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
    assert re.search(r"train_schedule_preflight\s+4", result.stdout)
    assert re.search(r"train_phase\s+4", result.stdout)
    assert "results/train/run_a/first/trainer_state.json" in result.stdout
    assert "results/train/run_a/second/trainer_state.json" in result.stdout
    assert "results/train/run_b/first/trainer_state.json" in result.stdout
    assert "results/train/run_b/second/trainer_state.json" in result.stdout
    assert "--metadata" in result.stdout
    assert "--schedule-only" in result.stdout
    assert "results/train/run_a/first/schedule.json" in result.stdout
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
    assert re.search(r"train_schedule_preflight\s+4", cli_result.stdout + cli_result.stderr)
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
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: head, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: fold_1, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
design:
  candidates: {candidates}
  checkpoint_dirs:
    - {checkpoint_a}
    - {checkpoint_b}
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

    # shard_candidates is a checkpoint (its shard count depends on its own input's real
    # row count, which may not exist at DAG-construction time — see design.smk), so a
    # plain --dry-run can't resolve design_shard/merge_designs until it has actually run.
    # Executing it for real here is cheap: it only splits a BED file.
    checkpoint_result = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(WORKFLOW),
            "--configfile",
            str(config),
            "--cores",
            "1",
            str(tmp_path / "results" / "design" / "shards"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert checkpoint_result.returncode == 0, checkpoint_result.stdout + checkpoint_result.stderr

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
    # The target/method dispatch stays a direct CLI flag; the tuning settings reach the command
    # as a validated DesignConfig written to the --params file (cli/design.py, F03/S5).
    assert "--target K562" in output
    assert "--params" in output
    assert '"objective": "selective-activation"' in output
    assert '"gain_transform": "log2-fold-change"' in output
    assert '"gain_pseudocount": 1.0' in output
    assert '"offtarget_boost_weight": 2.5' in output
    assert '"offtarget_boost_tolerance": 0.05' in output
    assert '"offtarget_temperature": 0.4' in output
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
    assert '"objective": "selective-activation"' in cli_output
    assert '"gain_transform": "log2-fold-change"' in cli_output


def test_attribution_stage_chains_into_design(tmp_path):
    """The attribution stage's core_regions.bed can be design's candidates input."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    intervals.touch()
    fasta.touch()
    candidates.write_text("chr1\t100\t700\tcand1\nchr1\t2000\t2600\tcand2\n")

    results = tmp_path / "results"
    config = tmp_path / "attribution-config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: fold_1, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
attribution:
  candidates: {candidates}
  shards: 2
  runs: [fold_0, fold_1]
  stride: 4
  fix_width: 300
  bigwig: true
  targets:
    - {{name: hl60, track: atac_hl60, settings: {{quantile: 0.92}}}}
design:
  candidates: {results / "attribution" / "hl60" / "core_regions.bed"}
  shards: 1
  design_runs: [fold_0]
  holdout_run: fold_1
  targets:
    - {{name: hl60_design, target: HL-60, group_by: source, method: ism}}
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

    assert re.search(r"attribute_shard\s+2", result.stdout)
    assert re.search(r"merge_attributions\s+1", result.stdout)
    assert "regulonado attribute" in result.stdout
    # The named track is a direct CLI flag; common and per-target settings reach the command as
    # a validated AttributionConfig written to the --params file (cli/attribute.py, F03/S5).
    assert "--track atac_hl60" in result.stdout
    assert "--params" in result.stdout
    assert '"stride": 4' in result.stdout
    assert '"fix_width": 300' in result.stdout
    assert '"quantile": 0.92' in result.stdout
    assert '"bigwig": true' in result.stdout
    # And design must run downstream of attribution, not in parallel with it: shard_candidates
    # (a checkpoint, since its shard count depends on content merge_attributions produces)
    # depends on attribution's merged output, and merge_designs in turn depends on the
    # checkpoint — Snakemake can't resolve its exact shard-level inputs in a single dry-run
    # before the checkpoint has actually executed, which is the correctness fix itself: the
    # old code guessed a shard count from a file that might not exist yet, silently wrong.
    assert re.search(r"checkpoint shard_candidates:\n\s+input: .*core_regions\.bed", result.stdout)
    assert re.search(r"shard_candidates\s+1", result.stdout)
    assert "will result in alteration of the DAG of jobs" in result.stdout
    assert re.search(r"rule merge_designs:\n\s+input: <TBD>", result.stdout)


def test_design_from_attribution_resolves_the_same_core_regions_path(tmp_path):
    """`design.from_attribution: hl60` is shorthand for hand-typing
    `results_dir/attribution/hl60/core_regions.bed` as `design.candidates`."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    intervals.touch()
    fasta.touch()
    candidates.write_text("chr1\t100\t700\tcand1\n")

    results = tmp_path / "results"
    config = tmp_path / "from-attribution-config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: fold_1, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
attribution:
  candidates: {candidates}
  shards: 1
  runs: [fold_0, fold_1]
  targets:
    - {{name: hl60, track: atac_hl60}}
design:
  from_attribution: hl60
  shards: 1
  design_runs: [fold_0]
  holdout_run: fold_1
  targets:
    - {{name: hl60_design, target: HL-60, group_by: source, method: ism}}
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
    expected = str(results / "attribution" / "hl60" / "core_regions.bed")
    assert re.search(
        rf"checkpoint shard_candidates:\n\s+input: {re.escape(expected)}", result.stdout
    )


def test_design_from_attribution_rejects_an_unknown_target_name(tmp_path):
    """Fails fast at DAG-construction time rather than deep in a shell command."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    intervals.touch()
    fasta.touch()
    candidates.write_text("chr1\t100\t700\tcand1\n")

    results = tmp_path / "results"
    config = tmp_path / "from-attribution-bad-config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: fold_1, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
attribution:
  candidates: {candidates}
  shards: 1
  runs: [fold_0, fold_1]
  targets:
    - {{name: hl60, track: atac_hl60}}
design:
  from_attribution: does_not_exist
  shards: 1
  design_runs: [fold_0]
  holdout_run: fold_1
  targets:
    - {{name: hl60_design, target: HL-60, group_by: source, method: ism}}
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
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert result.returncode != 0
    assert "does_not_exist" in result.stderr + result.stdout


def test_attribution_target_group_and_track_annotations_reach_the_pipeline(tmp_path):
    """A `target`/`group_by` attribution target (instead of `track`) reaches the CLI as
    `--target`/`--group-by`, and `inputs.track_annotations` reaches `tracks assemble` as
    `--annotations` — a separate DAG input from track_sheet/bigwig_dir, so it doesn't force
    track_discovery to re-run (docs/track-table.md#adding-metadata-after-discovery)."""
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    annotations = tmp_path / "groups.csv"
    intervals.touch()
    fasta.touch()
    candidates.write_text("chr1\t100\t700\tcand1\n")
    annotations.write_text("track_name,group\natac_hl60,hl60\n")

    results = tmp_path / "results"
    config = tmp_path / "attribution-group-config.yaml"
    config.write_text(
        f"""
results_dir: {results}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  track_annotations: {annotations}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
    - {{name: fold_1, seed: 20, recipe: finetune, backbone: {{type: borzoi, pretrained: model/b}}}}
attribution:
  candidates: {candidates}
  shards: 1
  runs: [fold_0, fold_1]
  targets:
    - {{name: hl60, target: hl60, group_by: group}}
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
    assert "--target hl60 --group-by group" in result.stdout
    assert "--track " not in result.stdout
    assert f"--annotations {annotations}" in result.stdout

    cli_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "regulonado",
            "pipeline",
            str(config),
            "attribution",
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
    assert str(results / "attribution" / "hl60" / "core_regions.bed") in cli_output


def test_attribution_stage_is_absent_unless_configured(tmp_path):
    """No 'attribution:' key means the rules are never defined and rule all is unaffected."""
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
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
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
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    assert result.returncode == 0, result.stderr
    assert "attribute_shard" not in result.stdout
    assert "merge_attributions" not in result.stdout
    assert "regulonado attribute" not in result.stdout
    assert "core_regions.bed" not in result.stdout


def test_design_shard_count_is_stable_across_runs_once_candidates_exist(tmp_path):
    """Regression guard for the DAG-time read_text() bug this checkpoint replaces.

    Before: `shard_candidates` read `design.candidates`'s line count as a plain
    module-level computation — correct once the file existed, but with no
    guarantee it stayed that way if re-evaluated before vs. after the file was
    written (as happens when candidates chain from another rule's output; see
    `test_attribution_stage_chains_into_design`). Now: `shard_candidates` is a
    checkpoint, so the shard count only ever comes from the file's real
    content, resolved fresh (and identically) every time it's asked for.
    """
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")

    intervals = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    candidates = tmp_path / "candidates.bed"
    intervals.touch()
    fasta.touch()
    # 5 rows, shards: 2 below -> min(2, 5) = 2 real shard files.
    candidates.write_text("\n".join(f"chr1\t{i * 100}\t{i * 100 + 50}" for i in range(5)) + "\n")

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  drop_missing: true
  dedupe_tracks: content
targets:
  profile:
    intervals: {intervals}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
    extract_threads: 1
    write_threads: 1
    stage_to_scratch: false
scaling:
  method: tmm
train:
  nproc_per_node: 1
  recipes:
    finetune:
      - {{name: first, preset: head_only}}
  runs:
    - {{name: fold_0, seed: 10, recipe: finetune, backbone: {{type: borzoi, pretrained: model/a}}}}
design:
  candidates: {candidates}
  shards: 2
  targets:
    - {{name: k562, target: K562, group_by: source, method: ism}}
"""
    )
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")}
    shards_dir = tmp_path / "results" / "design" / "shards"

    def dry_run():
        return subprocess.run(
            [
                snakemake,
                "--snakefile",
                str(WORKFLOW),
                "--configfile",
                str(config),
                "--cores",
                "1",
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    # 1. Before the checkpoint has run, Snakemake correctly refuses to guess the shard
    #    count — merge_designs' inputs are unresolved rather than a wrong number.
    before = dry_run()
    assert before.returncode == 0, before.stderr
    assert re.search(r"rule merge_designs:\n\s+input: <TBD>", before.stdout)

    # 2. Run the checkpoint for real.
    build_checkpoint = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(WORKFLOW),
            "--configfile",
            str(config),
            "--cores",
            "1",
            str(shards_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert build_checkpoint.returncode == 0, build_checkpoint.stdout + build_checkpoint.stderr
    assert sorted(p.name for p in shards_dir.glob("*.bed")) == ["0.bed", "1.bed"]

    # 3. Now resolvable, and resolves to the real count — not the requested-shards guess.
    after = dry_run()
    assert after.returncode == 0, after.stderr
    assert re.search(r"design_shard\s+2", after.stdout)

    # 4. Re-resolving (a second dry-run) gives the *same* count: no silent drift.
    again = dry_run()
    assert re.search(r"design_shard\s+2", again.stdout)
    assert sorted(p.name for p in shards_dir.glob("*.bed")) == ["0.bed", "1.bed"]


def _snakemake_dry_run(tmp_path, config, *targets):
    snakemake = shutil.which("snakemake", path=str(Path(sys.executable).parent))
    if snakemake is None:
        pytest.skip("Snakemake is an optional workflow dependency")
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
            *targets,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )
    return result.stdout + result.stderr, result.returncode


def _pipeline_dry_run(tmp_path, config, stage="train"):
    return subprocess.run(
        [sys.executable, "-m", "regulonado", "pipeline", str(config), stage, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path / "cache")},
    )


def _region_count_files(tmp_path):
    """A two-chromosome region set plus anchor/background/candidate files and a genome."""
    import polars as pl

    regions = tmp_path / "regions.parquet"
    pl.DataFrame(
        {"chrom": ["chr1", "chr2"], "start": [1000, 1000], "end": [3114, 3114]}
    ).write_parquet(regions)
    for name in ("genome.fa", "anchor.bed", "background.bed", "candidates.bed"):
        (tmp_path / name).touch()
    return regions


CURRICULUM = """
    curriculum:
      - {name: pretrain, preset: pretrain}
      - {name: specific, preset: specific}
      - {name: target, preset: target}
"""


def _cached_run(name, *, seed=0, backbone="type: alphagenome, pretrained: all_folds", cache=""):
    return f"""
    - name: {name}
      seed: {seed}
      recipe: curriculum
      backbone: {{{backbone}}}
      trunk: cached
      target: region_counts
      target_group: HL-60
{cache}"""


def _count_only_config(tmp_path, *, runs=None, recipe_settings=""):
    """Region counts from BAMs only: no bigWig source, profile target or scaling."""
    regions = _region_count_files(tmp_path)
    (tmp_path / "tracks.csv").write_text("track_name,sample_id,group,assay\na,SRR1,hl60,atac\n")
    train = ""
    if runs is not None:
        train = "train:\n  recipes:" + CURRICULUM + recipe_settings + "  runs:" + "".join(runs)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  fasta: {tmp_path / "genome.fa"}
  track_sheet: {tmp_path / "tracks.csv"}
  bam_dir: {tmp_path / "bams"}
targets:
  region_counts:
    regions: {regions}
    anchor_regions: {tmp_path / "anchor.bed"}
    background_regions: {tmp_path / "background.bed"}
    exclude_regions: {tmp_path / "candidates.bed"}
{train}"""
    )
    return config


def test_cached_run_plans_counts_embeddings_and_chained_phases(tmp_path):
    """A trunk: cached run pulls in BAM tracks, region counts, its cache and its recipe."""
    pytest.importorskip("hydra")
    config = _count_only_config(tmp_path, runs=[_cached_run("run_a", seed=7)])
    results = tmp_path / "results"
    output, code = _snakemake_dry_run(tmp_path, config)
    assert code == 0, output

    # Tracks come from BAMs; no bigWig stage is defined at all.
    assert "--format bam" in output
    for bigwig_rule in ("track_interval_means", "scale_factors", "track_qc", "build_dataset"):
        assert not re.search(rf"(?<![a-z_]){bigwig_rule}\b", output)

    # Counting fans out per track from a checkpoint, so its gather is still <TBD>.
    assert "checkpoint region_count_track_names:" in output
    assert re.search(r"rule region_counts_gather:\n\s+input: .*<TBD>", output)
    assert str(results / "region_counts" / "dataset" / "counts.parquet") in output

    # Embedding: one job per chromosome, reading only the region set.
    cache = results / "embeddings" / "alphagenome-all_folds-ctx1048576-stride524288"
    assert re.search(r"embed_chrom\s+2", output)
    assert re.search(r"embed_done\s+1", output)
    assert str(cache / "chr1.parquet") in output
    embed_inputs = re.findall(r"rule embed_chrom:\n\s+input: (.*)", output)
    assert embed_inputs and all("region_set.parquet" in line for line in embed_inputs)
    assert "--backbone alphagenome --pretrained all_folds" in output

    # Training: the recipe's three phases, chained by warm starts, through one CLI.
    assert re.search(r"train_phase\s+3", output)
    assert "train_schedule_preflight" not in output
    assert f"--trunk cached --embeddings {cache} --target-group HL-60" in output
    for preset in ("pretrain", "specific", "target"):
        assert f"--preset {preset}" in output
    assert "--set seed=7" in output
    assert f'++data.exclude_regions="{tmp_path / "candidates.bed"}"' in output
    assert "Warm-starting run_a/specific from" in output
    assert "Warm-starting run_a/target from" in output
    assert re.search(r"rule all:\n\s+input: .*run_a/target/trainer_state\.json", output)

    cli = _pipeline_dry_run(tmp_path, config)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert re.search(r"train_phase\s+3", cli.stdout + cli.stderr)


def test_runs_on_one_trunk_share_an_embedding_cache(tmp_path):
    runs = [
        _cached_run("seed_0", seed=0),
        _cached_run("seed_1", seed=1),
        _cached_run(
            "flashzoi",
            backbone="type: borzoi, pretrained: johahi/flashzoi-replicate-0",
            cache="      cache: {pool_to: 128}\n",
        ),
    ]
    output, code = _snakemake_dry_run(tmp_path, _count_only_config(tmp_path, runs=runs))
    assert code == 0, output
    assert re.search(r"embed_done\s+2", output)
    assert re.search(r"embed_chrom\s+4", output)
    assert "borzoi-johahi_flashzoi-replicate-0-pool128" in output


def test_region_counts_without_runs_builds_the_dataset(tmp_path):
    output, code = _snakemake_dry_run(tmp_path, _count_only_config(tmp_path))
    assert code == 0, output
    assert "region_counts_gather" in output
    assert "embed_chrom" not in output
    assert "train_phase" not in output


def test_training_typos_fail_before_scheduling(tmp_path):
    pytest.importorskip("hydra")
    typo = """      - name: extra
        preset: target
        settings:
          data:
            exclude_region: typo.bed
"""
    config = _count_only_config(tmp_path, runs=[_cached_run("run_a")], recipe_settings=typo)
    result = _pipeline_dry_run(tmp_path, config)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "data.exclude_region" in output
    assert "Building DAG" not in output


def test_live_and_cached_runs_share_one_track_table(tmp_path):
    """A profile target discovers bigWig tracks carrying their BAMs; region counts count
    those same tracks, with anchors shared from scaling."""
    pytest.importorskip("hydra")
    regions = _region_count_files(tmp_path)
    (tmp_path / "intervals.bed").touch()
    (tmp_path / "tracks.csv").write_text(
        "track_name,bigwig,bam,group,assay\na,a.bw,a.bam,hl60,atac\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  fasta: {tmp_path / "genome.fa"}
  track_sheet: {tmp_path / "tracks.csv"}
targets:
  profile:
    intervals: {tmp_path / "intervals.bed"}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
  region_counts:
    regions: {regions}
scaling:
  method: anchor
  anchor_regions: {tmp_path / "anchor.bed"}
  background_regions: {tmp_path / "background.bed"}
train:
  common:
    data:
      apply_squash: false
  recipes:
    finetune:
      - {{name: head, preset: head_only}}
{CURRICULUM}
  runs:
    - name: profile_run
      seed: 0
      recipe: finetune
      backbone: {{type: borzoi, pretrained: model/a}}
{_cached_run("counts_run")}"""
    )
    output, code = _snakemake_dry_run(tmp_path, config)
    assert code == 0, output
    assert "--format bigwig" in output
    assert re.search(
        r"checkpoint region_count_track_names:\n\s+input: \S+/tracks/tracks\.parquet", output
    )# Only the live run gets a CPU schedule preflight; both train through one rule.
    assert re.search(r"train_schedule_preflight\s+1", output)
    assert re.search(r"train_phase\s+4", output)
    assert "--metadata" in output and "--trunk cached" in output
    assert 'backbone="borzoi"' in output


def test_region_count_rules_are_absent_for_profile_only_configs(tmp_path):
    (tmp_path / "intervals.bed").touch()
    (tmp_path / "genome.fa").touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
results_dir: {tmp_path / "results"}
inputs:
  fasta: {tmp_path / "genome.fa"}
  bigwig_dir: {tmp_path / "bigwigs"}
targets:
  profile:
    intervals: {tmp_path / "intervals.bed"}
    context_length: 100
    bin_size: 10
    n_pred_bins: 4
    shift_max_bp: 0
"""
    )
    output, code = _snakemake_dry_run(tmp_path, config)
    assert code == 0, output
    assert "build_dataset" in output
    for rule in ("region_set", "embed_chrom", "region_count_track_names"):
        assert rule not in output
