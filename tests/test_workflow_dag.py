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
dataset:
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
dataset:
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
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
dataset:
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
  runs:
    - {{name: fold_0, seed: 10, pretrained_model: model/a}}
    - {{name: fold_1, seed: 20, pretrained_model: model/b}}
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
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
  track_annotations: {annotations}
dataset:
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
  runs:
    - {{name: fold_0, seed: 10, pretrained_model: model/a}}
    - {{name: fold_1, seed: 20, pretrained_model: model/b}}
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
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
dataset:
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
  runs:
    - {{name: fold_0, seed: 10, pretrained_model: model/a}}
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
  intervals: {intervals}
  fasta: {fasta}
  bigwig_dir: {tmp_path / "bigwigs"}
dataset:
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
  runs:
    - {{name: fold_0, seed: 10, pretrained_model: model/a}}
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
            [snakemake, "--snakefile", str(WORKFLOW), "--configfile", str(config),
             "--cores", "1", "--dry-run"],
            check=False, capture_output=True, text=True, env=env,
        )

    # 1. Before the checkpoint has run, Snakemake correctly refuses to guess the shard
    #    count — merge_designs' inputs are unresolved rather than a wrong number.
    before = dry_run()
    assert before.returncode == 0, before.stderr
    assert re.search(r"rule merge_designs:\n\s+input: <TBD>", before.stdout)

    # 2. Run the checkpoint for real.
    build_checkpoint = subprocess.run(
        [snakemake, "--snakefile", str(WORKFLOW), "--configfile", str(config),
         "--cores", "1", str(shards_dir)],
        check=False, capture_output=True, text=True, env=env,
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
