from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shlex import join as shell_join
from typing import Annotated, Optional

import numpy as np
import torch
import typer
from loguru import logger
from regulonado.cli.pipeline import pipeline as _pipeline

app = typer.Typer(no_args_is_help=True)
normalization_app = typer.Typer(
    no_args_is_help=True,
    help="Infer, correct, and apply per-track signal normalisation factors.",
)


def _version_callback(value: bool) -> None:
    if value:
        try:
            package_version = version("regulonado")
        except PackageNotFoundError:
            package_version = "unknown"
        try:
            from regulonado import _rs

            rust_version = _rs.version()
        except (ImportError, AttributeError):
            rust_version = "unknown"
        typer.echo(f"regulonado {package_version} (Rust extension {rust_version})")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show Python package and Rust extension versions.",
    ),
) -> None:
    """Build datasets, train models, run inference, and manage workflows."""


def _parse_seqnado_projects(values: Optional[list[str]]) -> dict[str, str]:
    """Parse repeated ``PATH`` / ``NAME=PATH`` project options.

    Without an explicit name a project is labelled by the directory containing
    its output dir, which for a SeqNado layout is the project folder itself
    (``2026-08-10_myproj/seqnado_output`` -> ``2026-08-10_myproj``).
    """
    projects: dict[str, str] = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep:
            path = name
            name = Path(path).expanduser().resolve().parent.name
        if not path:
            raise typer.BadParameter(
                f"Expected PATH or NAME=PATH, got {value!r}", param_hint="--seqnado-project"
            )
        if name in projects:
            raise typer.BadParameter(
                f"Duplicate project name {name!r}; give each one an explicit NAME=PATH",
                param_hint="--seqnado-project",
            )
        projects[name] = path
    return projects


@app.command()
def config(
    output: Annotated[
        Optional[Path], typer.Option("-o", "--output", help="Where to write the config YAML.")
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive/--no-interactive",
            help="Prompt for values, or take every default without asking.",
        ),
    ] = True,
    from_seqnado: Annotated[
        Optional[list[str]],
        typer.Option(
            "--from-seqnado",
            help=(
                "SeqNado output directory as PATH or NAME=PATH (repeatable). "
                "Tracks and their annotation are read from the project."
            ),
        ),
    ] = None,
    genome: Annotated[
        Optional[str],
        typer.Option("--genome", help="Genome name from the shared SeqNado genome registry."),
    ] = None,
    project_name: Annotated[
        Optional[str], typer.Option("--name", help="Project name; seeds the results directory.")
    ] = None,
    fill_missing: Annotated[
        Optional[Path],
        typer.Option(
            "--fill-missing",
            help=(
                "Load an existing config and fill in what was left unset. "
                "Overwrites in place unless -o/--output or --new-file is given."
            ),
        ),
    ] = None,
    new_file: Annotated[
        bool,
        typer.Option(
            "--new-file",
            help="With --fill-missing, write '<name>.filled.yaml' instead of overwriting.",
        ),
    ] = False,
    track_sheet_out: Annotated[
        Optional[Path],
        typer.Option("--track-sheet-out", help="Where to write the derived track sheet."),
    ] = None,
    assume_same_genome: Annotated[
        bool,
        typer.Option(
            "--assume-same-genome",
            help="Aggregate projects whose reference genome cannot be read from their config.",
        ),
    ] = False,
) -> None:
    """Generate a workflow config, interactively or from flags.

    Follows the same shape as `seqnado config`: sequential prompts with defaults
    in brackets, validated as you answer. Pass --no-interactive to take every
    default instead, which is the scripting and CI path.
    """
    import yaml
    from regulonado.config.generator import build_config

    base = None
    if fill_missing is not None:
        if not fill_missing.exists():
            typer.echo(f"Config not found: {fill_missing}", err=True)
            raise typer.Exit(1)
        try:
            # Parsed as a plain mapping, not validated: the whole point of
            # --fill-missing is that the input is incomplete.
            base = yaml.safe_load(fill_missing.read_text()) or {}
        except yaml.YAMLError as exc:
            typer.echo(f"Could not parse {fill_missing}: {exc}", err=True)
            raise typer.Exit(1) from exc
        if not isinstance(base, dict):
            typer.echo(f"{fill_missing} is not a YAML mapping", err=True)
            raise typer.Exit(1)

    try:
        built = build_config(
            interactive=interactive,
            project_name=project_name,
            genome=genome,
            from_seqnado=list(from_seqnado) if from_seqnado else None,
            base=base,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if output is not None:
        destination = output
    elif fill_missing is not None:
        destination = (
            fill_missing.parent / f"{fill_missing.stem}.filled{fill_missing.suffix}"
            if new_file
            else fill_missing
        )
    else:
        destination = Path("config.yaml")

    # A sheet is only written when tracks came from SeqNado; a hand-written
    # sheet named in the config is left exactly as the user wrote it.
    if built.inputs.seqnado_projects:
        sheet_path = track_sheet_out or destination.parent / "track_sheet.csv"
        try:
            from regulonado.tracks import TrackSheet

            sheet = TrackSheet.from_seqnado_projects(
                [
                    {"name": p.name, "path": p.path, "method": p.method, "scale": p.scale}
                    for p in built.inputs.seqnado_projects
                ],
                assume_same_genome=assume_same_genome,
            )
            sheet.to_csv(sheet_path)
            built.inputs.track_sheet = str(sheet_path)
            logger.info(f"Wrote track sheet with {len(sheet)} track(s) -> {sheet_path}")
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

    built.to_yaml(destination)
    logger.success(f"Wrote config -> {destination}")


@app.command()
def init(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Show what would be written.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite presets that already exist.")
    ] = False,
) -> None:
    """Install Snakemake execution presets into ~/.config/snakemake/.

    Uses the same directory and naming scheme as `seqnado init`, so a machine
    running both pipelines keeps one set of execution configs. Presets already
    present are left alone — including SeqNado's — so this is safe to re-run.
    """
    import shutil
    from importlib import resources

    from regulonado.cli.profiles import _packaged_profiles

    target_root = Path.home() / ".config" / "snakemake"
    packaged = _packaged_profiles()
    if not packaged:
        typer.echo("No packaged presets found; the installation may be incomplete.", err=True)
        raise typer.Exit(1)

    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)

    installed, skipped = 0, 0
    for shortcode, directory in sorted(packaged.items()):
        destination = target_root / directory
        if destination.exists() and not force:
            logger.info(f"Preset '{shortcode}' already installed: {destination}")
            skipped += 1
            continue

        source = resources.files("regulonado.workflow.profiles").joinpath(directory)
        if dry_run:
            typer.echo(f"[dry-run] would install '{shortcode}' -> {destination}")
            installed += 1
            continue

        with resources.as_file(source) as source_path:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        logger.info(f"Installed preset '{shortcode}' -> {destination}")
        installed += 1

    logger.info(f"{installed} preset(s) installed, {skipped} left unchanged.")

    genome_config = Path(os.environ.get("SEQNADO_CONFIG", Path.home()))
    genome_config = genome_config / ".config" / "seqnado" / "genome_config.json"
    if genome_config.exists():
        logger.info(f"Genome registry: {genome_config}")
    else:
        # Shared with SeqNado on purpose: one place to record where the FASTA,
        # chrom.sizes and blacklist for a genome live.
        logger.warning(
            f"No genome registry at {genome_config}. Run 'seqnado init' to create one, "
            f"or pass --genome paths to 'regulonado config' directly."
        )

    from regulonado.cli.profiles import format_available_presets

    typer.echo(f"Available presets: {format_available_presets()}")


@app.command()
def train(
    dataset: Annotated[
        Path,
        typer.Argument(help="Saved Regulonado/Hugging Face dataset directory"),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Run directory for checkpoints and diagnostics"),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", "-p", help="Named training preset"),
    ] = "head_only",
    metadata: Annotated[
        Optional[Path],
        typer.Option("--metadata", help="Metadata JSON to use instead of the dataset copy"),
    ] = None,
    nproc_per_node: Annotated[
        int,
        typer.Option(
            "--nproc-per-node",
            help="Use torchrun with this many local processes when >1",
        ),
    ] = 1,
    resume_from_checkpoint: Annotated[
        Optional[str],
        typer.Option(
            "--resume-from-checkpoint",
            help="Full Trainer resume from a checkpoint dir, or 'true' for latest in output-dir",
        ),
    ] = None,
    init_weights_from_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--init-weights-from-checkpoint",
            help="Warm start from model weights only with a fresh optimizer/scheduler",
        ),
    ] = None,
    max_steps: Annotated[
        Optional[int],
        typer.Option("--max-steps", help="Override trainer.max_steps"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", help="Override per-device train batch size"),
    ] = None,
    eval_batch_size: Annotated[
        Optional[int],
        typer.Option("--eval-batch-size", help="Override per-device eval batch size"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--learning-rate", "--lr", help="Override head learning rate"),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        typer.Option("--backbone-lr", help="Override backbone learning rate"),
    ] = None,
    num_workers: Annotated[
        Optional[int],
        typer.Option("--num-workers", help="Override DataLoader worker count"),
    ] = None,
    no_wandb: Annotated[
        bool,
        typer.Option("--no-wandb", help="Disable W&B reporting for this run"),
    ] = False,
    settings: Annotated[
        Optional[list[str]],
        typer.Option(
            "--set",
            help="Override one setting as KEY=VALUE (repeatable)",
        ),
    ] = None,
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print the resolved training config and exit"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the resolved command without running it"),
    ] = False,
) -> None:
    """Train one model from a named preset.

    Use ``--set`` for less common settings, for example
    ``--set trainer.max_eval_samples=200``. Use ``regulonado pipeline`` when
    several independent runs or warm-start phases should be orchestrated together.
    """
    if resume_from_checkpoint and init_weights_from_checkpoint:
        typer.echo(
            "Set only one of --resume-from-checkpoint or --init-weights-from-checkpoint.",
            err=True,
        )
        raise typer.Exit(1)

    overrides = [
        f"+experiment={preset}",
        f"data.path={dataset}",
    ]
    if metadata is not None:
        overrides.append(f"data.metadata_path={metadata}")
    if output_dir is not None:
        overrides.append(f"output_dir={output_dir}")
    if resume_from_checkpoint is not None:
        overrides.append(f"trainer.resume_from_checkpoint={resume_from_checkpoint}")
    if init_weights_from_checkpoint is not None:
        overrides.append(f"trainer.init_weights_from_checkpoint={init_weights_from_checkpoint}")
    if max_steps is not None:
        overrides.append(f"trainer.max_steps={max_steps}")
    if batch_size is not None:
        overrides.append(f"trainer.batch_size={batch_size}")
    if eval_batch_size is not None:
        overrides.append(f"trainer.eval_batch_size={eval_batch_size}")
    if learning_rate is not None:
        overrides.append(f"trainer.learning_rate={learning_rate}")
    if backbone_lr is not None:
        overrides.append(f"trainer.backbone_learning_rate={backbone_lr}")
    if num_workers is not None:
        overrides.append(f"trainer.num_workers={num_workers}")
    if no_wandb:
        overrides.append("trainer.report_to=[]")
    for setting in settings or []:
        if "=" not in setting or not setting.split("=", 1)[0].strip():
            raise typer.BadParameter(
                f"Setting must be KEY=VALUE, got {setting!r}", param_hint="--set"
            )
        overrides.append(setting)

    if print_config:
        try:
            from regulonado.training.compose import resolved_training_config
        except ImportError as exc:
            typer.echo(
                "Hydra is required to inspect training presets. Install regulonado[train].",
                err=True,
            )
            raise typer.Exit(127) from exc
        try:
            rendered = resolved_training_config(preset, overrides[1:])
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not compose preset {preset!r}: {exc}", param_hint="--preset/--set"
            ) from exc
        typer.echo(rendered)
        return

    if nproc_per_node > 1:
        import os
        import random

        # Avoid port collisions when multiple jobs land on the same node.
        job_id = int(os.environ.get("SLURM_JOB_ID", 0))
        master_port = 29500 + (job_id % 1000) if job_id else random.randint(29500, 30499)
        command = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            "regulonado.training.runner",
            *overrides,
        ]
    else:
        command = [sys.executable, "-m", "regulonado.training.runner", *overrides]

    typer.echo(shell_join(command))
    if dry_run:
        return
    raise typer.Exit(subprocess.run(command).returncode)


@normalization_app.command("infer")
def scale(
    bigwig_dir: Annotated[Path, typer.Argument(help="Directory containing .bw / .bigwig files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
    glob: Annotated[str, typer.Option("--glob", help="Glob pattern for bigwig files")] = "*.bw",
) -> None:
    """Infer scale factors for all bigwig files in a directory."""
    from regulonado.normalization import infer_scale_factors, save_scale_factors

    bw_files = sorted(bigwig_dir.glob(glob))
    if not bw_files:
        typer.echo(f"No files matching '{glob}' found in {bigwig_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(bw_files)} bigwig files")
    df = infer_scale_factors(bw_files, max_workers=max_workers)
    save_scale_factors(df, output, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {output}")


@normalization_app.command("original")
def calculate_original_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: <metadata_dir>/scale_factors.parquet)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
) -> None:
    """Infer original scale factors for the final_bigwig_paths recorded in a dataset metadata file.

    Output rows are sorted by track_index so they can be applied directly by position.
    """
    import json

    import pandas as pd
    from regulonado.normalization import (
        compute_clip_thresholds,
        infer_scale_factors,
        save_scale_factors,
    )

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)

    with metadata.open() as fh:
        meta = json.load(fh)

    track_records = meta.get("final_track_records", [])
    if not track_records:
        typer.echo("No 'final_track_records' found in metadata.", err=True)
        raise typer.Exit(1)

    bin_size: int = int(meta.get("bin_size", 32))

    # Sort records by track_index to define the canonical order.
    track_records = sorted(track_records, key=lambda r: r["track_index"])
    bw_paths = [Path(r["resolved_path"]) for r in track_records]

    ext = "parquet" if fmt == "parquet" else "csv"
    out_path = output if output is not None else metadata.parent / f"scale_factors.{ext}"

    typer.echo(f"Metadata : {metadata}")
    typer.echo(f"Tracks   : {len(bw_paths)}")
    typer.echo(f"Bin size : {bin_size} bp")
    typer.echo(f"Output   : {out_path}")

    df = infer_scale_factors(bw_paths, max_workers=max_workers)

    # bamnado returns scale_factor = library_size / 1e9, which is the RPKM→raw-counts
    # factor without the bin_size term.  RPKM = reads / (lib/1e6) / (bin_size/1e3),
    # so raw_count = RPKM × (lib/1e6) × (bin_size/1e3) = RPKM × sf_bamnado × bin_size.
    df["scale_factor"] = df["scale_factor"] * bin_size

    # Join track_index and resolved_path from the records, then sort so row i
    # corresponds to track i — enabling direct positional application.
    records_df = pd.DataFrame(
        [
            {"track_index": r["track_index"], "resolved_path": r["resolved_path"]}
            for r in track_records
        ]
    )
    df = df.merge(records_df, left_on="path", right_on="resolved_path", how="left")
    df = df.drop(columns=["resolved_path"]).sort_values("track_index").reset_index(drop=True)

    df = compute_clip_thresholds(df)

    # Put the fields consumed during training first.
    priority = ["track_index", "scale_factor", "clip_soft", "clip_hard"]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@normalization_app.command("tmm")
def calculate_tmm_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help=(
                "Scale-factors parquet from normalization original "
                "(default: <metadata_dir>/scale_factors.parquet)"
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: overwrites --scale-factors input)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to use for TMM estimation"),
    ] = "train",
    trim_m: Annotated[
        float,
        typer.Option(
            "--trim-m",
            help="Fraction to trim from each M-value tail (edgeR default 0.3)",
        ),
    ] = 0.3,
    trim_a: Annotated[
        float,
        typer.Option(
            "--trim-a",
            help="Fraction to trim from each A-value tail (edgeR default 0.05)",
        ),
    ] = 0.05,
    min_count: Annotated[
        float,
        typer.Option(
            "--min-count",
            help="Minimum pseudo-count for a region to be included",
        ),
    ] = 1.0,
) -> None:
    """Compute edgeR-style TMM normalisation factors from the Arrow dataset.

    Reads per-sample mean RPKM from the Arrow shards under <metadata_dir>/<split>/,
    converts to pseudo-counts using library sizes from the scale-factors parquet,
    and runs TMM estimation over the full set of genomic regions.

    The output parquet gains a ``tmm_factor`` column and the ``scale_factor``
    column is updated to ``old_scale_factor / tmm_factor`` so that multiplying
    any raw RPKM BigWig value by the new scale_factor yields TMM-normalised
    approximate raw counts.

    Run ``regulonado enrich-metadata`` afterwards to write the updated values
    into ``final_track_records`` in the metadata JSON.

    \b
    Typical workflow::

    regulonado normalization original metadata.json
    regulonado normalization tmm metadata.json
    regulonado enrich-metadata metadata.json scale_factors.parquet --output enriched.json
    """
    import json

    import pandas as pd
    from regulonado.normalization import compute_tmm_factors, read_dataset_means, save_scale_factors

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)

    with metadata.open() as fh:
        meta = json.load(fh)

    dataset_dir = metadata.parent
    bin_size: int = int(meta.get("bin_size", 32))

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = scale_factors if scale_factors is not None else dataset_dir / f"scale_factors.{ext}"
    out_path = output if output is not None else sf_path

    if not sf_path.exists():
        typer.echo(
            f"Scale-factors file not found: {sf_path}\n"
            "Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)

    sf_df = pd.read_parquet(sf_path) if str(sf_path).endswith(".parquet") else pd.read_csv(sf_path)

    if "library_size" not in sf_df.columns:
        typer.echo("Column 'library_size' missing from scale-factors file.", err=True)
        raise typer.Exit(1)
    if "scale_factor" not in sf_df.columns:
        typer.echo(
            (
                "Column 'scale_factor' missing from scale-factors file.  "
                "Run 'regulonado normalization original' first."
            ),
            err=True,
        )
        raise typer.Exit(1)

    library_sizes = sf_df.sort_values("track_index")["library_size"].to_numpy(dtype=float)

    typer.echo(f"Dataset  : {dataset_dir}")
    typer.echo(f"Split    : {split}")
    typer.echo(f"Tracks   : {len(library_sizes)}")
    typer.echo(f"Bin size : {bin_size} bp")
    typer.echo("")

    means, n_tracks, n_bins = read_dataset_means(dataset_dir, split=split)

    if n_tracks != len(library_sizes):
        typer.echo(
            f"Track count mismatch: dataset has {n_tracks} tracks, "
            f"scale-factors file has {len(library_sizes)}.",
            err=True,
        )
        raise typer.Exit(1)

    region_length_kb = n_bins * bin_size / 1000.0
    typer.echo(f"Samples  : {means.shape[0]}")
    typer.echo(f"Bins/sample: {n_bins}  ({region_length_kb:.1f} kb)")
    typer.echo("")

    tmm = compute_tmm_factors(
        means,
        library_sizes,
        region_length_kb,
        trim_m=trim_m,
        trim_a=trim_a,
        min_count=min_count,
    )

    # Report
    sf_sorted = sf_df.sort_values("track_index").reset_index(drop=True)
    typer.echo(
        f"{'Track':>5}  {'samplename':<30}  {'tmm_factor':>12}  {'old_sf':>12}  {'new_sf':>12}"
    )
    for i, (_, row) in enumerate(sf_sorted.iterrows()):
        old_sf = float(row["scale_factor"])
        new_sf = old_sf / tmm[i]
        name = str(row.get("samplename", i))[:30]
        typer.echo(
            f"{int(row['track_index']):>5}  {name:<30}  {tmm[i]:>12.6f}  "
            f"{old_sf:>12.6f}  {new_sf:>12.6f}"
        )

    # Write updated parquet: add tmm_factor, overwrite scale_factor
    sf_df = sf_df.sort_values("track_index").reset_index(drop=True)
    sf_df["tmm_factor"] = tmm
    sf_df["scale_factor"] = sf_df["scale_factor"] / sf_df["tmm_factor"]

    priority = ["track_index", "scale_factor", "tmm_factor", "clip_soft", "clip_hard"]
    rest = [c for c in sf_df.columns if c not in priority]
    sf_df = sf_df[priority + rest]

    save_scale_factors(sf_df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"\nSaved updated scale factors to {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@normalization_app.command("seqnado")
def calculate_seqnado_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    project: Annotated[
        Path,
        typer.Option("--project", help="SeqNado output directory (seqnado_output/)."),
    ],
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help="Scale-factors parquet from 'normalization original'.",
        ),
    ] = None,
    method: Annotated[
        Optional[str],
        typer.Option(
            "--method", "-m", help="Spike-in method, e.g. orlando. Default: the only one."
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path (default: overwrites --scale-factors)."),
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Output format: csv or parquet")
    ] = "parquet",
) -> None:
    """Apply SeqNado's spike-in normalisation factors instead of estimating our own.

    Takes library-size factors from 'normalization original' and corrects them
    with the factors SeqNado already computed, exactly as 'normalization tmm'
    applies a TMM correction: ``scale_factor = old_scale_factor / factor``, with
    factors normalised to a geometric mean of 1 so the overall magnitude is
    unchanged.

    Only valid within a single SeqNado project — its factors are not comparable
    across projects. Use 'tmm' when aggregating several.
    """
    import json

    import numpy as np
    import pandas as pd
    from regulonado.normalization import save_scale_factors

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = (
        scale_factors if scale_factors is not None else metadata.parent / f"scale_factors.{ext}"
    )
    if not sf_path.exists():
        typer.echo(
            f"Scale factors not found: {sf_path}. Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)
    out_path = output if output is not None else sf_path

    from regulonado._seqnado import SeqNadoUnavailableError, open_project

    try:
        seqnado_project = open_project(project)
        factors = seqnado_project.load_normalisation_factors(method)
    except SeqNadoUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        typer.echo(
            f"No normalisation factors in {project}: {exc}. SeqNado writes these "
            f"under resources/<method>/normalisation_factors.tsv when spike-in "
            f"normalisation is enabled.",
            err=True,
        )
        raise typer.Exit(1) from exc

    sample_column = next((c for c in ("sample", "sample_id", "sample_name") if c in factors), None)
    factor_column = next(
        (c for c in ("norm_factor", "scale_factor", "factor") if c in factors), None
    )
    if sample_column is None or factor_column is None:
        typer.echo(
            f"Could not find sample and factor columns in SeqNado's normalisation "
            f"table; got columns: {', '.join(map(str, factors.columns))}",
            err=True,
        )
        raise typer.Exit(1)

    # load_normalisation_factors(None) concatenates every method's table and
    # tags each row with its method. Silently keeping the first row per sample
    # would pick a spike-in method at random, so make the caller choose.
    if method is None and "method" in factors.columns:
        present = sorted(factors["method"].dropna().unique())
        if len(present) > 1:
            typer.echo(
                f"{project} has normalisation factors for several spike-in methods "
                f"({', '.join(map(str, present))}). Pass --method to choose one.",
                err=True,
            )
            raise typer.Exit(1)

    lookup = (
        factors.dropna(subset=[sample_column, factor_column])
        .drop_duplicates(subset=[sample_column], keep="first")
        .set_index(sample_column)[factor_column]
        .astype(float)
    )

    with metadata.open() as fh:
        meta = json.load(fh)
    records = sorted(meta.get("final_track_records", []), key=lambda r: r["track_index"])
    if not records:
        typer.echo("No 'final_track_records' found in metadata.", err=True)
        raise typer.Exit(1)

    # Tracks carry both names; SeqNado keys its table on the sample name, which
    # for IP assays is '<sample>_<ip>' — the same string as the bigwig stem.
    names = [
        record.get("sample_id") or record.get("track_name") or Path(record["path"]).stem
        for record in records
    ]
    missing = [name for name in names if name not in lookup.index]
    if missing:
        available = ", ".join(map(str, lookup.index[:8]))
        typer.echo(
            f"{len(missing)}/{len(names)} track(s) have no SeqNado normalisation "
            f"factor: {', '.join(missing[:8])}.\nFactors are available for: {available}",
            err=True,
        )
        raise typer.Exit(1)

    values = np.asarray([lookup[name] for name in names], dtype=float)
    if np.any(values <= 0):
        typer.echo("SeqNado normalisation factors must be positive.", err=True)
        raise typer.Exit(1)
    # Normalise to geometric mean 1, matching how the TMM correction is applied.
    values = values / float(np.exp(np.mean(np.log(values))))

    df = pd.read_parquet(sf_path) if fmt == "parquet" else pd.read_csv(sf_path)
    df = df.sort_values("track_index").reset_index(drop=True)
    if len(df) != len(records):
        typer.echo(
            f"Scale factors have {len(df)} row(s) but metadata has {len(records)} track(s).",
            err=True,
        )
        raise typer.Exit(1)

    df["seqnado_norm_factor"] = values
    df["scale_factor"] = df["scale_factor"] / values

    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Applied {len(values)} SeqNado normalisation factor(s) -> {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@normalization_app.command("bamnado")
def calculate_bamnado_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    bam_dir: Annotated[
        Path,
        typer.Option(
            "--bam-dir",
            help="Directory of BAM files, one per track, named <track-stem>.bam",
        ),
    ],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help=("bamnado bam-normalize method: tmm, csaw-background, cpm, "
                  "median-of-ratios, spike-in"),
        ),
    ] = "csaw-background",
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help=(
                "Scale-factors parquet from normalization original "
                "(default: <metadata_dir>/scale_factors.parquet)"
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: overwrites --scale-factors input)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    bin_size_bp: Annotated[
        int,
        typer.Option("--bin-size-bp", help="bamnado background bin size in bp"),
    ] = 10_000,
    exclude_top_percent: Annotated[
        float,
        typer.Option(
            "--exclude-top-percent", help="Drop this percentage of highest-count bins first"
        ),
    ] = 5.0,
    reference_sample: Annotated[
        Optional[str],
        typer.Option(
            "--reference-sample", help="Sample name to use as reference (default: bamnado's choice)"
        ),
    ] = None,
    logratio_trim: Annotated[
        float,
        typer.Option("--logratio-trim", help="TMM trim fraction for M-values"),
    ] = 0.3,
    sum_trim: Annotated[
        float,
        typer.Option("--sum-trim", help="TMM trim fraction for A-values"),
    ] = 0.05,
    exogenous_prefix: Annotated[
        Optional[str],
        typer.Option(
            "--exogenous-prefix",
            help="Reference-name prefix for spike-in sequences (--method spike-in)",
        ),
    ] = None,
) -> None:
    """Correct scale factors using bamnado's own between-sample normalisation.

    Runs ``bamnado bam-normalize`` directly on BAM files to get a correction
    factor per track, then divides the existing scale_factor (from
    'normalization original') by that correction — same convention as
    'normalization tmm', but the correction comes from bamnado's TMM,
    csaw-background, CPM, median-of-ratios, or spike-in estimator over the
    full BAM rather than regulonado's own dataset-restricted TMM.

    BAM files are matched to tracks by filename stem: track N's bigwig
    'sample1.bw' must have a matching 'sample1.bam' in --bam-dir.

    Run ``regulonado enrich-metadata`` afterwards to write the updated values
    into ``final_track_records`` in the metadata JSON.
    """
    import json

    import pandas as pd
    from regulonado.normalization import compute_bamnado_norm_factors, save_scale_factors

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)
    if not bam_dir.is_dir():
        typer.echo(f"BAM directory not found: {bam_dir}", err=True)
        raise typer.Exit(1)

    with metadata.open() as fh:
        meta = json.load(fh)

    dataset_dir = metadata.parent
    track_records = sorted(meta.get("final_track_records", []), key=lambda r: r["track_index"])
    if not track_records:
        typer.echo("No 'final_track_records' found in metadata.", err=True)
        raise typer.Exit(1)

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = scale_factors if scale_factors is not None else dataset_dir / f"scale_factors.{ext}"
    out_path = output if output is not None else sf_path

    if not sf_path.exists():
        typer.echo(
            f"Scale-factors file not found: {sf_path}\n"
            "Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)

    bam_paths = []
    missing = []
    for record in track_records:
        stem = Path(record["resolved_path"]).stem
        bam_path = bam_dir / f"{stem}.bam"
        if not bam_path.exists():
            missing.append(str(bam_path))
        bam_paths.append(bam_path)
    if missing:
        typer.echo(
            "Missing BAM file(s) for these tracks (expected filename stem to match the bigwig):\n"
            + "\n".join(f"  {m}" for m in missing),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Metadata : {metadata}")
    typer.echo(f"BAM dir  : {bam_dir}")
    typer.echo(f"Method   : {method}")
    typer.echo(f"Tracks   : {len(bam_paths)}")
    typer.echo("")

    norm_factors = compute_bamnado_norm_factors(
        bam_paths,
        method=method,
        bin_size_bp=bin_size_bp,
        exclude_top_percent=exclude_top_percent,
        reference_sample=reference_sample,
        logratio_trim=logratio_trim,
        sum_trim=sum_trim,
        exogenous_prefix=exogenous_prefix,
    )

    sf_df = pd.read_parquet(sf_path) if str(sf_path).endswith(".parquet") else pd.read_csv(sf_path)
    if "scale_factor" not in sf_df.columns:
        typer.echo(
            (
                "Column 'scale_factor' missing from scale-factors file.  "
                "Run 'regulonado normalization original' first."
            ),
            err=True,
        )
        raise typer.Exit(1)
    if len(sf_df) != len(norm_factors):
        typer.echo(
            f"Track count mismatch: scale-factors file has {len(sf_df)}, "
            f"bamnado returned {len(norm_factors)}.",
            err=True,
        )
        raise typer.Exit(1)

    sf_df = sf_df.sort_values("track_index").reset_index(drop=True)
    sf_df["bamnado_method"] = method
    sf_df["bamnado_norm_factor"] = norm_factors
    sf_df["scale_factor"] = sf_df["scale_factor"] / sf_df["bamnado_norm_factor"]

    typer.echo(
        f"{'Track':>5}  {'samplename':<30}  {'norm_factor':>12}  {'old_sf':>12}  {'new_sf':>12}"
    )
    for i, row in sf_df.iterrows():
        old_sf = float(row["scale_factor"]) * float(row["bamnado_norm_factor"])
        name = str(row.get("samplename", i))[:30]
        typer.echo(
            f"{int(row['track_index']):>5}  {name:<30}  {norm_factors[i]:>12.6f}  "
            f"{old_sf:>12.6f}  {float(row['scale_factor']):>12.6f}"
        )

    priority = ["track_index", "scale_factor", "bamnado_norm_factor", "clip_soft", "clip_hard"]
    rest = [c for c in sf_df.columns if c not in priority]
    sf_df = sf_df[priority + rest]

    save_scale_factors(sf_df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"\nSaved updated scale factors to {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@app.command()
def recompress_dataset(
    src: Annotated[Path, typer.Argument(help="Source saved dataset directory")],
    dst: Annotated[Path, typer.Argument(help="Destination directory")],
    level: Annotated[int, typer.Option("--level", help="ZSTD compression level")] = 3,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Parallel shard workers")] = 4,
    max_batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--max-batch-size",
            help="Split Arrow record batches into sub-batches of at most this many rows",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Remove destination directory first if it exists"),
    ] = False,
    remove_src: Annotated[
        bool,
        typer.Option("--remove-src", help="Delete source dataset after successful recompression"),
    ] = False,
) -> None:
    """Rechunk/recompress a saved Arrow DatasetDict with ZSTD IPC compression."""
    from regulonado.recompress import recompress_dataset as _recompress_dataset

    try:
        _recompress_dataset(
            src,
            dst,
            level=level,
            workers=workers,
            max_batch_size=max_batch_size,
            overwrite=overwrite,
            remove_src=remove_src,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


@app.command()
def enrich_metadata(
    metadata: Annotated[
        Path, typer.Argument(help="Source regulonado_metadata.json")
    ],
    scale_factors: Annotated[
        Path,
        typer.Argument(help="Parquet or CSV produced by a normalization command"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for the enriched metadata JSON"),
    ],
    fields: Annotated[
        Optional[list[str]],
        typer.Option(
            "--field",
            "-f",
            help=(
                "Field to copy into final_track_records "
                "(repeat; default: all of scale_factor clip_soft clip_hard)"
            ),
        ),
    ] = None,
) -> None:
    """Create training metadata containing scale and clipping values."""
    import json

    import pandas as pd

    fields_to_copy = list(fields) if fields else ["scale_factor", "clip_soft", "clip_hard"]

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)
    if not scale_factors.exists():
        typer.echo(f"Scale-factors file not found: {scale_factors}", err=True)
        raise typer.Exit(1)
    if output.resolve() == metadata.resolve():
        raise typer.BadParameter(
            "Output must differ from the source metadata file", param_hint="OUTPUT"
        )

    sf_df = (
        pd.read_parquet(scale_factors)
        if str(scale_factors).endswith(".parquet")
        else pd.read_csv(scale_factors)
    )

    missing = [f for f in fields_to_copy if f not in sf_df.columns]
    if missing:
        typer.echo(f"Fields missing from scale-factors file: {missing}", err=True)
        raise typer.Exit(1)
    if "track_index" not in sf_df.columns:
        typer.echo("Scale-factors file has no track_index column", err=True)
        raise typer.Exit(1)
    if sf_df["track_index"].duplicated().any():
        typer.echo("Scale-factors file contains duplicate track_index values", err=True)
        raise typer.Exit(1)

    sf_by_idx: dict[int, dict] = {
        int(row["track_index"]): {f: row[f] for f in fields_to_copy} for _, row in sf_df.iterrows()
    }

    with metadata.open() as fh:
        meta = json.load(fh)

    records = meta.get("final_track_records", [])
    unmatched = [
        int(record["track_index"])
        for record in records
        if int(record["track_index"]) not in sf_by_idx
    ]
    if unmatched:
        typer.echo(
            f"No scale factors found for track_index values: {unmatched[:10]}", err=True
        )
        raise typer.Exit(1)
    updated = 0
    for record in records:
        idx = int(record["track_index"])
        if idx in sf_by_idx:
            record.update({k: float(v) for k, v in sf_by_idx[idx].items()})
            updated += 1

    meta["final_track_records"] = records
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        json.dump(meta, fh, indent=2)

    typer.echo(f"Wrote {output} with {updated}/{len(records)} updated track records")
    typer.echo(f"Fields written: {fields_to_copy}")


@app.command()
def build(
    bed_file: Annotated[Path, typer.Argument(help="BED file; column 4 used as fold label")],
    fasta_file: Annotated[
        Path, typer.Argument(help="Reference genome FASTA (.fai index required)")
    ],
    output_dir: Annotated[Path, typer.Argument(help="Output directory for the Arrow DatasetDict")],
    bigwig: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--bigwig",
            "-b",
            help="BigWig file (repeat for each track, order is preserved)",
        ),
    ] = None,
    bigwig_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--bigwig-dir",
            help="Directory of BigWig files (sorted by name, alternative to --bigwig)",
        ),
    ] = None,
    bigwig_glob: Annotated[
        Optional[list[str]],
        typer.Option(
            "--bigwig-glob",
            help=(
                "Glob when using --bigwig-dir (repeatable). "
                "Default: '*.bw' and '*.bigWig', so SeqNado output is matched too."
            ),
        ),
    ] = None,
    track_sheet: Annotated[
        Optional[Path],
        typer.Option(
            "--track-sheet",
            help=(
                "CSV mapping tracks to annotation (condition, cell_line, ip, …). "
                "Supplies the ordered track list and populates the categorical "
                "ids the training code reads. Overrides --bigwig/--bigwig-dir."
            ),
        ),
    ] = None,
    seqnado_project: Annotated[
        Optional[list[str]],
        typer.Option(
            "--seqnado-project",
            help=(
                "SeqNado output directory as PATH or NAME=PATH (repeatable). "
                "Resolves track sheet rows that give only a sample_id, and with "
                "no --track-sheet builds the sheet from the project(s) directly."
            ),
        ),
    ] = None,
    split: Annotated[
        Optional[list[str]],
        typer.Option(
            "--split",
            "-s",
            help=(
                "Split definition NAME:FOLD1,FOLD2 (repeat for each split). "
                "Default: train:fold0-2,fold5-7, validation:fold4, test:fold3."
            ),
        ),
    ] = None,
    context_length: Annotated[
        int, typer.Option("--context-length", help="Input sequence length in bp")
    ] = 524_288,
    bin_size: Annotated[int, typer.Option("--bin-size", help="Signal bin size in bp")] = 32,
    n_pred_bins: Annotated[
        int, typer.Option("--n-pred-bins", help="Number of prediction bins")
    ] = 6_144,
    shift_max_bp: Annotated[
        int,
        typer.Option(
            "--shift-max-bp",
            help="Shift augmentation buffer on each side in bp (multiple of bin-size)",
        ),
    ] = 0,
    io_threads: Annotated[
        int,
        typer.Option("--io-threads", help="Threads for parallel BigWig reads per sample"),
    ] = 8,
    num_proc: Annotated[
        int, typer.Option("--num-proc", "-j", help="Parallel worker processes for generation")
    ] = 1,
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            "--cache-dir",
            help="Arrow cache dir (default: $SLURM_TMPDIR/hf_cache or /tmp/hf_cache)",
        ),
    ] = None,
    writer_batch_size: Annotated[
        int,
        typer.Option(
            "--writer-batch-size",
            help="Samples buffered before flushing an Arrow shard to scratch",
        ),
    ] = 500,
    stage: Annotated[
        bool,
        typer.Option(
            "--stage/--no-stage", help="Copy FASTA and BigWigs to scratch before building"
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Regenerate splits that already exist")
    ] = False,
    drop_missing: Annotated[
        bool,
        typer.Option(
            "--drop-missing",
            help="Drop missing BigWig paths instead of raising an error",
        ),
    ] = False,
    dedupe_tracks: Annotated[
        str,
        typer.Option(
            "--dedupe-tracks",
            help="Track deduplication mode: none, identity, or content",
        ),
    ] = "none",
    profile: Annotated[
        bool,
        typer.Option(
            "--profile/--no-profile",
            help="Emit per-phase timing summary to stderr after each split (fast path only)",
        ),
    ] = False,
    n_extract_threads: Annotated[
        int,
        typer.Option(
            "--n-extract-threads",
            help="Rayon thread count for Phase 1 Rust BigWig extraction (fast path only)",
        ),
    ] = 32,
    signal_sample_chunk: Annotated[
        int,
        typer.Option(
            "--signal-sample-chunk",
            help="Samples per block when transposing extracted signals for Arrow writing",
        ),
    ] = 8,
    signal_track_chunk: Annotated[
        int,
        typer.Option(
            "--signal-track-chunk",
            help="Tracks per block when transposing extracted signals for Arrow writing",
        ),
    ] = 128,
    arrow_batch_size: Annotated[
        int,
        typer.Option(
            "--arrow-batch-size",
            help="Samples per Rust-written Arrow record batch (RAM-bounded)",
        ),
    ] = 8,
    shard_target_mb: Annotated[
        int,
        typer.Option(
            "--shard-target-mb",
            help=(
                "Target on-disk size per Arrow shard file (chrom_pass). Shard "
                "sample count is derived from this and the schema; raise it for "
                "fewer/larger shards, lower it for more/smaller ones."
            ),
        ),
    ] = 256,
    shard_size: Annotated[
        Optional[int],
        typer.Option(
            "--shard-size",
            help=(
                "Explicit samples per Arrow shard file (chrom_pass); overrides "
                "--shard-target-mb when set."
            ),
        ),
    ] = None,
    arrow_compression: Annotated[
        str,
        typer.Option(
            "--arrow-compression",
            help="Arrow IPC compression: zstd, lz4, or none",
        ),
    ] = "lz4",
    arrow_write_threads: Annotated[
        Optional[int],
        typer.Option(
            "--arrow-write-threads",
            help=(
                "Concurrent Arrow shard writers for chrom_pass. Defaults to "
                "min(8, --n-extract-threads); lower this if memory is tight."
            ),
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=(
                "Build strategy: 'chrom_pass' (default, one shard per "
                "chromosome, ~10× fewer BigWig seeks) or 'fast' (sample-"
                "batched, single shard per split)."
            ),
        ),
    ] = "chrom_pass",
    chrom: Annotated[
        Optional[list[str]],
        typer.Option(
            "--chrom",
            help=(
                "Restrict each split to BED rows on this chromosome "
                "(repeat for several). The `index` column on every output "
                "row still refers to the absolute row in the full BED."
            ),
        ),
    ] = None,
) -> None:
    """Build an Arrow DatasetDict from BED / FASTA / BigWig sources.

    \b
    Examples
    --------
    # Two splits, 16 workers, shift aug, staged from Ceph scratch:
    regulonado build intervals.bed genome.fa out/ \\
        --bigwig-dir bw/ \\
        --split train:train --split validation:valid \\
        --shift-max-bp 128 --num-proc 16 --stage

    # Explicit ordered BigWig list, no fold filtering:
    regulonado build intervals.bed genome.fa out/ \\
        --bigwig plus.bw --bigwig minus.bw
    """
    from regulonado.dataset import DEFAULT_SPLITS, build_dataset_fast

    projects = _parse_seqnado_projects(seqnado_project)

    # --- resolve BigWig paths and annotation ---------------------------------
    annotations: Optional[dict] = None
    vocab: Optional[dict] = None

    if track_sheet is not None or projects:
        from regulonado.tracks import TrackSheet

        try:
            if track_sheet is not None:
                sheet = TrackSheet.from_csv(track_sheet, projects=projects or None)
            else:
                sheet = TrackSheet.from_seqnado_projects(
                    [{"name": name, "path": path} for name, path in projects.items()]
                )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

        bw_paths: list[str] = [str(p) for p in sheet.bigwig_paths]
        annotations = sheet.annotations_by_path()
        _, vocab = sheet.to_track_records()
    elif bigwig_dir is not None:
        # Default covers both extensions: ReguloNado writes '.bw', SeqNado '.bigWig'.
        globs = list(bigwig_glob) if bigwig_glob else ["*.bw", "*.bigWig"]
        matched = {p for pattern in globs for p in bigwig_dir.glob(pattern)}
        bw_paths = [str(p) for p in sorted(matched)]
        if not bw_paths:
            typer.echo(f"No files matching {globs} in {bigwig_dir}", err=True)
            raise typer.Exit(1)
    elif bigwig:
        bw_paths = [str(p) for p in bigwig]
    else:
        typer.echo(
            "Provide --bigwig files, --bigwig-dir, --track-sheet or --seqnado-project.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Tracks : {len(bw_paths)}")

    # --- parse --split NAME:FOLD1,FOLD2 --------------------------------------
    splits: dict[str, list[str]] = {}
    for s in split or []:
        if ":" not in s:
            typer.echo(f"Invalid --split '{s}' — expected NAME:FOLD1,FOLD2", err=True)
            raise typer.Exit(1)
        name, _, folds_str = s.partition(":")
        splits[name] = [f.strip() for f in folds_str.split(",") if f.strip()]

    if not splits:
        splits = DEFAULT_SPLITS

    typer.echo(f"Splits : {list(splits)}")
    typer.echo(f"Output : {output_dir}")

    build_dataset_fast(
        bed_file=bed_file,
        fasta_file=fasta_file,
        bigwig_paths=bw_paths,
        output_dir=output_dir,
        splits=splits,
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        n_extract_threads=n_extract_threads,
        signal_sample_chunk=signal_sample_chunk,
        signal_track_chunk=signal_track_chunk,
        arrow_batch_size=arrow_batch_size,
        shard_size=shard_size,
        shard_target_mb=shard_target_mb,
        arrow_compression=arrow_compression,
        arrow_write_threads=arrow_write_threads,
        num_proc=num_proc,
        cache_dir=cache_dir,
        writer_batch_size=writer_batch_size,
        stage_to_scratch=stage,
        overwrite=overwrite,
        drop_missing=drop_missing,
        dedupe_tracks=dedupe_tracks,
        annotations=annotations,
        track_metadata_vocab=vocab,
        profile=profile,
        strategy=strategy,
        chrom_filter=list(chrom) if chrom else None,
        return_dataset=False,
    )


@app.command()
def predict(
    checkpoint_dir: Annotated[
        Path,
        typer.Argument(
            help="HF RegulonadoModel directory; legacy run roots supported with --dataset"
        ),
    ],
    fasta_file: Annotated[
        Path,
        typer.Argument(help="Genome FASTA (needs a .fai index) used to build model inputs"),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to write one BigWig per selected track"),
    ],
    dataset: Annotated[
        Optional[Path],
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset dir with regulonado_metadata.json; required only for legacy checkpoints",
        ),
    ] = None,
    bed_file: Annotated[
        Optional[Path],
        typer.Option("--bed", help="Predict the central window centred on each BED row"),
    ] = None,
    whole_genome: Annotated[
        bool,
        typer.Option("--whole-genome", help="Tile each chromosome into adjacent windows instead"),
    ] = False,
    chromsizes: Annotated[
        Optional[Path],
        typer.Option(
            "--chromsizes",
            help="chrom<TAB>size file selecting which chromosomes to tile and their lengths "
            "(also used for the BigWig header; default: all FASTA contigs via .fai)",
        ),
    ] = None,
    tracks: Annotated[
        Optional[str],
        typer.Option(
            "--tracks",
            help="Comma-separated track names or indices to write (default: all tracks)",
        ),
    ] = None,
    rtol: Annotated[
        float,
        typer.Option("--rtol", help="Relative tolerance for collapsing adjacent equal bins"),
    ] = 0.01,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Windows per forward pass"),
    ] = 4,
    device: Annotated[
        Optional[str],
        typer.Option("--device", help="Torch device (default: cuda if available else cpu)"),
    ] = None,
    inverse_squash: Annotated[
        bool,
        typer.Option(
            "--inverse-squash",
            help="Undo the (x+1)^0.75 training squash so values approximate raw counts",
        ),
    ] = False,
) -> None:
    """Generate BigWig tracks from trained model predictions, collapsing equal bins.

    Provide exactly one of --bed (targeted regions) or --whole-genome (tiling). Only the central
    prediction window is written, so windows never overlap. Note: --whole-genome with all tracks
    produces many large files; subset with --tracks.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    from regulonado.inference import predict_to_bigwig

    if bool(bed_file) == whole_genome:
        typer.echo("Provide exactly one of --bed or --whole-genome", err=True)
        raise typer.Exit(2)

    written = predict_to_bigwig(
        checkpoint_dir,
        fasta_file,
        output_dir,
        dataset_dir=dataset,
        bed_path=bed_file,
        whole_genome=whole_genome,
        chromsizes_path=chromsizes,
        tracks=[t.strip() for t in tracks.split(",")] if tracks else None,
        rtol=rtol,
        batch_size=batch_size,
        device=device,
        inverse_squash=inverse_squash,
    )
    typer.echo(f"Wrote {len(written)} BigWig(s) to {output_dir}")
    for path in written:
        typer.echo(f"  {path}")


def _trajectory_table(wandb_module, history: list[dict]):
    """Build a wandb.Table of the per-round trajectory, one row per round.

    Column set is the union across rounds (round 0 and non-improving rounds lack "positions";
    AdaLead rounds lack "n_edits"), so every row gets every column, blank where not recorded.
    """
    fixed = [
        "round",
        "sequence",
        "energy",
        "specificity",
        "target",
        "target_gain",
        "offtarget_boost",
        "n_edits",
    ]
    dynamic = sorted(
        {key for entry in history for key in entry if key not in (*fixed, "positions")}
    )
    columns = [*fixed, "positions", *dynamic]
    rows = []
    for entry in history:
        row = []
        for column in columns:
            value = entry.get(column, "")
            if column == "positions" and value:
                value = ",".join(str(p) for p in value)
            row.append(value)
        rows.append(row)
    return wandb_module.Table(columns=columns, data=rows)


@app.command()
def design(
    candidates: Annotated[
        Path, typer.Option("--candidates", help="BED of enhancer candidates to optimise.")
    ],
    checkpoint: Annotated[
        list[Path],
        typer.Option(
            "--checkpoint", help="Design-fold checkpoint dir; repeat once per fold optimised."
        ),
    ],
    fasta_file: Annotated[
        Path, typer.Option("--fasta", help="Genome FASTA (needs a .fai index).")
    ],
    target: Annotated[
        str, typer.Option("--target", help="Target group value in the --group-by column.")
    ],
    out_dir: Annotated[Path, typer.Option("--out", help="Directory to write design outputs.")],
    intervals: Annotated[
        Optional[Path],
        typer.Option(
            "--intervals",
            help="Build-time interval BED the folds were trained on; default: the 'bed_file' "
            "recorded in --dataset-dir's regulonado_metadata.json.",
        ),
    ] = None,
    holdout_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--holdout-checkpoint", help="Held-out fold checkpoint: scored but never optimised."
        ),
    ] = None,
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option("--dataset-dir", help="Dataset dir with regulonado_metadata.json."),
    ] = None,
    group_by: Annotated[
        str, typer.Option("--group-by", help="Track annotation column defining cell-type groups.")
    ] = "source",
    track_sheet: Annotated[
        Optional[Path],
        typer.Option("--track-sheet", help="CSV mapping tracks to their --group-by annotation."),
    ] = None,
    exclude_track: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude-track",
            help="Exact model track name to exclude; repeat for multiple tracks.",
        ),
    ] = None,
    method: Annotated[str, typer.Option("--method", help="'ism' or 'adalead'.")] = "ism",
    rounds: Annotated[int, typer.Option("--rounds", help="Search rounds.")] = 20,
    top_k: Annotated[
        int, typer.Option("--top-k", help="ISM: substitutions accepted per round.")
    ] = 1,
    pad: Annotated[
        int,
        typer.Option("--pad", help="Widen each candidate's editable span by N bp on both sides."),
    ] = 0,
    ism_stride: Annotated[
        int, typer.Option("--ism-stride", help="ISM: only test every Nth editable position.")
    ] = 1,
    ism_positions: Annotated[
        Optional[Path],
        typer.Option(
            "--ism-positions", help="ISM: BED restricting the search to these positions."
        ),
    ] = None,
    population_size: Annotated[
        int, typer.Option("--population-size", help="AdaLead: population size.")
    ] = 20,
    model_queries_per_batch: Annotated[
        Optional[int],
        typer.Option(
            "--model-queries-per-batch", help="AdaLead oracle query budget per round."
        ),
    ] = None,
    top_n: Annotated[
        int, typer.Option("--top-n", help="Number of final candidates in the artifact.")
    ] = 10,
    mu: Annotated[float, typer.Option("--mu", help="AdaLead: per-base mutation rate scale.")] = 1.0,
    recomb_rate: Annotated[
        float, typer.Option("--recomb-rate", help="AdaLead: per-base recombination rate.")
    ] = 0.1,
    threshold: Annotated[
        float, typer.Option("--threshold", help="AdaLead: parent fitness threshold.")
    ] = 0.1,
    rho: Annotated[
        int, typer.Option("--rho", help="AdaLead: recombination cycles per proposal round.")
    ] = 2,
    on_missing: Annotated[
        str,
        typer.Option("--on-missing", help="'error', 'center' or 'skip' for unmatched candidates."),
    ] = "error",
    offtarget_reduction: Annotated[
        str, typer.Option("--offtarget-reduction", help="'logsumexp', 'max' or 'mean'.")
    ] = "logsumexp",
    objective: Annotated[
        str,
        typer.Option(
            "--objective", help="'specificity' or seed-relative 'selective-activation'."
        ),
    ] = "specificity",
    offtarget_boost_weight: Annotated[
        float,
        typer.Option(
            "--offtarget-boost-weight",
            help="Selective activation: weight applied to positive off-target gain.",
        ),
    ] = 1.0,
    offtarget_boost_tolerance: Annotated[
        float,
        typer.Option(
            "--offtarget-boost-tolerance",
            help="Selective activation: off-target gain ignored before applying the penalty.",
        ),
    ] = 0.0,
    offtarget_temperature: Annotated[
        float,
        typer.Option(
            "--offtarget-temperature", help="Temperature for soft off-target aggregation."
        ),
    ] = 1.0,
    target_alpha: Annotated[
        float, typer.Option("--target-alpha", help="Scale applied to the on-target score.")
    ] = 1.0,
    bending_factor: Annotated[
        float, typer.Option("--bending-factor", help="Bending transform strength.")
    ] = 0.0,
    bin_reduction: Annotated[
        str, typer.Option("--bin-reduction", help="Bin statistic: mean or topk.")
    ] = "mean",
    topk_bins: Annotated[
        int, typer.Option("--topk-bins", help="Number of bins in top-K statistic.")
    ] = 10,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed for AdaLead.")] = None,
    fold_mode: Annotated[
        str, typer.Option("--fold-mode", help="'resident' or 'sequential' fold residency.")
    ] = "resident",
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Forward-pass batch size.")
    ] = 8,
    device: Annotated[
        Optional[str], typer.Option("--device", help="Torch device (default: cuda if available).")
    ] = None,
    wandb: Annotated[
        bool,
        typer.Option("--wandb/--no-wandb", help="Log per-round specificity to Weights & Biases."),
    ] = False,
    wandb_project: Annotated[
        str, typer.Option("--wandb-project", help="W&B project (one run per candidate).")
    ] = "regulonado-design",
    wandb_group: Annotated[
        Optional[str],
        typer.Option("--wandb-group", help="W&B group label for this design invocation."),
    ] = None,
) -> None:
    """Mutate endogenous enhancer candidates to sharpen cell-type specificity.

    Optimises exactly the spans in --candidates; context for each comes from the dataset window
    (from --intervals) that contains it. Provide 3 --checkpoint folds to optimise against and,
    ideally, a 4th --holdout-checkpoint to confirm the design isn't fold-specific overfitting.
    """
    import json
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    if method not in ("ism", "adalead"):
        raise typer.BadParameter("Expected 'ism' or 'adalead'", param_hint="--method")
    if on_missing not in ("error", "center", "skip"):
        raise typer.BadParameter("Expected 'error', 'center' or 'skip'", param_hint="--on-missing")
    if fold_mode not in ("resident", "sequential"):
        raise typer.BadParameter("Expected 'resident' or 'sequential'", param_hint="--fold-mode")
    if bin_reduction not in ("mean", "topk"):
        raise typer.BadParameter("Expected 'mean' or 'topk'", param_hint="--bin-reduction")
    if objective not in ("specificity", "selective-activation"):
        raise typer.BadParameter(
            "Expected 'specificity' or 'selective-activation'", param_hint="--objective"
        )
    if offtarget_boost_weight < 0:
        raise typer.BadParameter("Must be >= 0", param_hint="--offtarget-boost-weight")
    if offtarget_boost_tolerance < 0:
        raise typer.BadParameter("Must be >= 0", param_hint="--offtarget-boost-tolerance")
    if offtarget_temperature <= 0:
        raise typer.BadParameter("Must be > 0", param_hint="--offtarget-temperature")
    if not checkpoint:
        raise typer.BadParameter("Provide at least one --checkpoint", param_hint="--checkpoint")
    import pyfaidx
    from regulonado.design.objective import SpecificityEnergy, resolve_track_groups
    from regulonado.design.predictor import FoldEnsemble, FoldSpec
    from regulonado.design.report import DesignRecord, write_designs
    from regulonado.design.search import AdaLeadConfig, adalead, ism_greedy
    from regulonado.design.sequence import DatasetWindowIndex, resolve_seeds
    from regulonado.inference import _parse_bed, one_hot_context

    # Seed all relevant libraries once.  Candidate-specific generators below are
    # derived from this value and never rely on global NumPy state.
    run_seed = 0 if seed is None else int(seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    if intervals is None:
        if dataset_dir is None:
            raise typer.BadParameter(
                "Provide --intervals, or --dataset-dir with a regulonado_metadata.json "
                "recording 'bed_file'",
                param_hint="--intervals",
            )
        metadata = json.loads((dataset_dir / "regulonado_metadata.json").read_text())
        bed_file = metadata.get("bed_file")
        if not bed_file:
            raise typer.BadParameter(
                f"No 'bed_file' recorded in {dataset_dir / 'regulonado_metadata.json'}",
                param_hint="--intervals",
            )
        intervals = Path(bed_file)

    typer.echo(f"Loading {len(checkpoint)} design fold(s)...")
    design_folds = [FoldSpec(checkpoint_dir=c, dataset_dir=dataset_dir) for c in checkpoint]
    ensemble = FoldEnsemble(design_folds, device=device, batch_size=batch_size, mode=fold_mode)

    holdout_ensemble = None
    if holdout_checkpoint is not None:
        holdout_ensemble = FoldEnsemble(
            [FoldSpec(checkpoint_dir=holdout_checkpoint, dataset_dir=dataset_dir)],
            device=device,
            batch_size=batch_size,
            mode=fold_mode,
        )
        holdout_geometry = (
            holdout_ensemble.context_length,
            holdout_ensemble.n_pred_bins,
            holdout_ensemble.bin_size,
        )
        design_geometry = (ensemble.context_length, ensemble.n_pred_bins, ensemble.bin_size)
        if holdout_geometry != design_geometry:
            raise typer.BadParameter(
                f"Held-out fold geometry {holdout_geometry} does not match design fold "
                f"geometry {design_geometry}",
                param_hint="--holdout-checkpoint",
            )

    index = DatasetWindowIndex.from_bed(
        intervals,
        context_length=ensemble.context_length,
        n_pred_bins=ensemble.n_pred_bins,
        bin_size=ensemble.bin_size,
    )
    seeds = resolve_seeds(candidates, index, on_missing=on_missing, pad=pad)
    typer.echo(f"Resolved {len(seeds)} candidate(s) against dataset windows")

    groups = resolve_track_groups(
        ensemble.track_names,
        group_by=group_by,
        target=target,
        exclude_tracks=set(exclude_track or ()),
        track_sheet=track_sheet,
        dataset_dir=dataset_dir,
    )
    typer.echo(f"Target group {target!r}; {len(groups.other_group_masks)} off-target group(s)")

    fasta = pyfaidx.Fasta(str(fasta_file), as_raw=True, sequence_always_upper=False)
    chrom_sizes = {name: len(fasta[name]) for name in fasta.keys()}

    ism_hits: list[tuple[str, int, int]] | None = None
    if ism_positions is not None:
        ism_hits = _parse_bed(ism_positions)

    if method == "ism":
        total_positions = sum(
            len(range(seed.editable.start, seed.editable.stop, ism_stride)) for seed in seeds
        )
        typer.echo(
            f"Projected ISM forward passes: ~{total_positions * 3 * rounds * len(checkpoint):,} "
            f"({total_positions} position(s) x 3 alt bases x {rounds} round(s) x "
            f"{len(checkpoint)} fold(s))"
        )

    wandb_run = None
    if wandb:
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise typer.BadParameter(
                "--wandb requires the 'wandb' package (part of regulonado[train]).",
                param_hint="--wandb",
            ) from exc

    records: list[DesignRecord] = []
    seed_report: list[dict] = []
    for candidate_index, candidate_seed in enumerate(seeds, start=1):
        logger.info(
            f"[{candidate_index}/{len(seeds)}] {candidate_seed.name} "
            f"({candidate_seed.chrom}:{candidate_seed.cand_start}-"
            f"{candidate_seed.cand_end}), method={method}"
        )
        seed = candidate_seed
        chrom_length = chrom_sizes.get(seed.chrom)
        if chrom_length is None:
            raise typer.BadParameter(f"Chromosome {seed.chrom!r} not present in {fasta_file}")
        context = one_hot_context(fasta, seed.window, ensemble.context_length, chrom_length)

        energy_fn = SpecificityEnergy(
            ensemble,
            groups,
            seed.bins,
            target_alpha=target_alpha,
            bending_factor=bending_factor,
            offtarget_reduction=offtarget_reduction,
            offtarget_temperature=offtarget_temperature,
            bin_reduction=bin_reduction,
            topk_bins=topk_bins,
            objective=objective,
            offtarget_boost_weight=offtarget_boost_weight,
            offtarget_boost_tolerance=offtarget_boost_tolerance,
        )
        if objective == "selective-activation":
            energy_fn.set_reference(context[None])

        if wandb:
            wandb_run = _wandb.init(
                project=wandb_project,
                group=wandb_group or target,
                job_type=method,
                name=f"{seed.name}_{method}",
                reinit=True,
                config={
                    "candidate": seed.name,
                    "chrom": seed.chrom,
                    "start": seed.cand_start,
                    "end": seed.cand_end,
                    "fold_label": seed.fold_label,
                    "method": method,
                    "objective": objective,
                    "rounds": rounds,
                    "target": target,
                    "group_by": group_by,
                    "wandb_group": wandb_group or target,
                },
            )

        wandb_history: list[dict] = []

        def _on_round(entry: dict, seed_name: str = seed.name, run=wandb_run) -> None:
            summary = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in entry.items()
                if key not in ("round", "positions", "sequence")
            )
            logger.info(f"  [{seed_name}] round {entry['round']}: {summary}")
            if run is not None:
                # Log a new immutable table snapshot every round.  This keeps
                # candidate sequences and their score breakdowns durable even
                # if a later candidate fails or the job is interrupted.
                wandb_history.append(dict(entry))
                table = _trajectory_table(_wandb, wandb_history)
                scalars = {k: v for k, v in entry.items() if k not in ("positions", "sequence")}
                scalars.update({
                    f"track/{name}": entry[f"track_{index}"]
                    for index, name in enumerate(ensemble.track_names)
                    if f"track_{index}" in entry
                })
                run.log({**scalars, "candidate_results": table}, step=entry["round"])

        if method == "ism":
            positions = None
            if ism_hits is not None:
                positions = [
                    hit_start + offset - seed.window.ctx_start
                    for hit_chrom, hit_start, hit_end in ism_hits
                    if hit_chrom == seed.chrom
                    for offset in range(hit_end - hit_start)
                ]
            state = ism_greedy(
                energy_fn,
                seed,
                context,
                rounds=rounds,
                top_k=top_k,
                positions=positions,
                stride=ism_stride,
                batch_size=batch_size,
                rng=np.random.default_rng(np.random.SeedSequence([run_seed, candidate_index])),
                on_round=_on_round,
            )
        else:
            adalead_config = AdaLeadConfig(
                rounds=rounds, population_size=population_size, mu=mu, recomb_rate=recomb_rate,
                model_queries_per_batch=model_queries_per_batch,
                threshold=threshold,
                rho=rho,
            )
            state = adalead(
                energy_fn,
                seed,
                context,
                adalead_config,
                rng=np.random.default_rng(
                    np.random.SeedSequence([run_seed, candidate_index])
                ),
                on_round=_on_round,
            )

        final_result = energy_fn(state.context[None])

        holdout_result = None
        if holdout_ensemble is not None:
            holdout_energy_fn = SpecificityEnergy(
                holdout_ensemble,
                groups,
                seed.bins,
                target_alpha=target_alpha,
                bending_factor=bending_factor,
                offtarget_reduction=offtarget_reduction,
                offtarget_temperature=offtarget_temperature,
                bin_reduction=bin_reduction,
                topk_bins=topk_bins,
                objective=objective,
                offtarget_boost_weight=offtarget_boost_weight,
                offtarget_boost_tolerance=offtarget_boost_tolerance,
            )
            if objective == "selective-activation":
                holdout_energy_fn.set_reference(context[None])
            holdout_result = holdout_energy_fn(state.context[None])

        baseline_energy = state.history[0]["energy"]
        holdout_suffix = (
            f", holdout={float(holdout_result.energy[0]):.4f}" if holdout_result is not None else ""
        )
        logger.success(
            f"[{candidate_index}/{len(seeds)}] {seed.name} done: energy {baseline_energy:.4f} -> "
            f"{state.energy:.4f}{holdout_suffix}"
        )
        if wandb_run is not None:
            wandb_run.summary["final_energy"] = state.energy
            wandb_run.summary["baseline_energy"] = baseline_energy
            wandb_run.summary["final_specificity"] = float(final_result.specificity[0])
            wandb_run.summary["final_target_gain"] = float(final_result.target_gain[0])
            wandb_run.summary["final_offtarget_boost"] = float(
                final_result.offtarget_boost[0]
            )
            if holdout_result is not None:
                wandb_run.summary["holdout_energy"] = float(holdout_result.energy[0])
                wandb_run.summary["holdout_target"] = float(holdout_result.target[0])
                wandb_run.summary["holdout_target_gain"] = float(
                    holdout_result.target_gain[0]
                )
                wandb_run.summary["holdout_offtarget_boost"] = float(
                    holdout_result.offtarget_boost[0]
                )
            # Keep the final complete snapshot under a stable key as well.
            wandb_run.log({"trajectory": _trajectory_table(_wandb, state.history),
                           "candidate_results": _trajectory_table(_wandb, state.history)},
                          step=len(state.history) - 1)
            wandb_run.finish()

        records.append(
            DesignRecord(
                seed=seed,
                method=method,
                original_context=context,
                state=state,
                result=final_result,
                holdout_result=holdout_result,
            )
        )
        # Persist completed candidates immediately so an interrupted shard can
        # be resumed/recovered without losing earlier oracle work.
        write_designs(
            out_dir,
            records,
            run_info={"status": "in_progress", "completed_candidates": len(records)},
        )
        seed_report.append(
            {
                "name": seed.name,
                "chrom": seed.chrom,
                "start": seed.cand_start,
                "end": seed.cand_end,
                "window_start": seed.window.ctx_start,
                "window_end": seed.window.ctx_end,
                "fold_label": seed.fold_label,
            }
        )

    run_info = {
        "candidates": str(candidates),
        "intervals": str(intervals),
        "fasta": str(fasta_file),
        "checkpoints": [str(c) for c in checkpoint],
        "holdout_checkpoint": str(holdout_checkpoint) if holdout_checkpoint else None,
        "target": target,
        "objective": objective,
        "group_by": group_by,
        "method": method,
        "rounds": rounds,
        "pad": pad,
        "on_missing": on_missing,
        "fold_mode": fold_mode,
        "track_groups": {"target": groups.target, "labels": groups.labels},
        "excluded_tracks": sorted(exclude_track or ()),
        "seed_resolution": seed_report,
        "seed": run_seed,
        "topk_bins": topk_bins,
        "bin_reduction": bin_reduction,
        "target_alpha": target_alpha,
        "bending_factor": bending_factor,
        "offtarget_reduction": offtarget_reduction,
        "offtarget_temperature": offtarget_temperature,
        "offtarget_boost_weight": offtarget_boost_weight,
        "offtarget_boost_tolerance": offtarget_boost_tolerance,
        "batch_size": batch_size,
        "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "track_names": ensemble.track_names,
        "reproducibility": {
            "torch_deterministic": bool(torch.are_deterministic_algorithms_enabled())
        },
        "status": "complete",
    }
    write_designs(out_dir, records, run_info=run_info)
    typer.echo(f"Wrote {len(records)} design(s) to {out_dir}")


def main() -> None:
    app()


app.add_typer(normalization_app, name="normalization")
app.command("pipeline")(_pipeline)


if __name__ == "__main__":
    main()
