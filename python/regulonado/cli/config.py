from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)


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
    logger.info(f"Wrote config -> {destination}")


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
