from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from pydantic import ValidationError


def _load_params_mapping(params_path: Optional[Path]) -> dict[str, Any]:
    """Load a ``--params`` file (YAML or JSON; YAML parses both) into a plain mapping."""
    if params_path is None:
        return {}
    import yaml

    data = yaml.safe_load(params_path.read_text()) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("--params file must contain a mapping", param_hint="--params")
    return data


def _apply_cli_overrides(
    data: dict[str, Any],
    *,
    candidates: Optional[Path],
    checkpoint: Optional[list[Path]],
    fasta_file: Optional[Path],
    out_dir: Optional[Path],
    intervals: Optional[Path],
    dataset_dir: Optional[Path],
    track: Optional[str],
) -> dict[str, Any]:
    """Layer explicit CLI flags onto a ``--params``-loaded mapping; CLI flags win."""
    overrides = {
        "candidates": str(candidates) if candidates else None,
        "checkpoint_dirs": [str(c) for c in checkpoint] if checkpoint else None,
        "fasta": str(fasta_file) if fasta_file else None,
        "out_dir": str(out_dir) if out_dir else None,
        "intervals": str(intervals) if intervals else None,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if track is not None:
        targets = data.get("targets")
        if not targets:
            data["targets"] = [{"name": "cli", "track": track}]
        else:
            targets[0]["track"] = track
    return data


def attribute(
    params: Annotated[
        Optional[Path],
        typer.Option(
            "--params",
            help="YAML/JSON file parsed as AttributionConfig; explicit options below override it.",
        ),
    ] = None,
    candidates: Annotated[
        Optional[Path], typer.Option("--candidates", help="BED of candidate regions to scan.")
    ] = None,
    checkpoint: Annotated[
        Optional[list[Path]],
        typer.Option("--checkpoint", help="Fold checkpoint dir; repeat once per fold."),
    ] = None,
    fasta_file: Annotated[
        Optional[Path], typer.Option("--fasta", help="Genome FASTA (needs a .fai index).")
    ] = None,
    track: Annotated[
        Optional[str], typer.Option("--track", help="Track to attribute against: name or index.")
    ] = None,
    out_dir: Annotated[
        Optional[Path], typer.Option("--out", help="Directory to write attribution outputs.")
    ] = None,
    intervals: Annotated[
        Optional[Path],
        typer.Option(
            "--intervals",
            help="Build-time interval BED the folds were trained on; default: the 'bed_file' "
            "recorded in --dataset-dir's tracks.parquet.",
        ),
    ] = None,
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option("--dataset-dir", help="Dataset dir with tracks.parquet."),
    ] = None,
) -> None:
    """Locate the high-attribution core of each candidate by in-silico mutagenesis.

    Scores every alternative base at every position against one output track, then calls the
    contiguous sub-span that drives it — in practice the nucleosome-free core. The resulting
    core_regions.bed is designed to be fed straight back in as `regulonado design --candidates`,
    so that design and synthesis target only the span that matters.

    Provide `--params config.yaml` for the full set of ISM-sweep tuning options (bin/fold
    reduction, smoothing, thresholds, ...); the options above override anything it sets.
    """
    from regulonado.config.models import AttributionConfig
    from regulonado.design.attribute_run import run_attribution

    data = _apply_cli_overrides(
        _load_params_mapping(params),
        candidates=candidates,
        checkpoint=checkpoint,
        fasta_file=fasta_file,
        out_dir=out_dir,
        intervals=intervals,
        dataset_dir=dataset_dir,
        track=track,
    )

    try:
        config = AttributionConfig.model_validate(data)
        result = run_attribution(config)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(
        f"Called cores for {result.n_cores_called}/{result.n_candidates} candidate(s); "
        f"wrote {result.core_regions_bed}"
    )
