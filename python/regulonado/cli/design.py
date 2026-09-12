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
    holdout_checkpoint: Optional[Path],
    fasta_file: Optional[Path],
    intervals: Optional[Path],
    dataset_dir: Optional[Path],
    out_dir: Optional[Path],
    target: Optional[str],
    group_by: Optional[str],
    method: Optional[str],
) -> dict[str, Any]:
    """Layer explicit CLI flags onto a ``--params``-loaded mapping; CLI flags win."""
    overrides = {
        "candidates": str(candidates) if candidates else None,
        "checkpoint_dirs": [str(c) for c in checkpoint] if checkpoint else None,
        "holdout_checkpoint": str(holdout_checkpoint) if holdout_checkpoint else None,
        "fasta": str(fasta_file) if fasta_file else None,
        "intervals": str(intervals) if intervals else None,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "out_dir": str(out_dir) if out_dir else None,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if target is not None or group_by is not None or method is not None:
        targets = data.get("targets")
        if not targets:
            data["targets"] = [{"name": "cli"}]
            targets = data["targets"]
        if target is not None:
            targets[0]["target"] = target
        if group_by is not None:
            targets[0]["group_by"] = group_by
        if method is not None:
            targets[0]["method"] = method
    return data


def design(
    params: Annotated[
        Optional[Path],
        typer.Option(
            "--params",
            help="YAML/JSON file parsed as DesignConfig; explicit options below override it.",
        ),
    ] = None,
    candidates: Annotated[
        Optional[Path], typer.Option("--candidates", help="BED of enhancer candidates to optimise.")
    ] = None,
    checkpoint: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--checkpoint", help="Design-fold checkpoint dir; repeat once per fold optimised."
        ),
    ] = None,
    holdout_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--holdout-checkpoint", help="Held-out fold checkpoint: scored but never optimised."
        ),
    ] = None,
    fasta_file: Annotated[
        Optional[Path], typer.Option("--fasta", help="Genome FASTA (needs a .fai index).")
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Target group value in the --group-by column.")
    ] = None,
    group_by: Annotated[
        Optional[str],
        typer.Option("--group-by", help="Track annotation column defining cell-type groups."),
    ] = None,
    method: Annotated[Optional[str], typer.Option("--method", help="'ism' or 'adalead'.")] = None,
    out_dir: Annotated[
        Optional[Path], typer.Option("--out", help="Directory to write design outputs.")
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
        Optional[Path], typer.Option("--dataset-dir", help="Dataset dir with tracks.parquet.")
    ] = None,
) -> None:
    """Mutate endogenous enhancer candidates to sharpen cell-type specificity.

    Optimises exactly the spans in --candidates; context for each comes from the dataset window
    (from --intervals) that contains it. Provide 3 --checkpoint folds to optimise against and,
    ideally, a 4th --holdout-checkpoint to confirm the design isn't fold-specific overfitting.

    Provide `--params config.yaml` for the full set of search tuning options (rounds, AdaLead/ISM
    settings, the selective-activation objective, ...); the options above override anything it
    sets. See docs/design.md.
    """
    from regulonado.config.models import DesignConfig
    from regulonado.design.run import run_design

    data = _apply_cli_overrides(
        _load_params_mapping(params),
        candidates=candidates,
        checkpoint=checkpoint,
        holdout_checkpoint=holdout_checkpoint,
        fasta_file=fasta_file,
        intervals=intervals,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        target=target,
        group_by=group_by,
        method=method,
    )

    try:
        config = DesignConfig.model_validate(data)
        result = run_design(config)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"Wrote {result.n_candidates} design(s) to {result.out_dir}")
