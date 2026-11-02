"""Standalone training and generation for numeric-conditioned enhancer diffusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

generate_app = typer.Typer(
    no_args_is_help=True, help="Train and sample small enhancer diffusion models."
)


@generate_app.command("train")
def train(
    table: Annotated[
        Path, typer.Option(exists=True, help="TSV/parquet: chrom, sequence, activity columns.")
    ],
    params: Annotated[Path, typer.Option(exists=True, help="Diffusion training YAML.")],
    out: Annotated[Path, typer.Option(help="Checkpoint directory.")],
):
    import yaml
    from regulonado.design.diffusion import DiffusionConfig, train_diffusion

    try:
        config = DiffusionConfig.model_validate(yaml.safe_load(params.read_text()))
        path = train_diffusion(table, out, config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(path))


@generate_app.command("sample")
def generate(
    checkpoint: Annotated[Path, typer.Option(exists=True)],
    activities: Annotated[
        str, typer.Option(help='JSON numeric profile, e.g. {"HL60": 2.5, "K562": 0.2}.')
    ],
    length: Annotated[int, typer.Option(min=1)],
    out: Annotated[Path, typer.Option(help="Output directory: FASTA, TSV, provenance JSON.")],
    count: Annotated[int, typer.Option(min=1)] = 32,
    steps: Annotated[int, typer.Option(min=1)] = 64,
    guidance: Annotated[float, typer.Option(min=0)] = 1.0,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    device: str = "auto",
    min_distance: Annotated[int, typer.Option(min=0)] = 0,
):
    from regulonado.design.diffusion import condition_vector, load_checkpoint, sample
    from regulonado.design.generators import diverse_sequences

    try:
        profile = json.loads(activities)
        if not isinstance(profile, dict):
            raise ValueError("activities must be a JSON object")
        model, metadata = load_checkpoint(checkpoint, device)
        values, observed = condition_vector(metadata, profile)
        sequences = sample(
            model,
            values,
            observed,
            length=length,
            count=count,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )
        selected = diverse_sequences(sequences, count=count, min_distance=min_distance)
    except (ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    (out / "sequences.fa").write_text(
        "".join(f">diffusion_{i}\n{s}\n" for i, s in enumerate(selected))
    )
    (out / "sequences.tsv").write_text(
        "name\tsequence\tgc_fraction\n"
        + "".join(
            f"diffusion_{i}\t{s}\t{(s.count('G') + s.count('C')) / len(s):.6f}\n"
            for i, s in enumerate(selected)
        )
    )
    (out / "run.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(checkpoint.resolve()),
                activities=profile,
                length=length,
                requested=count,
                retained=len(selected),
                steps=steps,
                guidance=guidance,
                seed=seed,
                min_distance=min_distance,
                activity_columns=metadata["config"]["activity_columns"],
            ),
            indent=2,
        )
    )
    typer.echo(f"Generated {len(selected)} unique sequences in {out}")


@generate_app.command("select")
def select(
    tables: Annotated[
        list[Path], typer.Option("--table", exists=True, help="Scored TSV; repeat to pool methods.")
    ],
    out: Annotated[Path, typer.Option(help="Selected FASTA and TSV directory.")],
    count: Annotated[int, typer.Option(min=1)] = 24,
    min_distance: Annotated[int, typer.Option(min=0)] = 10,
    score: str = "energy",
):
    """Select a diverse panel from scored pools (lower score is better)."""
    import numpy as np
    import pandas as pd
    from regulonado.design.generators import diverse_sequences

    frames = []
    for path in tables:
        frame = pd.read_csv(path, sep="\t")
        if "sequence" not in frame or score not in frame:
            raise typer.BadParameter(f"{path} requires sequence and {score} columns")
        frame["pool"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined[score] = pd.to_numeric(combined[score], errors="coerce")
    valid = combined.sequence.fillna("").str.fullmatch("[ACGT]+") & np.isfinite(combined[score])
    ranked = combined.loc[valid].sort_values(score, kind="stable").drop_duplicates("sequence")
    sequences = diverse_sequences(ranked.sequence, count=count, min_distance=min_distance)
    selected = ranked.set_index("sequence").loc[sequences].reset_index()
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "selected.tsv", sep="\t", index=False)
    (out / "selected.fa").write_text(
        "".join(f">selected_{i}\n{s}\n" for i, s in enumerate(sequences))
    )
    (out / "run.json").write_text(
        json.dumps(
            dict(
                tables=[str(p.resolve()) for p in tables],
                count=count,
                retained=len(sequences),
                min_distance=min_distance,
                score=score,
            ),
            indent=2,
        )
    )
    typer.echo(f"Selected {len(sequences)} sequences in {out}")


@generate_app.command("prepare")
def prepare(
    dataset_dir: Annotated[Path, typer.Option(exists=True)],
    fasta: Annotated[Path, typer.Option(exists=True)],
    params: Annotated[Path, typer.Option(exists=True)],
    out: Annotated[Path, typer.Option()],
):
    """Prepare numeric labels and chromosome-disjoint DNA from region counts."""
    import yaml
    from regulonado.config.models import DiffusionStageConfig
    from regulonado.design.diffusion_data import prepare_diffusion_data

    config = DiffusionStageConfig.model_validate(yaml.safe_load(params.read_text()))
    prepare_diffusion_data(dataset_dir, fasta, out, config)


@generate_app.command("fit")
def fit(
    table: Annotated[Path, typer.Option(exists=True)],
    manifest: Annotated[Path, typer.Option(exists=True)],
    params: Annotated[Path, typer.Option(exists=True)],
    out: Annotated[Path, typer.Option()],
    encoder_run_dir: Annotated[Path | None, typer.Option()] = None,
):
    """Train the pipeline denoiser, resuming a matching last checkpoint."""
    from regulonado.design.diffusion_pipeline import fit_pipeline

    typer.echo(str(fit_pipeline(params, table, manifest, out, encoder_run_dir)))
