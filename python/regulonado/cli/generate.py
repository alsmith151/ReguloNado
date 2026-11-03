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
    activities: Annotated[str, typer.Option(help="JSON numeric activity profile.")],
    out: Annotated[Path, typer.Option(help="FASTA, proposal records and provenance directory.")],
    length: Annotated[int | None, typer.Option(min=1)] = None,
    template_fasta: Annotated[
        Path | None,
        typer.Option(exists=True, help="Single-record A/C/G/T FASTA required for edit mode."),
    ] = None,
    mode: Annotated[str, typer.Option(help="edit (default) or de_novo.")] = "edit",
    mask_fraction: Annotated[float, typer.Option(help="Fraction masked per edit proposal.")] = 0.25,
    count: Annotated[int, typer.Option(min=1)] = 32,
    steps: Annotated[int, typer.Option(min=1)] = 64,
    guidance: Annotated[float, typer.Option(min=0)] = 1.0,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    device: str = "auto",
    min_distance: Annotated[int, typer.Option(min=0)] = 0,
    wandb_project: Annotated[
        str | None, typer.Option(help="Upload full sequence records to this project.")
    ] = None,
    wandb_entity: str | None = None,
):
    """Sample edited enhancers by default, or explicitly generate de novo cores."""
    import logging
    import math

    from regulonado.design.diffusion import condition_vector, load_checkpoint, sample
    from regulonado.design.generators import diverse_sequences
    from regulonado.design.provenance import ProposalJournal, checkpoint_identity

    try:
        if mode not in {"edit", "de_novo"}:
            raise ValueError("mode must be edit or de_novo")
        if mode == "edit" and (not math.isfinite(mask_fraction) or not 0 < mask_fraction < 1):
            raise ValueError("Edit mask_fraction must be finite and strictly between zero and one")
        profile = json.loads(activities)
        if not isinstance(profile, dict):
            raise ValueError("activities must be a JSON object")
        template = None
        if template_fasta:
            from regulonado.design.provenance import read_template

            template = read_template(template_fasta)
            if length is not None and length != len(template):
                raise ValueError("Explicit length conflicts with the template FASTA")
            length = len(template)
        if mode == "edit" and template is None:
            raise ValueError(
                "Edit mode requires --template-fasta; use --mode de_novo for a new core"
            )
        if mode == "de_novo" and template is not None:
            raise ValueError("--template-fasta is only used in edit mode")
        if length is None:
            raise ValueError("De novo mode requires --length")
        model, metadata = load_checkpoint(checkpoint, device)
        values, observed = condition_vector(metadata, profile)
        journal = ProposalJournal(
            out / "provenance",
            parent=template,
            metadata=dict(
                checkpoint=str(checkpoint.resolve()),
                checkpoint_id=checkpoint_identity(checkpoint),
                conditions=profile,
                requested_proposals=count,
                steps=steps,
                guidance=guidance,
                random_seed=seed,
            ),
        )
        if template:
            journal.add(
                dict(
                    sequence=template,
                    mode=mode,
                    masked_positions=[],
                    requested_mask_fraction=0.0,
                    actual_mask_fraction=0.0,
                ),
                baseline=True,
            )
        proposals = sample(
            model,
            values,
            observed,
            length=length,
            count=count,
            steps=steps,
            guidance=guidance,
            seed=seed,
            mode=mode,
            template=template,
            mask_fraction=mask_fraction,
            return_records=True,
        )
        ids = [journal.add(proposal) for proposal in proposals]
        sequences = [proposal["sequence"] for proposal in proposals]
        selected = diverse_sequences(sequences, count=count, min_distance=min_distance)
    except (ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    first_id = {}
    for event_id, sequence in zip(ids, sequences):
        first_id.setdefault(sequence, event_id)
    selected_ids = {first_id[sequence] for sequence in selected}
    for event_id in journal.records:
        journal.update(event_id, selected=event_id in selected_ids, oracle_scored=False)
    (out / "sequences.fa").write_text(
        "".join(
            f">{first_id[sequence]} "
            f"sequence_id={journal.records[first_id[sequence]]['sequence_id']}\n{sequence}\n"
            for sequence in selected
        )
    )
    with (out / "sequences.tsv").open("w") as handle:
        handle.write("proposal_id\tsequence_id\tsequence\tgc_fraction\n")
        for sequence in selected:
            event_id = first_id[sequence]
            handle.write(
                f"{event_id}\t{journal.records[event_id]['sequence_id']}\t{sequence}\t"
                f"{(sequence.count('G') + sequence.count('C')) / len(sequence):.6f}\n"
            )
    info = dict(
        checkpoint=str(checkpoint.resolve()),
        activities=profile,
        length=length,
        requested=count,
        retained=len(selected),
        steps=steps,
        guidance=guidance,
        seed=seed,
        mode=mode,
        mask_fraction=mask_fraction if mode == "edit" else 1.0,
        min_distance=min_distance,
        activity_columns=metadata["config"]["activity_columns"],
        provenance_dir=str(journal.root),
    )
    (out / "run.json").write_text(json.dumps(info, indent=2))
    journal._json(
        "selection.json",
        dict(
            selected_proposal_ids=[first_id[s] for s in selected],
            oracle_scored=False,
            selection_method="unique/min_distance",
        ),
    )
    (journal.root / "selection.fa").write_text((out / "sequences.fa").read_text())
    run = None
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project, entity=wandb_entity, job_type="diffusion-sample", config=info
            )
            (journal.root / "wandb.json").write_text(json.dumps({"url": run.url, "id": run.id}))
            journal.upload(wandb, run, final=True)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "W&B upload failed; local records retained: %s", exc
            )
            journal._append("upload_errors.jsonl", {"error": str(exc)})
        finally:
            if run is not None:
                try:
                    run.finish()
                except Exception as exc:
                    journal._append("upload_errors.jsonl", {"error": str(exc)})
    typer.echo(
        f"Generated {len(proposals)} proposals; retained {len(selected)} unique sequences in {out}"
    )


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
