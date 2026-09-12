from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer


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
            help="Dataset dir with tracks.parquet; required only for legacy checkpoints",
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
