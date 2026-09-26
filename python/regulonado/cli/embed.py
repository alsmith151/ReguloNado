"""``regulonado embed``: cache frozen-backbone embeddings for region-count training."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

embed_app = typer.Typer(
    no_args_is_help=True,
    help="Cache frozen-backbone embeddings over region targets, for training a count head.",
)


def _resolve_device(name: str) -> str:
    """``"auto"`` -> cuda > mps > cpu; anything else is returned unchanged."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@embed_app.command("regions")
def regions(
    region_dataset: Annotated[
        Path,
        typer.Argument(
            help="Region table: 'counts regions' output, or a RegionCountData run directory"
        ),
    ],
    fasta: Annotated[Path, typer.Argument(help="Genome FASTA (a .fai index is built if missing)")],
    backbone: Annotated[
        str, typer.Option("--backbone", help="Backbone type: borzoi, enformer or alphagenome")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output embeddings directory")],
    pretrained: Annotated[
        Optional[str],
        typer.Option(
            "--pretrained",
            help="Pretrained checkpoint name/path (see python/configs/backbone/*.yaml)",
        ),
    ] = None,
    chroms: Annotated[
        Optional[list[str]],
        typer.Option("--chroms", help="Only cache these chromosomes (default: every chromosome)"),
    ] = None,
    rc: Annotated[
        bool, typer.Option("--rc", help="Also cache a reverse-complement pass (features_rc)")
    ] = False,
    pool_to: Annotated[
        Optional[int],
        typer.Option("--pool-to", help="Average adjacent bins up to this bp width"),
    ] = None,
    context: Annotated[
        int,
        typer.Option("--context", help="Input length (bp) for flexible backbones (AlphaGenome)"),
    ] = 1_048_576,
    stride: Annotated[
        int,
        typer.Option("--stride", help="Central bp kept per window for flexible backbones"),
    ] = 524_288,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Windows per forward pass")
    ] = 1,
    device: Annotated[
        str, typer.Option("--device", help="auto (cuda > mps > cpu), cpu, cuda or mps")
    ] = "auto",
    allow_random_init: Annotated[
        bool,
        typer.Option(
            "--allow-random-init",
            help="Allow a randomly initialised backbone when --pretrained is not given",
        ),
    ] = False,
    row_group_size: Annotated[
        int, typer.Option("--row-group-size", help="Parquet row group size per chromosome file")
    ] = 256,
) -> None:
    """Cache one backbone's frozen embeddings over every region's scored target.

    Writes ``<out>/<chrom>.parquet`` per chromosome (skipped if it already exists, so
    this is safe to rerun -- one job per chromosome via ``--chroms``) and a shared
    ``<out>/manifest.parquet``, checked for consistency on every rerun.
    """
    import polars as pl
    from regulonado.embeddings.cache import embed_regions as _embed_regions
    from regulonado.model.adapters import BackboneSpec, build_backbone_adapter
    from regulonado.sequence import open_genome

    regions_path = (
        region_dataset / "regions.parquet" if region_dataset.is_dir() else region_dataset
    )
    region_table = pl.read_parquet(regions_path)
    spec = BackboneSpec(
        backbone_type=backbone,  # type: ignore[arg-type]
        pretrained_name=pretrained,
        allow_random_init=allow_random_init,
    )
    adapter = build_backbone_adapter(spec)
    genome = open_genome(fasta)
    resolved_device = _resolve_device(device)

    _embed_regions(
        region_table,
        genome,
        adapter,
        out,
        backbone=backbone,
        checkpoint=pretrained or "",
        chroms=chroms,
        rc=rc,
        pool_to=pool_to,
        context=context,
        stride=stride,
        batch_size=batch_size,
        device=resolved_device,
        row_group_size=row_group_size,
    )
    typer.echo(f"Wrote embeddings to {out}")
