from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)


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


def build(
    bed_file: Annotated[Path, typer.Argument(help="BED file; column 4 used as fold label")],
    fasta_file: Annotated[
        Path, typer.Argument(help="Reference genome FASTA (.fai index required)")
    ],
    output_dir: Annotated[Path, typer.Argument(help="Output directory for the Arrow DatasetDict")],
    track_table: Annotated[
        Path,
        typer.Option(
            "--track-table",
            help=(
                "tracks.parquet from 'regulonado tracks assemble' — the sole source of "
                "track identity, order and per-track annotation. Discovery, dedupe and "
                "QC all happen upstream of this command; see 'regulonado tracks'."
            ),
        ),
    ],
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
    stage: Annotated[
        bool,
        typer.Option(
            "--stage/--no-stage", help="Copy FASTA and BigWigs to scratch before building"
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Regenerate splits that already exist")
    ] = False,
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
                "Target on-disk size per Arrow shard file (in_memory). Shard "
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
                "Explicit samples per Arrow shard file (in_memory); overrides "
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
                "Concurrent Arrow shard writers for in_memory. Defaults to "
                "min(8, --n-extract-threads); lower this if memory is tight."
            ),
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=(
                "Build strategy: 'in_memory' (default) decodes each "
                "chromosome's binned signal for all tracks into RAM once and "
                "slices every window from it — ~10x fewer BigWig seeks; "
                "'streaming' reads each window's interval from every BigWig "
                "per batch, split by split — bounded memory, kept mainly as "
                "a parity reference."
            ),
        ),
    ] = "in_memory",
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
    """Build an Arrow DatasetDict from BED / FASTA / a pre-assembled track table.

    Track discovery, dedupe and QC happen before this command, via
    'regulonado tracks discover/qc/assemble' — this is intentionally the only
    place a track list can be named, so there is one way to name tracks, not
    two.

    \b
    Examples
    --------
    regulonado build intervals.bed genome.fa out/ \\
        --track-table results/tracks/tracks.parquet \\
        --split train:train --split validation:valid \\
        --shift-max-bp 128 --n-extract-threads 16 --stage
    """
    from regulonado.dataset import DEFAULT_SPLITS, build_dataset

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

    typer.echo(f"Track table : {track_table}")
    typer.echo(f"Splits      : {list(splits)}")
    typer.echo(f"Output      : {output_dir}")

    build_dataset(
        bed_file=bed_file,
        fasta_file=fasta_file,
        track_table=track_table,
        output_dir=output_dir,
        splits=splits,
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        n_extract_threads=n_extract_threads,
        arrow_batch_size=arrow_batch_size,
        shard_size=shard_size,
        shard_target_mb=shard_target_mb,
        arrow_compression=arrow_compression,
        arrow_write_threads=arrow_write_threads,
        stage_to_scratch=stage,
        overwrite=overwrite,
        profile=profile,
        strategy=strategy,
        chrom_filter=list(chrom) if chrom else None,
        return_dataset=False,
    )
