from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)


def dataset(
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
    zstd_level: Annotated[
        int,
        typer.Option(
            "--zstd-level",
            help="ZSTD compression level (default 3)",
        ),
    ] = 3,
    write_threads: Annotated[
        Optional[int],
        typer.Option(
            "--write-threads",
            help=(
                "Concurrent Parquet shard writers. Defaults to "
                "min(8, --n-extract-threads); lower this if memory is tight."
            ),
        ),
    ] = None,
    rows_per_row_group: Annotated[
        int,
        typer.Option(
            "--rows-per-row-group",
            help="Rows per Parquet row group (default 1)",
        ),
    ] = 1,
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
    bin_denominator: Annotated[
        str,
        typer.Option(
            "--bin-denominator",
            help=(
                "How a bin's mean is formed: 'bin_width' (default) divides summed "
                "signal by the bin's in-contig width, so bases without a BigWig "
                "record count as zero; 'covered_bases' divides by recorded bases "
                "only, which inflates sparse bins in BigWigs that omit zeros."
            ),
        ),
    ] = "bin_width",
    missing_bins: Annotated[
        str,
        typer.Option(
            "--missing-bins",
            help=(
                "Value stored for bins with no data (past the contig end, or "
                "wholly NaN in the BigWig): 'nan' (default, maskable in "
                "training) or 'zero'."
            ),
        ),
    ] = "nan",
) -> None:
    """Build an Arrow DatasetDict from BED / FASTA / a pre-assembled track table.

    Track discovery, dedupe and QC happen before this command, via
    'regulonado tracks discover/qc/assemble' — this is intentionally the only
    place a track list can be named, so there is one way to name tracks, not
    two.

    \b
    Examples
    --------
    regulonado dataset intervals.bed genome.fa out/ \\
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
        shard_size=shard_size,
        shard_target_mb=shard_target_mb,
        zstd_level=zstd_level,
        write_threads=write_threads,
        rows_per_row_group=rows_per_row_group,
        stage_to_scratch=stage,
        overwrite=overwrite,
        profile=profile,
        strategy=strategy,
        chrom_filter=list(chrom) if chrom else None,
        bin_denominator=bin_denominator,
        missing_bins=missing_bins,
    )
