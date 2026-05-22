from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def scale(
    bigwig_dir: Annotated[Path, typer.Argument(help="Directory containing .bw / .bigwig files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: csv or parquet")] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
    glob: Annotated[str, typer.Option("--glob", help="Glob pattern for bigwig files")] = "*.bw",
) -> None:
    """Infer scale factors for all bigwig files in a directory."""
    from regulonado.scaling import infer_scale_factors, save_scale_factors

    bw_files = sorted(bigwig_dir.glob(glob))
    if not bw_files:
        typer.echo(f"No files matching '{glob}' found in {bigwig_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(bw_files)} bigwig files")
    df = infer_scale_factors(bw_files, max_workers=max_workers)
    save_scale_factors(df, output, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {output}")


@app.command()
def build(
    bed_file: Annotated[
        Path, typer.Argument(help="BED file; column 4 used as fold label")
    ],
    fasta_file: Annotated[
        Path, typer.Argument(help="Reference genome FASTA (.fai index required)")
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for the Arrow DatasetDict")
    ],
    bigwig: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--bigwig", "-b",
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
        str, typer.Option("--bigwig-glob", help="Glob when using --bigwig-dir")
    ] = "*.bw",
    split: Annotated[
        Optional[list[str]],
        typer.Option(
            "--split", "-s",
            help=(
                "Split definition NAME:FOLD1,FOLD2 (repeat for each split). "
                "Default: train:fold0-2,fold5-7, validation:fold4, test:fold3."
            ),
        ),
    ] = None,
    context_length: Annotated[
        int, typer.Option("--context-length", help="Input sequence length in bp")
    ] = 524_288,
    bin_size: Annotated[
        int, typer.Option("--bin-size", help="Signal bin size in bp")
    ] = 32,
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
        bool, typer.Option("--drop-missing", help="Drop missing BigWig paths instead of raising an error")
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile/--no-profile",
            help="Emit per-phase timing summary to stderr after each split (fast path only)",
        ),
    ] = False,
    no_fast_path: Annotated[
        bool,
        typer.Option(
            "--no-fast-path",
            help="Use the legacy pybigtools path instead of the two-phase Rust extraction",
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
            help="Samples per Rust-written Arrow record batch",
        ),
    ] = 8,
    arrow_compression: Annotated[
        str,
        typer.Option(
            "--arrow-compression",
            help="Arrow IPC compression: zstd, lz4, or none",
        ),
    ] = "zstd",
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
    from regulonado.dataset import DEFAULT_SPLITS, build_dataset, build_dataset_fast

    # --- resolve BigWig paths ------------------------------------------------
    if bigwig_dir is not None:
        bw_paths: list[str] = [str(p) for p in sorted(bigwig_dir.glob(bigwig_glob))]
        if not bw_paths:
            typer.echo(f"No files matching '{bigwig_glob}' in {bigwig_dir}", err=True)
            raise typer.Exit(1)
    elif bigwig:
        bw_paths = [str(p) for p in bigwig]
    else:
        typer.echo("Provide --bigwig files or --bigwig-dir.", err=True)
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

    if no_fast_path:
        build_dataset(
            bed_file=bed_file,
            fasta_file=fasta_file,
            bigwig_paths=bw_paths,
            output_dir=output_dir,
            splits=splits,
            context_length=context_length,
            bin_size=bin_size,
            n_pred_bins=n_pred_bins,
            shift_max_bp=shift_max_bp,
            n_io_threads=io_threads,
            num_proc=num_proc,
            cache_dir=cache_dir,
            writer_batch_size=writer_batch_size,
            stage_to_scratch=stage,
            overwrite=overwrite,
            drop_missing=drop_missing,
        )
    else:
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
            arrow_compression=arrow_compression,
            num_proc=num_proc,
            cache_dir=cache_dir,
            writer_batch_size=writer_batch_size,
            stage_to_scratch=stage,
            overwrite=overwrite,
            drop_missing=drop_missing,
            profile=profile,
            strategy=strategy,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
