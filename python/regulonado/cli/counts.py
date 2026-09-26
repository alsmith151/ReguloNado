"""``regulonado counts``: BAM event counting into a ``RegionCountData`` dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer

counts_app = typer.Typer(
    no_args_is_help=True,
    help="Count BAM events over regions and gather them into a RegionCountData run.",
)


def _read_regions(path: Path) -> pl.DataFrame:
    """Read a region set.

    Parquet keeps every column (``split``, provenance); BED is chrom/start/end only.
    """
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    from regulonado.genomics import read_intervals

    frame = read_intervals(path)
    return pl.from_pandas(frame[["chrom", "start", "end"]])


def _assign_splits(
    regions: pl.DataFrame, val_chroms: list[str] | None, test_chroms: list[str] | None
) -> pl.DataFrame:
    """Add a ``split`` column from ``--val-chroms``/``--test-chroms``, unless one already exists."""
    if "split" in regions.columns:
        return regions
    val = list(val_chroms or [])
    test = list(test_chroms or [])
    return regions.with_columns(
        pl.when(pl.col("chrom").is_in(test))
        .then(pl.lit("test"))
        .when(pl.col("chrom").is_in(val))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


@counts_app.command("bam")
def bam(
    regions: Annotated[
        Path,
        typer.Argument(help="Region set: BED or parquet with chrom/start/end, optionally split"),
    ],
    tracks: Annotated[
        Path, typer.Option("--tracks", help="tracks.parquet from 'regulonado tracks assemble'")
    ],
    anchor_regions: Annotated[
        Path, typer.Option("--anchor-regions", help="High-anchor BED/parquet")
    ],
    background_regions: Annotated[
        Path, typer.Option("--background-regions", help="Background BED/parquet")
    ],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o", help="Per-track parquet cache directory")
    ],
    track: Annotated[
        Optional[str],
        typer.Option("--track", help="Count only this track (for per-track cluster array jobs)"),
    ] = None,
    chrom_sizes: Annotated[
        Optional[Path],
        typer.Option("--chrom-sizes", help="chrom.sizes/.fai for the region bounds check"),
    ] = None,
    target_width: Annotated[
        int,
        typer.Option("--target-width", help="Width of the counted window, centred on each region"),
    ] = 1000,
    threads: Annotated[
        int, typer.Option("--threads", help="htslib decompression threads per BAM")
    ] = 1,
    max_workers: Annotated[
        int, typer.Option("--workers", "-w", help="Parallel BAMs when --track is not given")
    ] = 1,
) -> None:
    """Count BAM events over ``regions`` for every track (or one, with ``--track``).

    Resumable: each track's result is cached at ``<out-dir>/<track_name>.parquet``
    and skipped on a re-run. Run ``regulonado counts gather`` afterwards to
    assemble the per-track cache into one ``RegionCountData`` run.
    """
    from regulonado.counts.bam import BamRegionCounter, read_tracks_table
    from regulonado.genomics import read_chrom_sizes
    from regulonado.normalization import read_regions

    region_frame = _read_regions(regions)
    tracks_frame = read_tracks_table(tracks)
    if track is not None:
        tracks_frame = tracks_frame.filter(pl.col("track_name") == track)
        if tracks_frame.height == 0:
            typer.echo(f"No track named {track!r} in {tracks}", err=True)
            raise typer.Exit(1)

    sizes = read_chrom_sizes(chrom_sizes) if chrom_sizes is not None else None
    counter = BamRegionCounter(
        region_frame,
        target_width=target_width,
        chrom_sizes=sizes,
        anchor_windows=read_regions(anchor_regions),
        background_windows=read_regions(background_regions),
        threads=threads,
    )
    typer.echo(
        f"Regions     : {counter.n_regions} "
        f"(dropped {counter.n_dropped_out_of_bounds} out-of-bounds, "
        f"{counter.n_dropped_duplicates} duplicate)"
    )
    typer.echo(f"Tracks      : {tracks_frame.height}")
    typer.echo(f"Out dir     : {out_dir}")

    counter.count_tracks(tracks_frame, max_workers=max_workers, cache_dir=out_dir)
    typer.echo("Done.")


@counts_app.command("gather")
def gather(
    regions: Annotated[
        Path, typer.Argument(help="Same region set passed to 'counts bam'")
    ],
    tracks: Annotated[
        Path, typer.Option("--tracks", help="Same tracks.parquet passed to 'counts bam'")
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Per-track parquet cache written by 'counts bam'"),
    ],
    dataset_dir: Annotated[
        Path, typer.Option("--dataset-dir", help="Output RegionCountData run directory")
    ],
    chrom_sizes: Annotated[
        Optional[Path],
        typer.Option("--chrom-sizes", help="chrom.sizes/.fai for the region bounds check"),
    ] = None,
    target_width: Annotated[
        int,
        typer.Option("--target-width", help="Width of the counted window, centred on each region"),
    ] = 1000,
    val_chroms: Annotated[
        Optional[list[str]],
        typer.Option("--val-chroms", help="Chromosomes assigned split=val with no 'split' column"),
    ] = None,
    test_chroms: Annotated[
        Optional[list[str]],
        typer.Option("--test-chroms", help="Chromosomes assigned split=test with no 'split' col"),
    ] = None,
) -> None:
    """Gather ``counts bam``'s per-track cache into one ``RegionCountData`` run.

    Writes to ``--dataset-dir``.
    """
    from regulonado.counts.bam import BamRegionCounter, read_tracks_table
    from regulonado.counts.dataset import RegionCountData
    from regulonado.genomics import read_chrom_sizes

    region_frame = _assign_splits(_read_regions(regions), val_chroms, test_chroms)
    tracks_frame = read_tracks_table(tracks)

    sizes = read_chrom_sizes(chrom_sizes) if chrom_sizes is not None else None
    counter = BamRegionCounter(region_frame, target_width=target_width, chrom_sizes=sizes)

    counts_matrix, track_names, results = counter.gather(out_dir, tracks_frame)
    stats = counter.anchor_stats(results)
    tracks_meta = tracks_frame.join(stats, on="track_name", how="left", maintain_order="left")

    data = RegionCountData.from_arrays(counter.regions, counts_matrix, track_names, tracks_meta)
    written = data.write(dataset_dir)
    typer.echo(f"Wrote {data.n_regions} regions x {data.n_tracks} tracks to {dataset_dir}")
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")
