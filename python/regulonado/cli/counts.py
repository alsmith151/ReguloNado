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


@counts_app.command("regions")
def region_set(
    regions: Annotated[
        Path,
        typer.Argument(help="Region set: BED or parquet with chrom/start/end, optionally split"),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Canonical regions.parquet")],
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
    """Write the canonical region table ``counts bam``/``gather`` and ``embed regions`` share.

    Applies the same geometry, bounds check, de-duplication and split assignment as
    ``counts gather``, without reading any BAM -- so the embedding cache can be built
    from it while counting is still running, and its rows match the gathered dataset's.
    """
    from regulonado.counts.bam import BamRegionCounter
    from regulonado.genomics import read_chrom_sizes

    region_frame = _assign_splits(_read_regions(regions), val_chroms, test_chroms)
    sizes = read_chrom_sizes(chrom_sizes) if chrom_sizes is not None else None
    counter = BamRegionCounter(region_frame, target_width=target_width, chrom_sizes=sizes)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    counter.regions.write_parquet(tmp)
    tmp.replace(output)
    typer.echo(
        f"Wrote {counter.n_regions} regions to {output} "
        f"(dropped {counter.n_dropped_out_of_bounds} out-of-bounds, "
        f"{counter.n_dropped_duplicates} duplicate)"
    )


@counts_app.command("bam")
def bam(
    regions: Annotated[
        Path,
        typer.Argument(help="Region set: BED or parquet with chrom/start/end, optionally split"),
    ],
    tracks: Annotated[
        Path,
        typer.Option(
            "--tracks",
            help="tracks.parquet from 'regulonado tracks assemble' (discovered with "
            "'--format bam', or bigWig tracks that carry a 'bam')",
        ),
    ],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o", help="Per-track parquet cache directory")
    ],
    anchor_regions: Annotated[
        Optional[Path], typer.Option("--anchor-regions", help="High-anchor BED/parquet")
    ] = None,
    background_regions: Annotated[
        Optional[Path], typer.Option("--background-regions", help="Background BED/parquet")
    ] = None,
    source: Annotated[str, typer.Option(help="Region target source: bam or bigwig")] = "bam",
    aggregation: Annotated[str, typer.Option(help="BigWig mean, max, count, or logcount")] = "mean",
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
        int, typer.Option("--threads", help="BGZF decompression threads per BAM")
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
    from regulonado.counts.bam import BamRegionCounter, read_count_tracks
    from regulonado.genomics import read_chrom_sizes
    from regulonado.normalization import read_regions

    if source == "bigwig":
        from regulonado.counts.signal import count_signal_regions, signal_tracks

        counter = BamRegionCounter(
            _read_regions(regions),
            target_width=target_width,
            chrom_sizes=read_chrom_sizes(chrom_sizes) if chrom_sizes else None,
        )
        count_signal_regions(counter, signal_tracks(tracks), out_dir, aggregation, track)
        typer.echo("Done.")
        return
    if source != "bam" or anchor_regions is None or background_regions is None:
        raise typer.BadParameter("BAM source requires anchor-regions and background-regions")
    region_frame = _read_regions(regions)
    # Validate BAMs only for the track(s) this call counts, so one per-track job does
    # not fail on another track's missing file.
    tracks_frame = read_count_tracks(tracks, require_bam=False)
    if track is not None:
        tracks_frame = tracks_frame.filter(pl.col("track_name") == track)
        if tracks_frame.height == 0:
            typer.echo(f"No track named {track!r} in {tracks}", err=True)
            raise typer.Exit(1)
    no_bam = tracks_frame.filter(pl.col("bam").is_null())["track_name"].to_list()
    if no_bam:
        typer.echo(f"No 'bam' path for track(s): {no_bam}", err=True)
        raise typer.Exit(1)
    absent = [bam for bam in tracks_frame["bam"].to_list() if not Path(bam).exists()]
    if absent:
        typer.echo(f"BAM(s) not found: {absent}", err=True)
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
    regions: Annotated[Path, typer.Argument(help="Same region set passed to 'counts bam'")],
    tracks: Annotated[
        Path,
        typer.Option("--tracks", help="Same tracks.parquet passed to 'counts bam'"),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Per-track parquet cache written by 'counts bam'"),
    ],
    dataset_dir: Annotated[
        Path, typer.Option("--dataset-dir", help="Output RegionCountData run directory")
    ],
    source: Annotated[str, typer.Option(help="Region target source: bam or bigwig")] = "bam",
    aggregation: Annotated[str, typer.Option(help="BigWig mean, max, count, or logcount")] = "mean",
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

    Writes to ``--dataset-dir``. Tracks with no finite ``log_size_factor`` (anchor not
    clearing background) are dropped with a warning.
    """
    import numpy as np
    from regulonado.counts.bam import BamRegionCounter, read_count_tracks
    from regulonado.counts.dataset import RegionCountData
    from regulonado.genomics import read_chrom_sizes

    region_frame = _assign_splits(_read_regions(regions), val_chroms, test_chroms)
    if source == "bigwig":
        from regulonado.counts.signal import gather_signal_regions, signal_tracks

        counter = BamRegionCounter(
            region_frame,
            target_width=target_width,
            chrom_sizes=read_chrom_sizes(chrom_sizes) if chrom_sizes else None,
        )
        data = gather_signal_regions(
            counter, signal_tracks(tracks), out_dir, dataset_dir, aggregation
        )
        typer.echo(f"Wrote {data.n_regions} regions x {data.n_tracks} tracks to {dataset_dir}")
        return
    if source != "bam":
        raise typer.BadParameter("source must be bam or bigwig")
    tracks_frame = read_count_tracks(tracks, require_bam=False)

    sizes = read_chrom_sizes(chrom_sizes) if chrom_sizes is not None else None
    counter = BamRegionCounter(region_frame, target_width=target_width, chrom_sizes=sizes)

    counts_matrix, track_names, results = counter.gather(out_dir, tracks_frame)
    stats = counter.anchor_stats(results)
    tracks_meta = tracks_frame.join(stats, on="track_name", how="left", maintain_order="left")

    # A track whose anchor windows don't clear background has no size factor, so the count
    # model can't place it on the common scale: leave it out of the dataset, loudly.
    scalable = np.isfinite(stats["log_size_factor"].to_numpy())
    if not scalable.any():
        typer.echo("No track has a finite log_size_factor; nothing to train on.", err=True)
        raise typer.Exit(1)
    if not scalable.all():
        dropped = stats.filter(~pl.Series(scalable))
        typer.echo(
            f"WARNING: dropping {dropped.height} track(s) whose anchor does not clear "
            "background (no finite log_size_factor):",
            err=True,
        )
        for row in dropped.iter_rows(named=True):
            typer.echo(
                f"  {row['track_name']}: count_scale_low={row['count_scale_low']:.4g} "
                f"count_scale_high={row['count_scale_high']:.4g}",
                err=True,
            )
        counts_matrix = counts_matrix[:, scalable]
        track_names = [name for name, keep in zip(track_names, scalable, strict=True) if keep]
        tracks_meta = tracks_meta.filter(pl.Series(scalable))

    data = RegionCountData.from_arrays(counter.regions, counts_matrix, track_names, tracks_meta)
    written = data.write(dataset_dir)
    typer.echo(f"Wrote {data.n_regions} regions x {data.n_tracks} tracks to {dataset_dir}")
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")


@counts_app.command("bigwig")
def bigwig(
    regions: Annotated[Path, typer.Argument(help="BED/parquet regions")],
    tracks: Annotated[Path, typer.Option("--tracks", help="Assembled BigWig tracks.parquet")],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Per-track summary cache")],
    aggregation: Annotated[str, typer.Option(help="mean, max, count, or logcount")] = "mean",
    track: Annotated[Optional[str], typer.Option(help="Only this track")] = None,
    target_width: Annotated[int, typer.Option(help="Scored region width")] = 1000,
    chrom_sizes: Annotated[
        Optional[Path], typer.Option(help="Contig sizes for bounds checks")
    ] = None,
) -> None:
    """Summarize BigWig signal over regions; assemble with counts gather --source bigwig."""
    bam(
        regions=regions,
        tracks=tracks,
        out_dir=out_dir,
        source="bigwig",
        aggregation=aggregation,
        track=track,
        target_width=target_width,
        chrom_sizes=chrom_sizes,
    )
