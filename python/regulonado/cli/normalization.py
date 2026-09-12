from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

normalization_app = typer.Typer(
    no_args_is_help=True,
    help="Infer, correct, and apply per-track signal normalisation factors.",
)


@normalization_app.command("infer")
def scale(
    bigwig_dir: Annotated[Path, typer.Argument(help="Directory containing .bw / .bigwig files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
    glob: Annotated[str, typer.Option("--glob", help="Glob pattern for bigwig files")] = "*.bw",
) -> None:
    """Infer scale factors for all bigwig files in a directory."""
    from regulonado.normalization import infer_scale_factors, save_scale_factors

    bw_files = sorted(bigwig_dir.glob(glob))
    if not bw_files:
        typer.echo(f"No files matching '{glob}' found in {bigwig_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(bw_files)} bigwig files")
    df = infer_scale_factors(bw_files, max_workers=max_workers)
    save_scale_factors(df, output, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {output}")


def _included_tracks(track_table: Path) -> pd.DataFrame:
    """Included rows of a ``tracks.parquet``-shaped table, ordered by ``track_index``."""
    from regulonado.tracks_table import read_track_table

    if not track_table.exists():
        typer.echo(f"Track table not found: {track_table}", err=True)
        raise typer.Exit(1)
    table = read_track_table(track_table)
    included = table[table["status"] == "included"].sort_values("track_index").reset_index(
        drop=True
    )
    if included.empty:
        typer.echo(f"No included tracks in {track_table}.", err=True)
        raise typer.Exit(1)
    return included


@normalization_app.command("original")
def calculate_original_scaling(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet (or any tracks.parquet)")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: <track_table_dir>/scale_factors.parquet)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    bin_size: Annotated[
        int, typer.Option("--bin-size", help="build.bin_size; scales RPKM to raw counts")
    ] = 32,
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
) -> None:
    """Infer original scale factors for every included track in a track table.

    Output rows are sorted by track_index so they can be applied directly by position.
    """
    from regulonado.normalization import (
        compute_clip_thresholds,
        infer_scale_factors,
        save_scale_factors,
    )

    included = _included_tracks(track_table)
    bw_paths = [Path(p) for p in included["resolved_path"]]

    ext = "parquet" if fmt == "parquet" else "csv"
    out_path = output if output is not None else track_table.parent / f"scale_factors.{ext}"

    typer.echo(f"Track table : {track_table}")
    typer.echo(f"Tracks      : {len(bw_paths)}")
    typer.echo(f"Bin size    : {bin_size} bp")
    typer.echo(f"Output      : {out_path}")

    df = infer_scale_factors(bw_paths, max_workers=max_workers)

    # bamnado returns scale_factor = library_size / 1e9, which is the RPKM→raw-counts
    # factor without the bin_size term.  RPKM = reads / (lib/1e6) / (bin_size/1e3),
    # so raw_count = RPKM × (lib/1e6) × (bin_size/1e3) = RPKM × sf_bamnado × bin_size.
    df["scale_factor"] = df["scale_factor"] * bin_size

    # Join track_index/track_name from the table, then sort so row i corresponds
    # to track i — enabling direct positional application.
    join_cols = included[["track_index", "track_name", "resolved_path"]]
    df = df.merge(
        join_cols, left_on="path", right_on="resolved_path", how="left", validate="one_to_one"
    )
    df = df.drop(columns=["resolved_path"]).sort_values("track_index").reset_index(drop=True)

    df = compute_clip_thresholds(df)

    # Put the fields consumed during training first.
    priority = ["track_index", "track_name", "scale_factor", "clip_soft", "clip_hard"]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {out_path}")


@normalization_app.command("anchor")
def calculate_anchor_scaling(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet (or any tracks.parquet)")
    ],
    anchor_regions: Annotated[
        Path, typer.Option("--anchor-regions", help="High-anchor BED/parquet")
    ],
    background_regions: Annotated[
        Path, typer.Option("--background-regions", help="Background BED/parquet")
    ],
    heldout_regions: Annotated[Optional[Path], typer.Option("--heldout-regions")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f")] = "parquet",
    bin_size: Annotated[int, typer.Option("--bin-size", help="build.bin_size")] = 32,
    window_stat_bp: Annotated[int, typer.Option("--window-stat-bp")] = 1000,
    window_stat_bp_by_assay: Annotated[
        Optional[str],
        typer.Option(
            "--window-stat-bp-by-assay",
            help='JSON {"assay": bp}; overrides --window-stat-bp for named assays',
        ),
    ] = None,
    background_sample: Annotated[Optional[int], typer.Option("--background-sample")] = 5000,
    max_workers: Annotated[int, typer.Option("--workers", "-w")] = 16,
) -> None:
    """Scale tracks to housekeeping-promoter anchor units.

    Degenerate tracks (``quality == "failed"``) are written as-is, not dropped
    here — ``regulonado tracks qc --check anchor`` (or ``qc.rules``) decides
    whether to drop them, since that decision belongs to the opt-in QC gate,
    not to scaling.
    """
    import json

    from regulonado.normalization import anchor_scale_factors, save_scale_factors

    included = _included_tracks(track_table)
    out_path = output or track_table.parent / f"scale_factors.{fmt}"
    by_assay = json.loads(window_stat_bp_by_assay) if window_stat_bp_by_assay else None
    df = anchor_scale_factors(
        included["resolved_path"].tolist(),
        anchor_regions,
        background_regions,
        heldout_regions=heldout_regions,
        bin_size=bin_size,
        window_stat_bp=window_stat_bp,
        window_stat_bp_by_assay=by_assay,
        assays=included.get("assay", pd.Series([None] * len(included))).tolist(),
        background_sample=background_sample,
        max_workers=max_workers,
    )
    df["track_name"] = included["track_name"].to_numpy()
    degenerate = int((df["quality"] == "failed").sum())
    if degenerate:
        typer.echo(f"{degenerate} degenerate anchor track(s) (anchor <= background).", err=True)
    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved anchor scale factors to {out_path}")
    typer.echo("Training requirement: data.apply_squash=false")


@normalization_app.command("tmm")
def calculate_tmm_scaling(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet (or any tracks.parquet)")
    ],
    intervals: Annotated[
        Optional[Path],
        typer.Option("--intervals", help="BED of sampling regions (inputs.intervals)"),
    ] = None,
    interval_means: Annotated[
        Optional[Path],
        typer.Option(
            "--interval-means",
            help="Precomputed 'tracks interval-means' output; skips a fresh scan when given",
        ),
    ] = None,
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help=(
                "Scale-factors parquet from normalization original "
                "(default: <track_table_dir>/scale_factors.parquet)"
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: overwrites --scale-factors input)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    bin_size: Annotated[int, typer.Option("--bin-size", help="build.bin_size")] = 32,
    n_pred_bins: Annotated[int, typer.Option("--n-pred-bins", help="build.n_pred_bins")] = 6_144,
    shift_max_bp: Annotated[int, typer.Option("--shift-max-bp", help="build.shift_max_bp")] = 0,
    sample_n: Annotated[
        Optional[int],
        typer.Option("--sample-n", help="Sample this many BED rows instead of scanning all"),
    ] = None,
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
    trim_m: Annotated[
        float,
        typer.Option(
            "--trim-m",
            help="Fraction to trim from each M-value tail (edgeR default 0.3)",
        ),
    ] = 0.3,
    trim_a: Annotated[
        float,
        typer.Option(
            "--trim-a",
            help="Fraction to trim from each A-value tail (edgeR default 0.05)",
        ),
    ] = 0.05,
    min_count: Annotated[
        float,
        typer.Option(
            "--min-count",
            help="Minimum pseudo-count for a region to be included",
        ),
    ] = 1.0,
) -> None:
    """Compute edgeR-style TMM normalisation factors directly from the BigWigs.

    Scans ``--intervals`` once per track (:func:`regulonado.qc.track_interval_means`,
    the same scan the ``interval_signal``/``replicate_concordance`` QC checks
    share), converts to pseudo-counts using library sizes from the scale-factors
    parquet, and runs TMM estimation over the sampled regions.

    The output parquet gains a ``tmm_factor`` column and the ``scale_factor``
    column is updated to ``old_scale_factor / tmm_factor`` so that multiplying
    any raw RPKM BigWig value by the new scale_factor yields TMM-normalised
    approximate raw counts.

    \b
    Typical workflow::

    regulonado normalization original tracks/discovered.parquet
    regulonado normalization tmm tracks/discovered.parquet --intervals intervals.bed
    """
    from regulonado.normalization import compute_tmm_factors, save_scale_factors
    from regulonado.qc import intervals_from_bed, load_interval_means, track_interval_means

    included = _included_tracks(track_table)
    if interval_means is None and intervals is None:
        typer.echo("Provide --intervals or --interval-means.", err=True)
        raise typer.Exit(1)

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = (
        scale_factors if scale_factors is not None else track_table.parent / f"scale_factors.{ext}"
    )
    out_path = output if output is not None else sf_path

    if not sf_path.exists():
        typer.echo(
            f"Scale-factors file not found: {sf_path}\n"
            "Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)

    sf_df = pd.read_parquet(sf_path) if str(sf_path).endswith(".parquet") else pd.read_csv(sf_path)

    if "library_size" not in sf_df.columns:
        typer.echo("Column 'library_size' missing from scale-factors file.", err=True)
        raise typer.Exit(1)
    if "scale_factor" not in sf_df.columns:
        typer.echo(
            (
                "Column 'scale_factor' missing from scale-factors file.  "
                "Run 'regulonado normalization original' first."
            ),
            err=True,
        )
        raise typer.Exit(1)

    sf_df = sf_df.sort_values("track_index").reset_index(drop=True)
    library_sizes = sf_df["library_size"].to_numpy(dtype=float)
    if len(library_sizes) != len(included):
        typer.echo(
            f"Track count mismatch: track table has {len(included)} tracks, "
            f"scale-factors file has {len(library_sizes)}.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Track table : {track_table}")
    typer.echo(f"Tracks      : {len(library_sizes)}")
    typer.echo(f"Bin size    : {bin_size} bp")
    typer.echo("")

    if interval_means is not None:
        means = load_interval_means(interval_means, included["track_name"].tolist())
    else:
        windows = intervals_from_bed(
            intervals, n_pred_bins=n_pred_bins, bin_size=bin_size, shift_max_bp=shift_max_bp,
            sample_n=sample_n,
        )
        means = track_interval_means(
            included["resolved_path"].tolist(), windows, max_workers=max_workers
        )
    shift_bins = shift_max_bp // bin_size
    region_length_kb = (n_pred_bins + 2 * shift_bins) * bin_size / 1000.0
    typer.echo(f"Regions     : {means.shape[0]}  ({region_length_kb:.1f} kb each)")
    typer.echo("")

    tmm = compute_tmm_factors(
        means,
        library_sizes,
        region_length_kb,
        trim_m=trim_m,
        trim_a=trim_a,
        min_count=min_count,
    )

    # Report
    typer.echo(
        f"{'Track':>5}  {'track_name':<30}  {'tmm_factor':>12}  {'old_sf':>12}  {'new_sf':>12}"
    )
    for i, (_, row) in enumerate(sf_df.iterrows()):
        old_sf = float(row["scale_factor"])
        new_sf = old_sf / tmm[i]
        name = str(included.loc[i, "track_name"])[:30]
        typer.echo(
            f"{int(row['track_index']):>5}  {name:<30}  {tmm[i]:>12.6f}  "
            f"{old_sf:>12.6f}  {new_sf:>12.6f}"
        )

    # Write updated parquet: add tmm_factor, overwrite scale_factor
    sf_df["tmm_factor"] = tmm
    sf_df["scale_factor"] = sf_df["scale_factor"] / sf_df["tmm_factor"]
    if "track_name" not in sf_df.columns:
        sf_df["track_name"] = included["track_name"].to_numpy()

    priority = ["track_index", "track_name", "scale_factor", "tmm_factor", "clip_soft", "clip_hard"]
    rest = [c for c in sf_df.columns if c not in priority]
    sf_df = sf_df[priority + rest]

    save_scale_factors(sf_df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"\nSaved updated scale factors to {out_path}")


@normalization_app.command("seqnado")
def calculate_seqnado_scaling(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet (or any tracks.parquet)")
    ],
    project: Annotated[
        Path,
        typer.Option("--project", help="SeqNado output directory (seqnado_output/)."),
    ],
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help="Scale-factors parquet from 'normalization original'.",
        ),
    ] = None,
    method: Annotated[
        Optional[str],
        typer.Option(
            "--method", "-m", help="Spike-in method, e.g. orlando. Default: the only one."
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path (default: overwrites --scale-factors)."),
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Output format: csv or parquet")
    ] = "parquet",
) -> None:
    """Apply SeqNado's spike-in normalisation factors instead of estimating our own.

    Takes library-size factors from 'normalization original' and corrects them
    with the factors SeqNado already computed, exactly as 'normalization tmm'
    applies a TMM correction: ``scale_factor = old_scale_factor / factor``, with
    factors normalised to a geometric mean of 1 so the overall magnitude is
    unchanged.

    Only valid within a single SeqNado project — its factors are not comparable
    across projects. Use 'tmm' when aggregating several.
    """
    import numpy as np
    from regulonado.normalization import save_scale_factors

    included = _included_tracks(track_table)

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = (
        scale_factors if scale_factors is not None else track_table.parent / f"scale_factors.{ext}"
    )
    if not sf_path.exists():
        typer.echo(
            f"Scale factors not found: {sf_path}. Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)
    out_path = output if output is not None else sf_path

    from regulonado._seqnado import SeqNadoUnavailableError, open_project

    try:
        seqnado_project = open_project(project)
        factors = seqnado_project.load_normalisation_factors(method)
    except SeqNadoUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        typer.echo(
            f"No normalisation factors in {project}: {exc}. SeqNado writes these "
            f"under resources/<method>/normalisation_factors.tsv when spike-in "
            f"normalisation is enabled.",
            err=True,
        )
        raise typer.Exit(1) from exc

    sample_column = next((c for c in ("sample", "sample_id", "sample_name") if c in factors), None)
    factor_column = next(
        (c for c in ("norm_factor", "scale_factor", "factor") if c in factors), None
    )
    if sample_column is None or factor_column is None:
        typer.echo(
            f"Could not find sample and factor columns in SeqNado's normalisation "
            f"table; got columns: {', '.join(map(str, factors.columns))}",
            err=True,
        )
        raise typer.Exit(1)

    # load_normalisation_factors(None) concatenates every method's table and
    # tags each row with its method. Silently keeping the first row per sample
    # would pick a spike-in method at random, so make the caller choose.
    if method is None and "method" in factors.columns:
        present = sorted(factors["method"].dropna().unique())
        if len(present) > 1:
            typer.echo(
                f"{project} has normalisation factors for several spike-in methods "
                f"({', '.join(map(str, present))}). Pass --method to choose one.",
                err=True,
            )
            raise typer.Exit(1)

    lookup = (
        factors.dropna(subset=[sample_column, factor_column])
        .drop_duplicates(subset=[sample_column], keep="first")
        .set_index(sample_column)[factor_column]
        .astype(float)
    )

    # Tracks carry both names; SeqNado keys its table on the sample name, which
    # for IP assays is '<sample>_<ip>' — the same string as the bigwig stem.
    names = [
        row.get("sample_id") or row["track_name"] for _, row in included.iterrows()
    ]
    missing = [name for name in names if name not in lookup.index]
    if missing:
        available = ", ".join(map(str, lookup.index[:8]))
        typer.echo(
            f"{len(missing)}/{len(names)} track(s) have no SeqNado normalisation "
            f"factor: {', '.join(missing[:8])}.\nFactors are available for: {available}",
            err=True,
        )
        raise typer.Exit(1)

    values = np.asarray([lookup[name] for name in names], dtype=float)
    if np.any(values <= 0):
        typer.echo("SeqNado normalisation factors must be positive.", err=True)
        raise typer.Exit(1)
    # Normalise to geometric mean 1, matching how the TMM correction is applied.
    values = values / float(np.exp(np.mean(np.log(values))))

    df = pd.read_parquet(sf_path) if fmt == "parquet" else pd.read_csv(sf_path)
    df = df.sort_values("track_index").reset_index(drop=True)
    if len(df) != len(included):
        typer.echo(
            f"Scale factors have {len(df)} row(s) but track table has {len(included)} track(s).",
            err=True,
        )
        raise typer.Exit(1)

    df["seqnado_norm_factor"] = values
    df["scale_factor"] = df["scale_factor"] / values
    if "track_name" not in df.columns:
        df["track_name"] = included["track_name"].to_numpy()

    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Applied {len(values)} SeqNado normalisation factor(s) -> {out_path}")


@normalization_app.command("bamnado")
def calculate_bamnado_scaling(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet (or any tracks.parquet)")
    ],
    bam_dir: Annotated[
        Path,
        typer.Option(
            "--bam-dir",
            help="Directory of BAM files, one per track, named <track-stem>.bam",
        ),
    ],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help=("bamnado bam-normalize method: tmm, csaw-background, cpm, "
                  "median-of-ratios, spike-in"),
        ),
    ] = "csaw-background",
    scale_factors: Annotated[
        Optional[Path],
        typer.Option(
            "--scale-factors",
            "-s",
            help=(
                "Scale-factors parquet from normalization original "
                "(default: <metadata_dir>/scale_factors.parquet)"
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: overwrites --scale-factors input)",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: csv or parquet"),
    ] = "parquet",
    bin_size_bp: Annotated[
        int,
        typer.Option("--bin-size-bp", help="bamnado background bin size in bp"),
    ] = 10_000,
    exclude_top_percent: Annotated[
        float,
        typer.Option(
            "--exclude-top-percent", help="Drop this percentage of highest-count bins first"
        ),
    ] = 5.0,
    reference_sample: Annotated[
        Optional[str],
        typer.Option(
            "--reference-sample", help="Sample name to use as reference (default: bamnado's choice)"
        ),
    ] = None,
    logratio_trim: Annotated[
        float,
        typer.Option("--logratio-trim", help="TMM trim fraction for M-values"),
    ] = 0.3,
    sum_trim: Annotated[
        float,
        typer.Option("--sum-trim", help="TMM trim fraction for A-values"),
    ] = 0.05,
    exogenous_prefix: Annotated[
        Optional[str],
        typer.Option(
            "--exogenous-prefix",
            help="Reference-name prefix for spike-in sequences (--method spike-in)",
        ),
    ] = None,
) -> None:
    """Correct scale factors using bamnado's own between-sample normalisation.

    Runs ``bamnado bam-normalize`` directly on BAM files to get a correction
    factor per track, then divides the existing scale_factor (from
    'normalization original') by that correction — same convention as
    'normalization tmm', but the correction comes from bamnado's TMM,
    csaw-background, CPM, median-of-ratios, or spike-in estimator over the
    full BAM rather than regulonado's own dataset-restricted TMM.

    BAM files are matched to tracks by filename stem: track N's bigwig
    'sample1.bw' must have a matching 'sample1.bam' in --bam-dir.

    """
    from regulonado.normalization import compute_bamnado_norm_factors, save_scale_factors

    if not bam_dir.is_dir():
        typer.echo(f"BAM directory not found: {bam_dir}", err=True)
        raise typer.Exit(1)

    included = _included_tracks(track_table)

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = (
        scale_factors if scale_factors is not None else track_table.parent / f"scale_factors.{ext}"
    )
    out_path = output if output is not None else sf_path

    if not sf_path.exists():
        typer.echo(
            f"Scale-factors file not found: {sf_path}\n"
            "Run 'regulonado normalization original' first.",
            err=True,
        )
        raise typer.Exit(1)

    bam_paths = []
    missing = []
    for resolved_path in included["resolved_path"]:
        stem = Path(resolved_path).stem
        bam_path = bam_dir / f"{stem}.bam"
        if not bam_path.exists():
            missing.append(str(bam_path))
        bam_paths.append(bam_path)
    if missing:
        typer.echo(
            "Missing BAM file(s) for these tracks (expected filename stem to match the bigwig):\n"
            + "\n".join(f"  {m}" for m in missing),
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Track table : {track_table}")
    typer.echo(f"BAM dir     : {bam_dir}")
    typer.echo(f"Method      : {method}")
    typer.echo(f"Tracks      : {len(bam_paths)}")
    typer.echo("")

    norm_factors = compute_bamnado_norm_factors(
        bam_paths,
        method=method,
        bin_size_bp=bin_size_bp,
        exclude_top_percent=exclude_top_percent,
        reference_sample=reference_sample,
        logratio_trim=logratio_trim,
        sum_trim=sum_trim,
        exogenous_prefix=exogenous_prefix,
    )

    sf_df = pd.read_parquet(sf_path) if str(sf_path).endswith(".parquet") else pd.read_csv(sf_path)
    if "scale_factor" not in sf_df.columns:
        typer.echo(
            (
                "Column 'scale_factor' missing from scale-factors file.  "
                "Run 'regulonado normalization original' first."
            ),
            err=True,
        )
        raise typer.Exit(1)
    if len(sf_df) != len(norm_factors):
        typer.echo(
            f"Track count mismatch: scale-factors file has {len(sf_df)}, "
            f"bamnado returned {len(norm_factors)}.",
            err=True,
        )
        raise typer.Exit(1)

    sf_df = sf_df.sort_values("track_index").reset_index(drop=True)
    sf_df["bamnado_method"] = method
    sf_df["bamnado_norm_factor"] = norm_factors
    sf_df["scale_factor"] = sf_df["scale_factor"] / sf_df["bamnado_norm_factor"]
    if "track_name" not in sf_df.columns:
        sf_df["track_name"] = included["track_name"].to_numpy()

    typer.echo(
        f"{'Track':>5}  {'track_name':<30}  {'norm_factor':>12}  {'old_sf':>12}  {'new_sf':>12}"
    )
    for i, row in sf_df.iterrows():
        old_sf = float(row["scale_factor"]) * float(row["bamnado_norm_factor"])
        name = str(row["track_name"])[:30]
        typer.echo(
            f"{int(row['track_index']):>5}  {name:<30}  {norm_factors[i]:>12.6f}  "
            f"{old_sf:>12.6f}  {float(row['scale_factor']):>12.6f}"
        )

    priority = ["track_index", "track_name", "scale_factor", "bamnado_norm_factor", "clip_soft",
                "clip_hard"]
    rest = [c for c in sf_df.columns if c not in priority]
    sf_df = sf_df[priority + rest]

    save_scale_factors(sf_df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"\nSaved updated scale factors to {out_path}")
