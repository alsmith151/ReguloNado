"""``regulonado tracks``: discover, QC, and assemble the one canonical track table.

Five subcommands mirror the target DAG in the track-table design: ``discover``
resolves raw BigWig sources into ``tracks/_stages/discovered.parquet``; ``qc``
computes opt-in signal-quality metrics into a report; ``assemble`` merges
discovery + scaling + QC into ``tracks.parquet`` (the only file anything else
should read); ``show``/``targets`` are read-only views. See CLAUDE.md's "Track
table" section and docs/track-table.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

tracks_app = typer.Typer(
    no_args_is_help=True,
    help="Discover, QC, and assemble tracks.parquet before the dataset build.",
)

# Bare scale_factors.parquet column -> its namespaced name in tracks.parquet.
# 'path'/'resolved_path'/'track_index' are dropped before merging: they are
# already on the discovery side and would otherwise collide.
_SCALE_COLUMN_RENAME = {
    "background": "scale_background",
    "clip_soft": "scale_clip_soft",
    "clip_hard": "scale_clip_hard",
    "anchor_reference": "scale_anchor_reference",
    "background_q50": "scale_background_q50",
    "background_q99": "scale_background_q99",
    "heldout_reference": "scale_heldout_reference",
    "heldout_recovery": "scale_heldout_recovery",
    "window_stat_bp": "scale_window_stat_bp",
    "quality": "scale_quality",
    "tmm_factor": "scale_tmm_factor",
    "library_size": "scale_library_size",
    "bamnado_method": "scale_bamnado_method",
    "bamnado_norm_factor": "scale_bamnado_norm_factor",
    "seqnado_norm_factor": "scale_seqnado_norm_factor",
    "samplename": "scale_samplename",
}

# The full canonical column set, always present in tracks.parquet regardless of
# which stages ran — an unrun stage leaves its columns null rather than absent,
# so a reader never has to branch on which stages were enabled ("stable schema").
_CANONICAL_SCALE_COLUMNS = tuple(_SCALE_COLUMN_RENAME.values()) + ("scale_factor",)
_CANONICAL_QC_COLUMNS = (
    "qc_verdict",
    "qc_failed_rules",
    "qc_bases_covered",
    "qc_genome_sum",
    "qc_genome_min",
    "qc_genome_max",
    "qc_n_chroms_with_signal",
    "qc_is_constant",
    "qc_nonzero_bin_fraction",
    "qc_interval_mean",
    "qc_interval_sd",
    "qc_dynamic_range",
    "qc_top1pct_signal_fraction",
    "qc_replicate_r",
    "qc_nearest_other_r",
    "qc_nearest_is_replicate",
    "qc_anchor_verdict",
)


def _parse_seqnado_projects(values: Optional[list[str]]) -> dict[str, str]:
    """Parse repeated ``PATH`` / ``NAME=PATH`` project options.

    Without an explicit name a project is labelled by the directory containing
    its output dir, which for a SeqNado layout is the project folder itself
    (``2026-08-10_myproj/seqnado_output`` -> ``2026-08-10_myproj``).
    """
    projects: dict[str, str] = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep:
            path = name
            name = Path(path).expanduser().resolve().parent.name
        if not path:
            raise typer.BadParameter(
                f"Expected PATH or NAME=PATH, got {value!r}", param_hint="--seqnado-project"
            )
        if name in projects:
            raise typer.BadParameter(
                f"Duplicate project name {name!r}; give each one an explicit NAME=PATH",
                param_hint="--seqnado-project",
            )
        projects[name] = path
    return projects


def _read_table(path: Path):
    from regulonado.tracks_table import read_track_table

    if not path.exists():
        typer.echo(f"Track table not found: {path}", err=True)
        raise typer.Exit(1)
    return read_track_table(path)


@tracks_app.command("discover")
def discover(
    output: Annotated[Path, typer.Argument(help="Where to write discovered.parquet")],
    bigwig: Annotated[
        Optional[list[Path]],
        typer.Option("--bigwig", "-b", help="BigWig file (repeat, order preserved)"),
    ] = None,
    bigwig_dir: Annotated[
        Optional[Path],
        typer.Option("--bigwig-dir", help="Directory of BigWig files (sorted by name)"),
    ] = None,
    bigwig_glob: Annotated[
        Optional[list[str]],
        typer.Option(
            "--bigwig-glob",
            help="Glob for --bigwig-dir (repeatable). Default: '*.bw' and '*.bigWig'.",
        ),
    ] = None,
    track_sheet: Annotated[
        Optional[Path],
        typer.Option(
            "--track-sheet",
            help="CSV mapping tracks to annotation; supplies the ordered track list.",
        ),
    ] = None,
    seqnado_project: Annotated[
        Optional[list[str]],
        typer.Option(
            "--seqnado-project",
            help="SeqNado output directory as PATH or NAME=PATH (repeatable).",
        ),
    ] = None,
    drop_missing: Annotated[
        bool, typer.Option("--drop-missing", help="Drop missing BigWig paths instead of raising")
    ] = False,
    dedupe_tracks: Annotated[
        str, typer.Option("--dedupe-tracks", help="none, identity, or content")
    ] = "none",
) -> None:
    """Resolve raw BigWig sources into the discovery-stage track table.

    Precedence: an explicit ``--track-sheet``/``--seqnado-project`` combination,
    else ``--bigwig-dir``, else explicit ``--bigwig`` files.
    """
    from regulonado.dataset.discovery import discover_tracks
    from regulonado.tracks_table import write_track_table

    projects = _parse_seqnado_projects(seqnado_project)
    annotations: Optional[dict] = None

    if track_sheet is not None or projects:
        from regulonado.tracks import TrackSheet

        try:
            if track_sheet is not None:
                sheet = TrackSheet.from_csv(track_sheet, projects=projects or None)
            else:
                sheet = TrackSheet.from_seqnado_projects(
                    [{"name": name, "path": path} for name, path in projects.items()]
                )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        bw_paths: list[str] = [str(p) for p in sheet.bigwig_paths]
        annotations = sheet.annotations_by_path()
    elif bigwig_dir is not None:
        globs = list(bigwig_glob) if bigwig_glob else ["*.bw", "*.bigWig"]
        matched = {p for pattern in globs for p in bigwig_dir.glob(pattern)}
        bw_paths = [str(p) for p in sorted(matched)]
        if not bw_paths:
            typer.echo(f"No files matching {globs} in {bigwig_dir}", err=True)
            raise typer.Exit(1)
    elif bigwig:
        bw_paths = [str(p) for p in bigwig]
    else:
        typer.echo(
            "Provide --bigwig files, --bigwig-dir, --track-sheet or --seqnado-project.",
            err=True,
        )
        raise typer.Exit(1)

    df = discover_tracks(
        bw_paths, drop_missing=drop_missing, dedupe_tracks=dedupe_tracks, annotations=annotations
    )
    write_track_table(df, output)
    n_included = int((df["status"] == "included").sum())
    typer.echo(f"Discovered {len(df)} track(s), {n_included} included -> {output}")


@tracks_app.command("interval-means")
def interval_means_cmd(
    track_table: Annotated[Path, typer.Argument(help="tracks/_stages/discovered.parquet")],
    output: Annotated[Path, typer.Option("-o", "--output", help="interval_means.parquet")],
    intervals: Annotated[Path, typer.Option("--intervals", help="inputs.intervals BED")],
    bin_size: Annotated[int, typer.Option("--bin-size")] = 32,
    n_pred_bins: Annotated[int, typer.Option("--n-pred-bins")] = 6_144,
    shift_max_bp: Annotated[int, typer.Option("--shift-max-bp")] = 0,
    sample_n: Annotated[Optional[int], typer.Option("--sample-n")] = None,
    max_workers: Annotated[int, typer.Option("--workers", "-w")] = 16,
) -> None:
    """One shared BigWig scan, consumed by ``tmm`` scaling and interval-based QC checks."""
    from regulonado import qc as qc_module

    included = _read_table(track_table)
    included = included[included["status"] == "included"].sort_values("track_index")
    windows = qc_module.intervals_from_bed(
        intervals, n_pred_bins=n_pred_bins, bin_size=bin_size, shift_max_bp=shift_max_bp,
        sample_n=sample_n,
    )
    means = qc_module.track_interval_means(
        included["resolved_path"].tolist(), windows, max_workers=max_workers
    )
    qc_module.save_interval_means(means, included["track_name"].tolist(), output)
    typer.echo(f"Wrote {means.shape[0]} region(s) x {means.shape[1]} track(s) -> {output}")


def _parse_rules(values: Optional[list[str]]) -> dict[str, dict[str, float]]:
    """Parse repeated ``COLUMN:min=X,max=Y`` rule options."""
    rules: dict[str, dict[str, float]] = {}
    for value in values or []:
        column, sep, bounds_str = value.partition(":")
        if not sep:
            raise typer.BadParameter(f"Expected COLUMN:min=X,max=Y, got {value!r}")
        bounds: dict[str, float] = {}
        for part in bounds_str.split(","):
            if not part:
                continue
            key, _, num = part.partition("=")
            if key not in {"min", "max"}:
                raise typer.BadParameter(f"Rule bound must be 'min' or 'max', got {key!r}")
            bounds[key] = float(num)
        rules[column] = bounds
    return rules


@tracks_app.command("qc")
def qc_cmd(
    track_table: Annotated[
        Path, typer.Argument(help="tracks/_stages/discovered.parquet")
    ],
    output: Annotated[Path, typer.Option("-o", "--output", help="qc_report.parquet")],
    check: Annotated[
        Optional[list[str]],
        typer.Option(
            "--check", help="Repeatable: sparsity, interval_signal, replicate_concordance, anchor"
        ),
    ] = None,
    intervals: Annotated[
        Optional[Path],
        typer.Option("--intervals", help="BED for interval_signal/replicate_concordance"),
    ] = None,
    interval_means: Annotated[
        Optional[Path],
        typer.Option(
            "--interval-means",
            help="Precomputed 'tracks interval-means' output; skips a fresh scan when given",
        ),
    ] = None,
    scale_factors: Annotated[
        Optional[Path], typer.Option("--scale-factors", help="Needed for --check anchor")
    ] = None,
    bin_size: Annotated[int, typer.Option("--bin-size")] = 32,
    n_pred_bins: Annotated[int, typer.Option("--n-pred-bins")] = 6_144,
    shift_max_bp: Annotated[int, typer.Option("--shift-max-bp")] = 0,
    sample_n: Annotated[
        Optional[int], typer.Option("--sample-n", help="Sample this many BED rows")
    ] = None,
    rule: Annotated[
        Optional[list[str]],
        typer.Option("--rule", help="COLUMN:min=X,max=Y (repeatable)"),
    ] = None,
    max_workers: Annotated[int, typer.Option("--workers", "-w")] = 16,
) -> None:
    """Compute opt-in QC metrics and a verdict; does not drop or mutate tracks.

    Dropping (``qc.drop_degenerate`` + ``qc.rules``) happens in ``tracks
    assemble``, which is the step that always runs and so is the one place a
    track's ``status`` is decided.
    """
    from regulonado import qc as qc_module

    checks = list(check) if check else []
    unknown = sorted(set(checks) - set(qc_module.CHECKS))
    if unknown:
        typer.echo(f"Unknown check(s): {unknown}; choose from {qc_module.CHECKS}", err=True)
        raise typer.Exit(1)

    table = _read_table(track_table)
    included = table[table["status"] == "included"].sort_values("track_index").reset_index(
        drop=True
    )
    if included.empty:
        typer.echo("No included tracks to QC.", err=True)
        raise typer.Exit(1)

    metrics = pd.DataFrame(
        [qc_module.header_stats(p) for p in included["resolved_path"]], index=included.index
    )

    if "interval_signal" in checks or "replicate_concordance" in checks:
        if interval_means is not None:
            means = qc_module.load_interval_means(interval_means, included["track_name"].tolist())
        elif intervals is not None:
            windows = qc_module.intervals_from_bed(
                intervals, n_pred_bins=n_pred_bins, bin_size=bin_size, shift_max_bp=shift_max_bp,
                sample_n=sample_n,
            )
            means = qc_module.track_interval_means(
                included["resolved_path"].tolist(), windows, max_workers=max_workers
            )
        else:
            typer.echo(
                "--intervals or --interval-means is required for "
                "interval_signal/replicate_concordance",
                err=True,
            )
            raise typer.Exit(1)
        if "interval_signal" in checks:
            sig = pd.DataFrame(qc_module.interval_signal_stats(means), index=included.index)
            metrics = pd.concat([metrics, sig], axis=1)
        if "replicate_concordance" in checks:
            group_col = next(
                (c for c in ("scaling_group", "condition") if c in included.columns), None
            )
            labels = included[group_col].where(included[group_col].notna(), None).tolist() \
                if group_col else [None] * len(included)
            rep = pd.DataFrame(
                qc_module.replicate_concordance_stats(means, labels), index=included.index
            )
            metrics = pd.concat([metrics, rep], axis=1)

    if "anchor" in checks:
        if scale_factors is None:
            typer.echo("--scale-factors is required for --check anchor", err=True)
            raise typer.Exit(1)
        sf = (
            pd.read_parquet(scale_factors)
            if str(scale_factors).endswith(".parquet")
            else pd.read_csv(scale_factors)
        )
        by_name = sf.set_index("track_name") if "track_name" in sf.columns else None
        if by_name is None or "quality" not in by_name.columns:
            typer.echo("--scale-factors has no 'track_name'/'quality' column", err=True)
            raise typer.Exit(1)
        metrics["qc_anchor_verdict"] = [
            by_name.loc[name, "quality"] if name in by_name.index else None
            for name in included["track_name"]
        ]

    rules = _parse_rules(rule)
    if "sparsity" in checks:
        rules.setdefault("qc_bases_covered", {"min": 1})
        rules.setdefault("qc_n_chroms_with_signal", {"min": 1})
    verdict, failed_rules = qc_module.evaluate_rules(metrics, rules)
    if "qc_anchor_verdict" in metrics.columns:
        anchor_failed = metrics["qc_anchor_verdict"] == "failed"
        verdict = verdict.where(~anchor_failed, "failed")
        failed_rules = pd.Series(
            [
                [*names, "qc_anchor_verdict"] if failed else names
                for names, failed in zip(failed_rules, anchor_failed)
            ],
            index=metrics.index,
        )

    report = pd.DataFrame({"track_name": included["track_name"]})
    report["qc_verdict"] = verdict.to_numpy()
    report["qc_failed_rules"] = failed_rules.to_numpy()
    report = pd.concat([report, metrics.add_prefix("").reset_index(drop=True)], axis=1)
    report = report.loc[:, ~report.columns.duplicated()]

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    report.to_parquet(tmp, index=False)
    tmp.replace(output)

    n_failed = int((report["qc_verdict"] == "failed").sum())
    typer.echo(f"QC'd {len(report)} track(s): {n_failed} failed -> {output}")


@tracks_app.command("assemble")
def assemble(
    discovered: Annotated[Path, typer.Argument(help="tracks/_stages/discovered.parquet")],
    output: Annotated[Path, typer.Option("-o", "--output", help="tracks.parquet")],
    scale_factors: Annotated[
        Optional[Path], typer.Option("--scale-factors", help="scale_factors.parquet")
    ] = None,
    qc_report: Annotated[
        Optional[Path], typer.Option("--qc-report", help="qc_report.parquet")
    ] = None,
    annotations: Annotated[
        Optional[Path],
        typer.Option(
            "--annotations",
            help="CSV/parquet with a 'track_name' column plus arbitrary extra columns (e.g. "
            "'group') to merge in — the way to attach grouping metadata after 'tracks discover' "
            "without redoing discovery from a hand-crafted --track-sheet.",
        ),
    ] = None,
    drop_degenerate: Annotated[
        bool, typer.Option("--drop-degenerate", help="Drop tracks with qc_verdict == 'failed'")
    ] = False,
    exclude_track: Annotated[
        Optional[list[str]],
        typer.Option("--exclude-track", help="track_name to force-exclude (repeatable)"),
    ] = None,
) -> None:
    """Join discovery + scaling + QC into ``tracks.parquet``, the one file everything else reads.

    ``tracks.parquet`` is a strict superset of every stage output: an unrun
    stage leaves its columns null rather than absent, so the schema is
    identical regardless of which stages ran.
    """
    from regulonado.tracks_table import read_track_table, write_track_table

    merged = read_track_table(discovered)

    scaling_method = None
    if scale_factors is not None:
        sf = (
            pd.read_parquet(scale_factors)
            if str(scale_factors).endswith(".parquet")
            else pd.read_csv(scale_factors)
        )
        sf = sf.drop(columns=[c for c in ("path", "resolved_path", "track_index") if c in sf], )
        sf = sf.rename(columns=_SCALE_COLUMN_RENAME)
        merged = merged.merge(sf, on="track_name", how="left", validate="one_to_one")
        scaling_method = "anchor" if "scale_anchor_reference" in merged.columns else "inferred"

    if qc_report is not None:
        qc_df = pd.read_parquet(qc_report) if str(qc_report).endswith(".parquet") \
            else pd.read_csv(qc_report)
        merged = merged.merge(qc_df, on="track_name", how="left", validate="one_to_one")

    if annotations is not None:
        ann_df = pd.read_parquet(annotations) if str(annotations).endswith(".parquet") \
            else pd.read_csv(annotations)
        if "track_name" not in ann_df.columns:
            typer.echo("--annotations file must have a 'track_name' column", err=True)
            raise typer.Exit(1)
        unknown_annotated = set(ann_df["track_name"]) - set(merged["track_name"])
        if unknown_annotated:
            typer.echo(
                f"--annotations track_name(s) not found: {sorted(unknown_annotated)}", err=True
            )
            raise typer.Exit(1)
        merged = merged.merge(ann_df, on="track_name", how="left", validate="one_to_one")

    for column in (*_CANONICAL_SCALE_COLUMNS, *_CANONICAL_QC_COLUMNS):
        if column not in merged.columns:
            merged[column] = None

    exclude = set(exclude_track or [])
    unknown_exclude = exclude - set(merged["track_name"])
    if unknown_exclude:
        typer.echo(f"--exclude-track name(s) not found: {sorted(unknown_exclude)}", err=True)
        raise typer.Exit(1)

    status = merged["status"].copy()
    status = status.where(~merged["track_name"].isin(exclude), "excluded")
    if drop_degenerate and "qc_verdict" in merged.columns:
        failed_qc = (status == "included") & (merged["qc_verdict"] == "failed")
        status = status.where(~failed_qc, "qc_failed")
    merged["status"] = status

    included_mask = (merged["status"] == "included").to_numpy()
    merged["track_index"] = pd.array([pd.NA] * len(merged), dtype="Int64")
    merged.loc[included_mask, "track_index"] = range(int(included_mask.sum()))

    write_track_table(merged, output, scaling_method=scaling_method)
    typer.echo(
        f"Assembled {len(merged)} track(s): {int(included_mask.sum())} included -> {output}"
    )


@tracks_app.command("show")
def show(track_table: Annotated[Path, typer.Argument()]) -> None:
    """Pretty-print a track table."""
    df = _read_table(track_table)
    columns = [
        c
        for c in ("track_index", "track_name", "status", "scale_factor", "qc_verdict")
        if c in df.columns
    ]
    typer.echo(df[columns].to_string(index=False))
    typer.echo("")
    typer.echo(df["status"].value_counts().to_string())


@tracks_app.command("targets")
def targets(
    track_table: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("-o", "--output", help="Borzoi-format targets.txt")],
) -> None:
    """Export a Borzoi-format ``targets.txt`` — interop, and a diffable text view.

    ``sum_stat``/``strand_pair`` have no ReguloNado equivalent: every row gets
    ``sum`` and self-pairs (``strand_pair == index``).
    """
    df = _read_table(track_table)
    included = df[df["status"] == "included"].sort_values("track_index")
    rows = [
        {
            "index": int(row["track_index"]),
            "identifier": row["track_name"],
            "file": row.get("resolved_path", ""),
            "clip": row.get("scale_clip_hard", ""),
            "clip_soft": row.get("scale_clip_soft", ""),
            "scale": row.get("scale_factor", ""),
            "sum_stat": "sum",
            "description": row.get("assay") or "",
            "strand_pair": int(row["track_index"]),
        }
        for _, row in included.iterrows()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    typer.echo(f"Wrote {len(rows)} target(s) -> {output}")
