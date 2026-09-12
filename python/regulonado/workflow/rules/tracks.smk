"""Track discovery, QC, and assembly into ``tracks.parquet`` — runs before the Arrow build.

Target DAG::

    track_discovery -> discovered.parquet
                             |
                   track_interval_means (shared scan, built only if needed)
                        +----+----+
              scale_factors      track_qc   <- parallel unless 'anchor' in qc.checks
                        +----+----+
                             v
                       track_assemble -> tracks.parquet
"""

import shlex

TRACKS_DIR = RESULTS / "tracks"
TRACKS_STAGE_DIR = TRACKS_DIR / "_stages"
QC = config.get("qc") or {"checks": []}
QC_CHECKS = QC.get("checks") or []
_NEEDS_INTERVAL_MEANS = bool({"interval_signal", "replicate_concordance"} & set(QC_CHECKS)) or (
    config["scaling"]["method"] == "tmm"
)


def track_source(input):
    """CLI flags naming where tracks and their annotation come from.

    Three sources, in precedence order: an explicit track sheet, one or more
    SeqNado projects, or a plain directory of BigWigs. A sheet and projects can
    be combined — rows that give only a `sample_id` are resolved against the
    named project.
    """
    parts = []
    if input.track_sheet:
        parts += ["--track-sheet", shlex.quote(str(input.track_sheet[0]))]
    for project in SEQNADO_PROJECTS:
        parts += ["--seqnado-project", shlex.quote(f"{project['name']}={project['path']}")]
    if not parts:
        parts += ["--bigwig-dir", shlex.quote(config["inputs"]["bigwig_dir"])]
    return " ".join(parts)


rule track_discovery:
    """Resolve raw BigWig sources into one row per track: included/dropped/missing."""
    input:
        track_sheet=(
            [config["inputs"]["track_sheet"]] if config["inputs"].get("track_sheet") else []
        ),
    params:
        track_source=lambda w, input: track_source(input),
        drop_missing=lambda w: "--drop-missing" if config["dataset"]["drop_missing"] else "",
        dedupe=config["dataset"]["dedupe_tracks"],
    output:
        table=str(TRACKS_STAGE_DIR / "discovered.parquet"),
    log:
        str(RESULTS / "logs" / "track_discovery.log"),
    shell:
        r"""
        regulonado tracks discover {output.table:q} \
            {params.track_source} \
            --dedupe-tracks {params.dedupe} \
            {params.drop_missing} \
            > {log:q} 2>&1
        """


if _NEEDS_INTERVAL_MEANS:

    rule track_interval_means:
        """One shared BigWig scan for 'tmm' scaling and interval-based QC checks."""
        input:
            table=str(TRACKS_STAGE_DIR / "discovered.parquet"),
            intervals=config["inputs"]["intervals"],
        params:
            bin_size=config["dataset"]["bin_size"],
            n_pred_bins=config["dataset"]["n_pred_bins"],
            shift_max_bp=config["dataset"]["shift_max_bp"],
            sample_n=QC.get("sample_windows") or 0,
            sample_arg=(
                f"--sample-n {QC['sample_windows']}" if QC.get("sample_windows") else ""
            ),
            max_workers=QC.get("max_workers", 16),
        output:
            table=str(TRACKS_STAGE_DIR / "interval_means.parquet"),
        log:
            str(RESULTS / "logs" / "track_interval_means.log"),
        shell:
            r"""
            regulonado tracks interval-means {input.table:q} \
                --output {output.table:q} \
                --intervals {input.intervals:q} \
                --bin-size {params.bin_size} \
                --n-pred-bins {params.n_pred_bins} \
                --shift-max-bp {params.shift_max_bp} \
                {params.sample_arg} \
                --workers {params.max_workers} \
                > {log:q} 2>&1
            """


rule track_qc:
    """Opt-in signal-quality checks; computes a verdict but never drops tracks itself."""
    input:
        table=str(TRACKS_STAGE_DIR / "discovered.parquet"),
        intervals=config["inputs"]["intervals"] if _NEEDS_INTERVAL_MEANS else [],
        interval_means=(
            str(TRACKS_STAGE_DIR / "interval_means.parquet") if _NEEDS_INTERVAL_MEANS else []
        ),
        scale_factors=(
            str(TRACKS_STAGE_DIR / "scale_factors.parquet") if "anchor" in QC_CHECKS else []
        ),
    params:
        checks=" ".join(f"--check {c}" for c in QC_CHECKS),
        interval_means_arg=(
            f"--interval-means {TRACKS_STAGE_DIR / 'interval_means.parquet'}"
            if _NEEDS_INTERVAL_MEANS
            else ""
        ),
        scale_factors_arg=(
            f"--scale-factors {TRACKS_STAGE_DIR / 'scale_factors.parquet'}"
            if "anchor" in QC_CHECKS
            else ""
        ),
        rules=" ".join(
            f"--rule {shlex.quote(column + ':' + ','.join(f'{k}={v}' for k, v in bounds.items()))}"
            for column, bounds in (QC.get("rules") or {}).items()
        ),
        max_workers=QC.get("max_workers", 16),
    output:
        report=str(TRACKS_STAGE_DIR / "qc_report.parquet"),
    log:
        str(RESULTS / "logs" / "track_qc.log"),
    shell:
        r"""
        regulonado tracks qc {input.table:q} \
            --output {output.report:q} \
            {params.checks} \
            {params.interval_means_arg} \
            {params.scale_factors_arg} \
            {params.rules} \
            --workers {params.max_workers} \
            > {log:q} 2>&1
        """


rule track_assemble:
    """Join discovery + scaling + QC into tracks.parquet, the one file everything else reads."""
    input:
        discovered=str(TRACKS_STAGE_DIR / "discovered.parquet"),
        scale_factors=str(TRACKS_STAGE_DIR / "scale_factors.parquet"),
        qc_report=str(TRACKS_STAGE_DIR / "qc_report.parquet") if QC_CHECKS else [],
        annotations=(
            [config["inputs"]["track_annotations"]]
            if config["inputs"].get("track_annotations")
            else []
        ),
    params:
        qc_report_arg=(
            f"--qc-report {TRACKS_STAGE_DIR / 'qc_report.parquet'}" if QC_CHECKS else ""
        ),
        annotations_arg=lambda w, input: (
            f"--annotations {shlex.quote(str(input.annotations[0]))}" if input.annotations else ""
        ),
        drop_degenerate=(
            "--drop-degenerate" if QC.get("drop_degenerate") and QC_CHECKS else ""
        ),
        exclude=" ".join(
            f"--exclude-track {shlex.quote(name)}"
            for name in config["inputs"].get("exclude_tracks") or []
        ),
    output:
        table=str(TRACKS_DIR / "tracks.parquet"),
    log:
        str(RESULTS / "logs" / "track_assemble.log"),
    shell:
        r"""
        regulonado tracks assemble {input.discovered:q} \
            --output {output.table:q} \
            --scale-factors {input.scale_factors:q} \
            {params.qc_report_arg} \
            {params.annotations_arg} \
            {params.drop_degenerate} \
            {params.exclude} \
            > {log:q} 2>&1
        """
