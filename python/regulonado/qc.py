"""Opt-in track QC: signal-quality checks that run before the Arrow build.

Four check families, one shared interval scan (:func:`track_interval_means`)
serving both ``tmm`` scaling and the ``interval_signal``/``replicate_concordance``
checks — as separate BigWig scans they would each pay the same I/O twice.
``anchor`` reuses the scale-factor table's own diagnostics rather than a second
pass. See docs/qc.md for what each check means per assay.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CHECKS = ("sparsity", "interval_signal", "replicate_concordance", "anchor")

# Diagnostics the anchor check reuses straight from the scale-factor table —
# it is free under anchor scaling and never triggers its own BigWig scan.
ANCHOR_SCALE_COLUMNS = (
    "scale_quality",
    "scale_heldout_recovery",
    "scale_anchor_reference",
    "scale_background_q50",
    "scale_background_q99",
)


def intervals_from_bed(
    bed_file: str | Path,
    *,
    n_pred_bins: int,
    bin_size: int,
    shift_max_bp: int,
    sample_n: int | None = None,
    seed: int = 0,
) -> list[tuple[str, int, int]]:
    """Signal windows for QC/scaling, reusing the build's own region geometry."""
    from regulonado.dataset.build import _load_bed_rows, _signal_intervals  # noqa: PLC0415

    bed_rows = _load_bed_rows(bed_file)
    intervals = _signal_intervals(bed_rows, n_pred_bins, bin_size, shift_max_bp)
    if sample_n is not None and len(intervals) > sample_n:
        rng = np.random.default_rng(seed)
        intervals = [intervals[i] for i in rng.choice(len(intervals), sample_n, replace=False)]
    return intervals


def track_interval_means(
    track_paths: Sequence[str | Path],
    intervals: Sequence[tuple[str, int, int]],
    *,
    max_workers: int = 16,
) -> np.ndarray:
    """Mean signal per (interval, track): ``(n_regions, n_tracks)``.

    One BigWig scan per track, shared by ``tmm`` scaling and the
    ``interval_signal``/``replicate_concordance`` QC checks.
    """
    import pybigtools

    windows = list(intervals)

    def scan(path: str | Path) -> np.ndarray:
        reader = pybigtools.open(str(path))
        try:
            means = np.array(
                [
                    float(reader.values(c, s, e, bins=1, summary="mean", exact=True)[0])
                    for c, s, e in windows
                ],
                dtype=np.float32,
            )
        finally:
            close = getattr(reader, "close", None)
            if close:
                close()
        np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return means

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        columns = list(executor.map(scan, track_paths))
    return np.stack(columns, axis=1)


def save_interval_means(means: np.ndarray, track_names: Sequence[str], path: str | Path) -> None:
    """Write the shared ``tracks/_stages/interval_means.parquet``.

    One row per track, its per-region vector as a list column — built once and
    read by both ``tmm`` scaling and the ``interval_signal``/
    ``replicate_concordance`` QC checks, so neither re-scans the BigWigs.
    """
    df = pd.DataFrame({"track_name": list(track_names), "values": list(means.T)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_interval_means(path: str | Path, track_names: Sequence[str]) -> np.ndarray:
    """Read ``interval_means.parquet`` as ``(n_regions, n_tracks)``, ordered by ``track_names``."""
    df = pd.read_parquet(path).set_index("track_name")
    missing = [name for name in track_names if name not in df.index]
    if missing:
        raise ValueError(f"interval_means.parquet is missing track(s): {missing}")
    columns = [np.asarray(df.loc[name, "values"], dtype=np.float32) for name in track_names]
    return np.stack(columns, axis=1)


def header_stats(path: str | Path) -> dict[str, object]:
    """Free, header-only metrics: always reported as context, gate only under ``sparsity``."""
    import pybigtools

    reader = pybigtools.open(str(path))
    try:
        summary = reader.info()["summary"]
        n_with_signal = 0
        for chrom, size in reader.chroms().items():
            value = reader.values(chrom, 0, size, bins=1, summary="mean", exact=True)[0]
            if value and not np.isnan(value):
                n_with_signal += 1
    finally:
        close = getattr(reader, "close", None)
        if close:
            close()
    return {
        "qc_bases_covered": int(summary.get("basesCovered", 0)),
        "qc_genome_sum": float(summary.get("sum", 0.0)),
        "qc_genome_min": float(summary.get("min", 0.0)),
        "qc_genome_max": float(summary.get("max", 0.0)),
        "qc_n_chroms_with_signal": n_with_signal,
        "qc_is_constant": bool(summary.get("min") == summary.get("max")),
    }


def sparsity_passes(
    stats: dict[str, object],
    *,
    min_bases_covered: int = 1,
    min_chroms_with_signal: int = 1,
) -> bool:
    """Free gate on the header stats: catches empty or constant BigWigs."""
    return (
        int(stats["qc_bases_covered"]) >= min_bases_covered
        and int(stats["qc_n_chroms_with_signal"]) >= min_chroms_with_signal
        and not bool(stats["qc_is_constant"])
    )


def interval_signal_stats(means: np.ndarray) -> list[dict[str, float]]:
    """Per-track distributional metrics over the sampled regions in ``means``."""
    n_regions = means.shape[0]
    top_k = max(1, int(np.ceil(0.01 * n_regions)))
    out: list[dict[str, float]] = []
    for column in means.T:
        q50 = float(np.quantile(column, 0.5))
        q99 = float(np.quantile(column, 0.99))
        total = float(column.sum())
        top1pct = float(np.sort(column)[-top_k:].sum() / total) if total > 0 else float("nan")
        out.append(
            {
                "qc_nonzero_bin_fraction": float(np.mean(column > 0)),
                "qc_interval_mean": float(column.mean()),
                "qc_interval_sd": float(column.std()),
                "qc_dynamic_range": (q99 / q50) if q50 > 0 else float("nan"),
                "qc_top1pct_signal_fraction": top1pct,
            }
        )
    return out


def replicate_concordance_stats(
    means: np.ndarray,
    group_labels: Sequence[str | None],
) -> list[dict[str, object]]:
    """Per-track Pearson concordance to same-group tracks.

    With no group labels this degrades to ``qc_nearest_other_r`` alone (the
    deeptools ``multiBigwigSummary`` + ``plotCorrelation`` equivalent, minus
    the replicate labelling) and ``qc_replicate_r``/``qc_nearest_is_replicate``
    stay null.
    """
    n_tracks = means.shape[1]
    corr = np.corrcoef(means, rowvar=False)
    has_groups = any(label is not None for label in group_labels)
    out: list[dict[str, object]] = []
    for i in range(n_tracks):
        others = [j for j in range(n_tracks) if j != i]
        if not others:
            out.append(
                {
                    "qc_replicate_r": None,
                    "qc_nearest_other_r": None,
                    "qc_nearest_is_replicate": None,
                }
            )
            continue
        nearest_j = max(others, key=lambda j: corr[i, j])
        nearest_r = float(corr[i, nearest_j])
        replicate_r = None
        nearest_is_replicate = None
        if has_groups and group_labels[i] is not None:
            same_group = [j for j in others if group_labels[j] == group_labels[i]]
            replicate_r = float(np.median([corr[i, j] for j in same_group])) if same_group else None
            nearest_is_replicate = bool(group_labels[nearest_j] == group_labels[i])
        out.append(
            {
                "qc_replicate_r": replicate_r,
                "qc_nearest_other_r": nearest_r,
                "qc_nearest_is_replicate": nearest_is_replicate,
            }
        )
    return out


def evaluate_rules(
    df: pd.DataFrame,
    rules: dict[str, dict[str, float]],
) -> tuple[pd.Series, pd.Series]:
    """Row-wise ``{column: {"min": x, "max": y}}`` threshold evaluation.

    Returns ``(qc_verdict, qc_failed_rules)`` where a row with no configured
    rule violated is ``"passed"`` with an empty failed-rules list.
    """
    failed: list[list[str]] = [[] for _ in range(len(df))]
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        fails = pd.Series(False, index=df.index)
        if "min" in bounds:
            fails |= values < bounds["min"]
        if "max" in bounds:
            fails |= values > bounds["max"]
        for position in np.flatnonzero(fails.fillna(False).to_numpy()):
            failed[position].append(column)
    verdict = pd.Series(["failed" if names else "passed" for names in failed], index=df.index)
    failed_rules = pd.Series(failed, index=df.index)
    return verdict, failed_rules
