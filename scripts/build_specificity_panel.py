"""Build the fixed region panel that training scores target-group specificity against.

The panel is a set of genomic regions, each with an **observed** specificity score computed
from the atlas bigwigs, so a model's predicted group-contrast channels can be scored against
it at every evaluation (``trainer.specificity_panel_path``). Built once, offline, from the
unique-enhancer-finding multiscale run.

Predictions stay in distribution: the panel only uses the training dataset's own prediction
windows (``--intervals``, the BED the dataset was built from). Each enhancer region is
assigned the window it sits most centrally in, and every region scored lies inside one of
those windows:

* ``curated``      — the hand-selected k-means cluster (the 85 of
  ``notebooks/01-target-specificity-on-ground-truth.ipynb``).
* ``kmeans_other`` — every other region the multiscale pipeline passed as target-specific.
* ``candidate``    — every other eligible, full-replicate-support HL-60 candidate region
  inside the same windows: negatives that share their windows' sequence context.

Each region carries its window (``window_chrom``/``window_start``/``window_end``) and the
window's ``fold`` from the BED's 4th column.

Two observed scores per region:

* ``observed_score`` — the validated statistic (``target_specificity_score``: per-track
  rolling-max@1000 bp inside the region -> anchor units, background subtracted -> replicate
  mean per group -> log2 -> target minus q0.9 of the other groups).
* ``observed_label_score`` — the reduction the training metric applies to predicted channels
  (:func:`regulonado.training.specificity_panel.region_scores`), applied here to the
  *observed* group-contrast labels. Its agreement with ``observed_score`` is the ceiling the
  training metric can reach, recorded in the sidecar JSON.

``observed_positive`` is ``observed_score >= --positive-threshold`` over every tier; the
threshold is fixed here so training runs stay comparable.

Usage::

    uv run python scripts/build_specificity_panel.py \\
        --spec-run <uef>/data/interim/03-multiscale-specificity/runs/<run> \\
        --selected <uef>/data/interim/04-examined-outputs/selected-kmeans-peak-groups.parquet \\
        --bigwig-dir <uef>/data/raw/atlas/bigwigs/bamnado/unscaled \\
        --intervals sequences_human.bed.gz --tracks tracks.parquet \\
        --out notebooks/results/target-specificity/specificity-panel.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from regulonado.target_specificity import (  # noqa: E402
    group_index_from_records,
    region_statistic,
    target_specificity_score,
)
from regulonado.training.group_contrast import (  # noqa: E402
    group_contrast_labels,
    group_replicate_weights,
)
from regulonado.training.runner import load_dataset_metadata, track_records  # noqa: E402
from regulonado.training.specificity_panel import region_scores  # noqa: E402

BIN_SIZE = 32
SMOOTHING_BINS = 31
PAD_BINS = SMOOTHING_BINS // 2


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def assign_windows(regions: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """Index into ``windows`` of the window each region sits most centrally in, or -1.

    A region qualifies for a window only when it lies entirely inside it; among those, the
    window maximising the region's distance to the nearer window edge wins.
    """
    out = np.full(len(regions), -1, dtype=np.int64)
    by_chrom = {c: w for c, w in windows.groupby("chrom")}
    for i, (chrom, start, end) in enumerate(
        zip(regions.chrom, regions.region_start, regions.region_end)
    ):
        w = by_chrom.get(chrom)
        if w is None:
            continue
        inside = w[(w.start <= start) & (w.end >= end)]
        if len(inside):
            margin = np.minimum(start - inside.start.to_numpy(), inside.end.to_numpy() - end)
            out[i] = inside.index[int(np.argmax(margin))]
    return out


def extract_profiles(regions: pd.DataFrame, bigwigs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean coverage ``(R, T, B)`` over each region padded by ``PAD_BINS`` each side."""
    import pybigtools

    lengths = (
        np.ceil((regions.region_end - regions.region_start) / BIN_SIZE).astype(np.int64).to_numpy()
    )
    total = lengths + 2 * PAD_BINS
    profiles = np.zeros((len(regions), len(bigwigs), int(total.max())), dtype=np.float32)
    spans = [
        (c, int(s) - PAD_BINS * BIN_SIZE, int(s) + int(n) * BIN_SIZE, int(n))
        for c, s, n in zip(regions.chrom, regions.region_start, total)
    ]

    def one_track(t: int) -> None:
        reader = pybigtools.open(str(bigwigs[t]))
        for r, (chrom, start, end, n_bins) in enumerate(spans):
            vals = reader.values(
                chrom, start, end, bins=n_bins, summary="mean", exact=True, missing=0
            )
            profiles[r, t, :n_bins] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one_track, range(len(bigwigs))))
    return profiles, lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec-run", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--bigwig-dir", type=Path, required=True)
    parser.add_argument(
        "--intervals", type=Path, required=True, help="dataset window BED (chrom, start, end, fold)"
    )
    parser.add_argument("--tracks", type=Path, default=Path("tracks.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-group", default="HL-60")
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--pseudocount", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--floor", type=float, default=0.139)
    args = parser.parse_args()

    best = pd.read_parquet(args.spec_run / "best-overall-target-windows.parquet")
    kmeans_ids = set(pd.read_parquet(args.spec_run / "kmeans-peak-groups.parquet").region_id)
    curated_ids = set(pd.read_parquet(args.selected).region_id)
    windows = pd.read_csv(
        args.intervals,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "fold"],
    )

    enhancers = best[best.region_id.isin(kmeans_ids)].reset_index(drop=True)
    enhancer_window = assign_windows(enhancers, windows)
    unplaced = int((enhancer_window < 0).sum())
    chosen = windows.loc[np.unique(enhancer_window[enhancer_window >= 0])]
    others = best[
        ~best.region_id.isin(kmeans_ids) & best.eligible_target_window & best.target_support.eq(3)
    ]
    others = others.reset_index(drop=True)
    other_window = assign_windows(others, chosen)

    enhancers["window_index"] = enhancer_window
    others["window_index"] = other_window
    panel = pd.concat(
        [enhancers[enhancer_window >= 0], others[other_window >= 0]], ignore_index=True
    )
    panel["tier"] = np.where(
        panel.region_id.isin(curated_ids),
        "curated",
        np.where(panel.region_id.isin(kmeans_ids), "kmeans_other", "candidate"),
    )
    placed = windows.loc[panel.window_index].reset_index(drop=True)
    panel["window_chrom"], panel["window_start"] = placed.chrom, placed.start
    panel["window_end"], panel["fold"] = placed.end, placed.fold
    cols = [
        "region_id",
        "chrom",
        "region_start",
        "region_end",
        "region_bp",
        "target_support",
        "tier",
        "window_chrom",
        "window_start",
        "window_end",
        "fold",
    ]
    panel = panel[cols].sort_values(["chrom", "region_start"]).reset_index(drop=True)
    print(
        f"panel: {panel.tier.value_counts().to_dict()} in {len(chosen)} windows; "
        f"{unplaced} enhancer regions not inside any dataset window"
    )

    records = track_records(load_dataset_metadata(None, args.tracks))
    bigwigs = [args.bigwig_dir / Path(str(r["path"])).name for r in records]
    missing = [p for p in bigwigs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} bigwigs missing under {args.bigwig_dir}, e.g. {missing[0]}"
        )
    group_index, group_names = group_index_from_records(records)
    target_index = group_names.index(args.target_group)
    scale_factor = np.array([float(r["scale_factor"]) for r in records], dtype=np.float32)
    background = np.array([float(r.get("background") or 0.0) for r in records], dtype=np.float32)

    profiles, lengths = extract_profiles(panel, bigwigs)

    # Validated statistic: region bins only (strip the padding).
    inner = np.zeros((len(panel), len(records), int(lengths.max())), dtype=np.float32)
    for r, n in enumerate(lengths):
        inner[r, :, :n] = profiles[r, :, PAD_BINS : PAD_BINS + n]
    stat = region_statistic(
        inner, lengths, bin_size=BIN_SIZE, window_stat_bp=1000, stat="rolling-max"
    )
    observed = target_specificity_score(
        stat,
        scale_factor=scale_factor,
        background=background,
        group_index=group_index,
        group_names=group_names,
        target_group=args.target_group,
        background_mode="subtract",
        pseudocount=args.pseudocount,
        reduction="quantile",
        quantile=args.quantile,
    )
    panel["observed_score"] = observed.score
    panel["observed_target_rank"] = observed.target_rank

    # Training-side reduction on observed labels: anchor units = signal * sf, background * sf.
    labels, _ = group_contrast_labels(
        torch.from_numpy(profiles * scale_factor[None, :, None]),
        group_weights=group_replicate_weights(torch.as_tensor(group_index), len(group_names)),
        background=torch.from_numpy(background * scale_factor),
        smoothing_bins=SMOOTHING_BINS,
        pseudocount=args.pseudocount,
        floor=args.floor,
    )
    region_mask = np.zeros(profiles.shape[::2], dtype=bool)  # (R, B)
    for r, n in enumerate(lengths):
        region_mask[r, PAD_BINS : PAD_BINS + n] = True
    label_score, label_rank = region_scores(
        labels, torch.from_numpy(region_mask), target_index=target_index, quantile=args.quantile
    )
    panel["observed_label_score"] = label_score.numpy()
    panel["observed_label_target_rank"] = label_rank.numpy()
    panel["observed_positive"] = panel.observed_score >= args.positive_threshold

    cur, ctl = (panel.tier == "curated").to_numpy(), (panel.tier == "candidate").to_numpy()
    pos = panel.observed_positive.to_numpy()
    summary = {
        "target_group": args.target_group,
        "positive_threshold": args.positive_threshold,
        "pseudocount": args.pseudocount,
        "quantile": args.quantile,
        "n_regions": len(panel),
        "n_windows": int(panel[["window_chrom", "window_start"]].drop_duplicates().shape[0]),
        "enhancers_not_in_any_window": unplaced,
        "tiers": panel.tier.value_counts().to_dict(),
        "observed_positive_by_tier": panel.groupby("tier")
        .observed_positive.sum()
        .astype(int)
        .to_dict(),
        "observed_score": {
            "curated_auc": auc_rank(panel.observed_score[cur], panel.observed_score[ctl]),
            "curated_rank1_fraction": float((panel.observed_target_rank[cur] == 1).mean()),
        },
        "ceiling_label_reduction": {
            "curated_auc": auc_rank(
                panel.observed_label_score[cur], panel.observed_label_score[ctl]
            ),
            "positive_auc": auc_rank(
                panel.observed_label_score[pos], panel.observed_label_score[~pos]
            ),
            "spearman_vs_observed_score": float(
                spearmanr(panel.observed_label_score, panel.observed_score)[0]
            ),
            "curated_rank1_fraction": float((panel.observed_label_target_rank[cur] == 1).mean()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
