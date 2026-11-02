"""Prepare chromosome-disjoint numeric diffusion labels from the count dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


def canonical(sequence):
    return min(sequence, sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1])


def chromosome_splits(regions, validation_chromosomes, test_chromosomes):
    """Preserve recorded splits; never silently relabel a chromosome."""
    frame = regions.copy()
    if "split" in frame:
        split = frame.split.astype(str).str.lower().map(ALIASES)
        if split.isna().any():
            raise ValueError("Region split labels must be train, val/validation, or test")
        frame["split"] = split
        if frame.groupby("chrom").split.nunique().gt(1).any():
            raise ValueError("Chromosome leakage: a chromosome occurs in multiple splits")
    else:
        if set(validation_chromosomes) & set(test_chromosomes):
            raise ValueError("Validation/test chromosomes overlap")
        frame["split"] = np.where(
            frame.chrom.isin(test_chromosomes),
            "test",
            np.where(frame.chrom.isin(validation_chromosomes), "validation", "train"),
        )
    return frame


def read_priorities(path):
    from regulonado.genomics import read_intervals

    frame = read_intervals(path)
    raw = pd.read_csv(path, sep="\t", header=None, comment="#")
    if raw.shape[1] < 8 or len(raw) != len(frame):
        raise ValueError("Specificity BED needs score in column 8; one per assay")
    frame["target_specificity"] = pd.to_numeric(raw[7], errors="raise")
    if not np.isfinite(frame.target_specificity).all():
        raise ValueError("Specificity scores must be finite")
    return frame.groupby(["chrom", "start", "end"], as_index=False).agg(
        target_specificity=("target_specificity", "mean"), assay_rows=("target_specificity", "size")
    )


def choose_regions(frame, limit, seed):
    priority = frame.loc[frame.priority]
    background = frame.loc[~frame.priority]
    if limit is None or limit >= len(frame):
        return frame.copy()
    if limit < len(priority):
        raise ValueError(f"Region limit {limit} is smaller than {len(priority)} curated loci")
    return pd.concat(
        [priority, background.sample(n=limit - len(priority), random_state=seed)]
    ).copy()


def prepare_diffusion_data(dataset_dir, fasta_path, out, config):
    import polars as pl
    import pyfaidx

    from regulonado.counts.dataset import group_count_rates

    dataset_dir, out = Path(dataset_dir), Path(out)
    regions = pd.read_parquet(dataset_dir / "regions.parquet")
    if not {"chrom", "start", "end"} <= set(regions):
        raise ValueError("Region table needs chrom/start/end")
    regions["dataset_row"] = np.arange(len(regions))
    regions = chromosome_splits(regions, config.validation_chromosomes, config.test_chromosomes)
    if regions[["chrom", "start", "end"]].isna().any().any():
        raise ValueError("Region coordinates must be present")
    region_count = len(regions)
    regions["priority"] = False
    regions["target_specificity"] = np.nan
    if config.specificity_regions:
        priority = read_priorities(config.specificity_regions)
        coordinates = (
            ["orig_chrom", "orig_start", "orig_end"]
            if {"orig_chrom", "orig_start", "orig_end"} <= set(regions)
            else ["chrom", "start", "end"]
        )
        mapping = dict(
            zip(zip(priority.chrom, priority.start, priority.end), priority.target_specificity)
        )
        keys = list(regions[coordinates].itertuples(index=False, name=None))
        regions["priority"] = [key in mapping for key in keys]
        regions["target_specificity"] = [mapping.get(key, np.nan) for key in keys]
        if not regions.priority.any():
            raise ValueError("No specificity regions match original dataset coordinates")
    if "ccre_class" in regions and config.ccre_classes:
        regions = regions.loc[
            regions.ccre_class.isin(config.ccre_classes) | regions.priority
        ].copy()
    start = "target_start" if "target_start" in regions else "start"
    end = "target_end" if "target_end" in regions else "end"
    if (regions[start] < 0).any() or (regions[end] <= regions[start]).any():
        raise ValueError("Invalid target interval coordinates")
    if (regions[end] - regions[start] < config.core_length).any():
        raise ValueError("core_length exceeds a supplied target interval")
    parts = [
        choose_regions(regions.loc[regions.split.eq("train")], config.train_regions, config.seed),
        choose_regions(
            regions.loc[regions.split.eq("validation")], config.validation_regions, config.seed + 1
        ),
        regions.loc[regions.split.eq("test")].copy(),
    ]
    if parts[0].empty or parts[1].empty:
        raise ValueError("Nonempty train and validation chromosome splits are required")
    selected = pd.concat(parts, ignore_index=True)
    records = []
    rng = np.random.default_rng(config.seed)
    with pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True) as fasta:
        for row in selected.to_dict("records"):
            lower, upper = int(row[start]), int(row[end])
            center = (lower + upper) // 2
            n_views = config.priority_views if row["priority"] and row["split"] == "train" else 1
            seen = set()
            for view in range(n_views):
                shift = (
                    int(rng.integers(-config.priority_jitter_bp, config.priority_jitter_bp + 1))
                    if view
                    else 0
                )
                core_start = max(
                    lower, min(center - config.core_length // 2 + shift, upper - config.core_length)
                )
                chrom = str(row["chrom"])
                if chrom not in fasta or core_start + config.core_length > len(fasta[chrom]):
                    raise ValueError(f"Region is outside the reference: {chrom}:{core_start}")
                sequence = str(fasta[chrom][core_start : core_start + config.core_length]).upper()
                if set(sequence) - set("ACGT"):
                    continue
                reverse = bool(view % 2 and config.priority_reverse_complement)
                if reverse:
                    sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                key = canonical(sequence)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    dict(
                        dataset_row=int(row["dataset_row"]),
                        chrom=chrom,
                        split=row["split"],
                        sequence=sequence,
                        canonical=key,
                        priority=bool(row["priority"]),
                        view=view,
                        core_start=core_start,
                        core_end=core_start + config.core_length,
                        reverse_complement=reverse,
                        target_specificity=row["target_specificity"],
                        sampling_weight=config.priority_sampling_weight if row["priority"] else 1.0,
                    )
                )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values("priority", ascending=False, kind="stable")
    if frame.empty:
        raise ValueError("No unambiguous DNA cores")
    # All eligible test cores is considered for exact/RC leakage checks, but never trained.
    test_keys = set(frame.loc[frame.split.eq("test"), "canonical"])
    validation = frame.loc[frame.split.eq("validation") & ~frame.canonical.isin(test_keys)].copy()
    validation = validation.drop_duplicates("canonical")
    forbidden = test_keys | set(validation.canonical)
    train = frame.loc[frame.split.eq("train") & ~frame.canonical.isin(forbidden)].copy()
    train = train.drop_duplicates("canonical")
    clean = pd.concat([train, validation], ignore_index=True)
    if train.empty or validation.empty:
        raise ValueError("Leakage/sequence filtering emptied a training or validation split")
    tracks = pd.read_parquet(dataset_dir / "tracks.parquet")
    needed = {"track_name", config.group_by, "log_size_factor"}
    if not needed <= set(tracks) or tracks[list(needed)].isna().any().any():
        raise ValueError(f"Tracks must supply nonmissing {sorted(needed)}")
    groups = sorted(tracks[config.group_by].astype(str).unique())
    if config.target not in groups or len(groups) < 2:
        raise ValueError("Need the target and at least one off-target group")
    if not np.isfinite(tracks.log_size_factor).all():
        raise ValueError("Track log-size factors must be finite")
    counts_path = dataset_dir / "counts.parquet"
    scan = pl.scan_parquet(counts_path)
    if scan.select(pl.len()).collect().item() != region_count:
        raise ValueError("Counts and regions must have identical row counts")
    rows = sorted(clean.dataset_row.unique().tolist())
    counts = (
        scan.with_row_index("dataset_row")
        .filter(pl.col("dataset_row").is_in(rows))
        .select(["dataset_row", *tracks.track_name.tolist()])
        .collect(engine="streaming")
        .to_pandas()
        .set_index("dataset_row")
    )
    group_index = np.array([groups.index(str(group)) for group in tracks[config.group_by]])
    rates = group_count_rates(
        counts.loc[clean.dataset_row, tracks.track_name].to_numpy(),
        group_index,
        tracks.log_size_factor.to_numpy(),
        len(groups),
    )
    activity = np.log1p(rates)
    clean[groups] = activity
    others = [group for group in groups if group != config.target]
    clean["target_activity"] = clean[config.target]
    clean["offtarget_mean"] = clean[others].mean(axis=1)
    clean["offtarget_max"] = clean[others].max(axis=1)
    clean["target_contrast"] = clean.target_activity - clean.offtarget_mean
    clean = clean.loc[clean.target_activity.notna() & clean.offtarget_mean.notna()].copy()
    if not {"train", "validation"} <= set(clean.split):
        raise ValueError("Missing labels emptied a training or validation split")
    for column in config.training.activity_columns:
        if column not in clean:
            raise ValueError(f"Unknown diffusion activity column {column!r}")
        if not np.isfinite(clean.loc[clean.split.eq("train"), column]).any():
            raise ValueError(f"No observed training values for {column}")
    out.mkdir(parents=True, exist_ok=True)
    clean.drop(columns="canonical").to_parquet(out / "training.parquet", index=False)
    chroms = {
        split: sorted(clean.loc[clean.split.eq(split), "chrom"].unique().tolist())
        for split in ["train", "validation"]
    }
    test_chroms = sorted(regions.loc[regions.split.eq("test"), "chrom"].unique().tolist())
    manifest = dict(
        dataset_dir=str(dataset_dir.resolve()),
        target=config.target,
        group_names=groups,
        split_counts=clean.split.value_counts().to_dict(),
        chromosomes={**chroms, "test": test_chroms},
        priority_views=clean.groupby("split").priority.sum().astype(int).to_dict(),
        heldout_test_sequences=len(test_keys),
        core_length=config.core_length,
        seed=config.seed,
        augmentation=(
            "Training-only jitter bounded by target span; labels remain source-region measurements"
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
