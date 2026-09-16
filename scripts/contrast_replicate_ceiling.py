"""Replicate ceiling for cross-track contrast (``contrast_pearson``).

For every track whose group (cell type) has another replicate in the same contrast family,
this scores that other replicate as if it were the prediction, using the same targets,
transforms and ``specificity_stats`` the training metric uses. Replicates are cycled within
their group, so group means, the group-balanced family mean and the strongest-group region
selection are exactly those of the real targets.

Two numbers per track:

- ``replicate_r``: contrast Pearson between two replicates, each carrying measurement noise.
- ``ceiling_r = sqrt(replicate_r)``: the expected contrast Pearson of a noise-free prediction
  against one noisy replicate, assuming replicates share the signal and have independent,
  equal noise. This is the number to compare ``contrast_pearson`` against.

Replicates from the same study share batch effects, so the ceiling is for reproducing these
tracks, not for biology free of batch. Groups with a single track have no ceiling and are
reported as missing.

Example::

    python scripts/contrast_replicate_ceiling.py \\
        --data-path /path/to/regulonado/dataset \\
        --split test \\
        --model-per-track /path/to/train/fold_0/unfreeze_all_stages/per_track_metrics/test.csv \\
        --output contrast_ceiling.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from regulonado.training.data import WindowParquetDataset
from regulonado.training.losses import contrast_group_weights, specificity_stats
from regulonado.training.runner import (
    _apply_dataset_transforms,
    load_dataset_metadata,
    track_records,
)
from torch.utils.data import DataLoader, Subset


def replicate_partners(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Index of each track's replicate partner, or its own index when it has none.

    Tracks are grouped by ``(assay_class, group)``; within a group of n replicates, track
    i's partner is track i+1 (cyclically), so the permutation preserves every group mean.
    """
    members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        family, group = record.get("assay_class"), record.get("group")
        if family is not None and group is not None:
            members[(str(family), str(group))].append(index)
    partners = np.arange(len(records))
    for indices in members.values():
        if len(indices) >= 2:
            partners[indices] = np.roll(indices, -1)
    return partners


def _pearson_and_sd_ratio(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sum_p, sum_t, sum_pt, sum_p2, sum_t2, n = stats.T
    var_p = n * sum_p2 - sum_p**2
    var_t = n * sum_t2 - sum_t**2
    cov = n * sum_pt - sum_p * sum_t
    valid = (var_p > 1e-12) & (var_t > 1e-12)
    r = np.full(len(n), np.nan)
    sd_ratio = np.full(len(n), np.nan)
    r[valid] = cov[valid] / np.sqrt(var_p[valid] * var_t[valid])
    sd_ratio[valid] = np.sqrt(var_p[valid] / var_t[valid])
    return r, sd_ratio


def _stack_labels(items: list[dict[str, Any]]) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.asarray(item["labels"]) for item in items]))


class _SignalOnly:
    """Run the training transform on a signal-only row: sequence is never read.

    A class rather than a closure so DataLoader workers can pickle it under spawn.
    """

    def __init__(self, transform: Any) -> None:
        self._transform = transform

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        return self._transform({**example, "sequence_tokens": np.zeros(0, dtype=np.uint8)})


def compute_ceiling(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_path = Path(args.data_path)
    metadata = load_dataset_metadata(
        data_path, Path(args.metadata_path) if args.metadata_path else None
    )
    records = track_records(metadata)
    data_cfg = {
        "apply_scale": args.apply_scale,
        "apply_squash": args.apply_squash,
        "apply_clip": args.apply_clip,
        "enable_rc_aug": False,
    }
    # Registering the split as "test" gives it the centre-cropped, non-augmented eval
    # transform; "train" is required by the helper and left unused.
    dataset = WindowParquetDataset(data_path, args.split, columns=("signal",))
    transformed = _apply_dataset_transforms(
        {
            "train": WindowParquetDataset(data_path, args.split, columns=("signal",)),
            "test": dataset,
        },
        metadata,
        records,
        data_cfg,
    )["test"]
    transformed.transform = _SignalOnly(transformed.transform)
    rows_available = len(transformed)
    subset: Any = transformed
    if args.max_examples is not None and args.max_examples < rows_available:
        chosen = np.random.default_rng(args.seed).choice(
            rows_available, args.max_examples, replace=False
        )
        subset = Subset(transformed, sorted(chosen.tolist()))

    weights = contrast_group_weights(
        [record.get("assay_class") for record in records],
        [record.get("group") for record in records],
    )
    partners = replicate_partners(records)
    partner_index = torch.from_numpy(partners)
    n_tracks = len(records)
    stats = np.zeros((n_tracks, 6), dtype=np.float64)
    # make_transform returns a closure, which only reaches workers by fork (spawn pickles).
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_stack_labels,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    n_examples = 0
    for labels in loader:
        target = labels.float()
        if target.shape[1] != n_tracks:  # stored as [B, L, T]
            target = target.transpose(1, 2)
        batch_stats = specificity_stats(
            target[:, partner_index],
            target,
            weights,
            region_bins=args.region_bins,
            pseudocount=args.pseudocount,
            active_fraction=args.active_fraction,
        )
        stats += batch_stats.sum(dim=0).double().numpy()
        n_examples += target.shape[0]
        print(f"\r{n_examples}/{len(subset)} examples", end="", flush=True)
    print()

    replicate_r, sd_ratio = _pearson_and_sd_ratio(stats)
    has_partner = partners != np.arange(n_tracks)
    model = _read_model_per_track(args.model_per_track) if args.model_per_track else {}
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = str(record.get("track_name") or f"track_{index}")
        r = float(replicate_r[index]) if has_partner[index] else math.nan
        ceiling = math.sqrt(r) if r > 0 else math.nan
        model_r = model.get(name, math.nan)
        rows.append(
            {
                "track_name": name,
                "group": record.get("group") or "",
                "assay_class": record.get("assay_class") or "",
                "partner_track": str(records[partners[index]].get("track_name"))
                if has_partner[index]
                else "",
                "replicate_r": r,
                "replicate_sd_ratio": float(sd_ratio[index]) if has_partner[index] else math.nan,
                "ceiling_r": ceiling,
                "model_contrast_pearson": model_r,
                "fraction_of_ceiling": model_r / ceiling if ceiling > 0 else math.nan,
            }
        )
    return rows


def _read_model_per_track(path: str) -> dict[str, float]:
    with open(path, newline="") as handle:
        return {row["track_name"]: float(row["contrast_pearson"]) for row in csv.DictReader(handle)}


def _median(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


def print_summary(rows: Sequence[Mapping[str, Any]], focus_groups: Sequence[str]) -> None:
    columns = ("replicate_r", "ceiling_r", "model_contrast_pearson", "fraction_of_ceiling")

    def line(label: str, subset: Sequence[Mapping[str, Any]]) -> str:
        n_ceiling = sum(math.isfinite(row["ceiling_r"]) for row in subset)
        values = "  ".join(f"{_median([row[c] for row in subset]):>8.3f}" for c in columns)
        return f"{label:<45} {len(subset):>4} {n_ceiling:>4}  {values}"

    header = "  ".join(f"{c.replace('model_contrast_pearson', 'model_r')[:8]:>8}" for c in columns)
    print(f"{'median over tracks':<45} {'n':>4} {'ceil':>4}  {header}")
    print(line("all", rows))
    for assay_class in sorted({row["assay_class"] for row in rows} - {""}):
        print(
            line(f"assay_class={assay_class}", [r for r in rows if r["assay_class"] == assay_class])
        )
    for group in focus_groups:
        subset = [row for row in rows if row["group"] == group]
        if subset:
            print(line(f"group={group}", subset))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-path", required=True, help="Built dataset directory (README.md + data/)."
    )
    parser.add_argument(
        "--metadata-path", help="tracks.parquet to use instead of <data-path>/tracks.parquet."
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output", default="contrast_replicate_ceiling.csv")
    parser.add_argument(
        "--model-per-track", help="per_track_metrics/<split>.csv from a training run."
    )
    parser.add_argument("--max-examples", type=int, help="Random subset of windows (default: all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    # Must match the run being compared: data.* and trainer.contrast_*.
    parser.add_argument("--apply-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-squash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--region-bins", type=int, default=16)
    parser.add_argument("--pseudocount", type=float, default=0.1)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument(
        "--focus-groups",
        nargs="*",
        default=[
            "HL-60",
            "MV-4-11",
            "THP-1",
            "KG-1",
            "Bone_marrow_primary_AML",
            "CD14_positive_monocyte",
        ],
    )
    args = parser.parse_args()

    rows = compute_ceiling(args)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, args.focus_groups)
    print(f"per-track results: {args.output}")


if __name__ == "__main__":
    main()
