"""Specificity objective: score a batch of one-hot sequences against a target cell-type group.

Replaces the skeleton's ``TargetFeatureEnergy``, keeping its ``clamp``/``bend`` transforms and
fixing the rest: explicit bool masks instead of an inverted Python int, an implemented
``fold_reduction``, replicate-track aggregation per group, and a softer off-target reduction than
a flat ``max``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["EnergyResult", "SpecificityEnergy", "TrackGroups", "resolve_track_groups"]

# group_by column -> the RegulonadoConfig.track_metadata key holding its id vector.
# Mirrors tracks.CATEGORICAL_FIELDS / training.runner._TRACK_METADATA_FIELD_MAP; the
# `track_cell_line_ids` name for `source` is inherited from the trained checkpoint's config.
GROUP_BY_METADATA_KEY = {
    "condition": "track_condition_ids",
    "source": "track_cell_line_ids",
    "assay": "track_assay_type_ids",
    "ip": "track_target_ids",
}


@dataclass(slots=True)
class TrackGroups:
    labels: list[str | None]  # per-track group label, len == n_tracks
    target: str
    target_idx: torch.Tensor  # bool mask (n_tracks,)
    other_group_masks: dict[str, torch.Tensor]


def resolve_track_groups(
    track_names: list[str],
    *,
    group_by: str = "source",
    target: str,
    track_sheet: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    config: object | None = None,
) -> TrackGroups:
    """Resolve per-track group labels and build target/off-target masks.

    Resolution order: an explicit ``track_sheet`` CSV, then ``final_track_records`` under
    ``dataset_dir``, then the id vectors already stored on a model ``config``.
    """
    if track_sheet is not None:
        labels = _labels_from_track_sheet(track_names, track_sheet, group_by)
    elif dataset_dir is not None:
        labels = _labels_from_dataset_metadata(track_names, dataset_dir, group_by)
    elif config is not None:
        labels = _labels_from_config_metadata(track_names, config, group_by)
    else:
        raise ValueError("resolve_track_groups needs one of track_sheet, dataset_dir, or config")

    distinct = sorted({label for label in labels if label is not None})
    if target not in distinct:
        raise ValueError(
            f"Target group {target!r} matches no track (group_by={group_by!r}); "
            f"available groups: {', '.join(distinct) or 'none'}"
        )

    target_idx = torch.tensor([label == target for label in labels], dtype=torch.bool)
    other_group_masks = {
        group: torch.tensor([label == group for label in labels], dtype=torch.bool)
        for group in distinct
        if group != target
    }
    return TrackGroups(
        labels=labels, target=target, target_idx=target_idx, other_group_masks=other_group_masks
    )


def _labels_from_track_sheet(
    track_names: list[str], track_sheet: str | Path, group_by: str
) -> list[str | None]:
    from regulonado.tracks import TrackRecord, TrackSheet

    if group_by not in TrackRecord.model_fields:
        raise ValueError(f"group_by={group_by!r} is not a known track sheet column")
    sheet = TrackSheet.from_csv(track_sheet)
    lookup = {
        (record.track_name or record.sample_id): getattr(record, group_by)
        for record in sheet.records
    }
    return [lookup.get(name) for name in track_names]


def _labels_from_dataset_metadata(
    track_names: list[str], dataset_dir: str | Path, group_by: str
) -> list[str | None]:
    metadata_path = Path(dataset_dir) / "regulonado_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    records = metadata.get("final_track_records") or []
    if not records:
        raise ValueError(f"No 'final_track_records' in dataset metadata under {dataset_dir}")
    by_name = {record.get("track_name"): record for record in records}
    labels: list[str | None] = []
    for index, name in enumerate(track_names):
        record = by_name.get(name) or (records[index] if index < len(records) else None)
        labels.append(record.get(group_by) if record else None)
    return labels


def _labels_from_config_metadata(
    track_names: list[str], config: object, group_by: str
) -> list[str | None]:
    metadata_key = GROUP_BY_METADATA_KEY.get(group_by)
    if metadata_key is None:
        raise ValueError(
            f"group_by={group_by!r} is not a known column; expected one of "
            f"{', '.join(GROUP_BY_METADATA_KEY)}"
        )
    track_metadata = getattr(config, "track_metadata", {}) or {}
    ids = track_metadata.get(metadata_key)
    if not ids:
        raise ValueError(
            f"No {metadata_key!r} in track metadata; pass --track-sheet or --dataset-dir"
        )
    if len(ids) != len(track_names):
        raise ValueError(
            f"{metadata_key!r} has {len(ids)} entries but there are {len(track_names)} tracks"
        )
    return [str(value) if value is not None and value >= 0 else None for value in ids]


@dataclass(slots=True)
class EnergyResult:
    energy: torch.Tensor  # (B,)
    target: torch.Tensor  # (B,)
    per_group: torch.Tensor  # (B, n_groups), group order == group_names
    per_fold_energy: torch.Tensor  # (n_folds, B)
    group_names: list[str]  # target first, then other_group_masks in dict order
    per_track: torch.Tensor  # (B, n_tracks), averaged over folds
    track_mean: torch.Tensor  # (B, n_tracks), mean across selected bins
    track_max: torch.Tensor  # (B, n_tracks), maximum selected-bin signal
    track_topk: torch.Tensor  # (B, n_tracks), mean of ordered top-K bins
    track_topk_ratio: torch.Tensor  # (B, n_tracks), top-K mean / mean


class SpecificityEnergy(nn.Module):
    """Lower is better: off-target signal minus (scaled) on-target signal."""

    def __init__(
        self,
        ensemble,
        groups: TrackGroups,
        bins: slice,
        target_alpha: float = 1.0,
        bending_factor: float = 0.0,
        a_min: float = -math.inf,
        a_max: float = math.inf,
        offtarget_reduction: Literal["max", "logsumexp", "mean"] = "logsumexp",
        offtarget_temperature: float = 1.0,
        fold_reduction: Literal["mean", "mean_plus_std"] = "mean",
        bin_reduction: Literal["mean", "topk"] = "mean",
        topk_bins: int = 10,
    ) -> None:
        super().__init__()
        self.ensemble = ensemble
        self.groups = groups
        self.bins = bins
        self.target_alpha = float(target_alpha)
        self.bending_factor = float(bending_factor)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.offtarget_reduction = offtarget_reduction
        self.offtarget_temperature = float(offtarget_temperature)
        self.fold_reduction = fold_reduction
        self.bin_reduction = bin_reduction
        self.topk_bins = int(topk_bins)
        if self.topk_bins < 1:
            raise ValueError("topk_bins must be >= 1")

    def bend(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.bending_factor:
            return tensor
        exp_neg = torch.exp(-tensor)
        adjustment = self.bending_factor * (exp_neg - 1.0)
        return tensor - adjustment

    def forward(self, one_hot_batch) -> EnergyResult:
        # torch.inference_mode() lives in the predictor, not here, so a future gradient
        # path through the energy is not blocked by the objective itself.
        preds = self.ensemble.predict(one_hot_batch)  # (n_folds, B, n_tracks, n_bins)
        preds = preds.clamp(self.a_min, self.a_max)
        preds = self.bend(preds)
        windowed = preds[..., self.bins]  # (n_folds, B, n_tracks, n_bins_in_window)
        if windowed.shape[-1] == 0:
            raise ValueError("bins selects no prediction bins")
        if self.bin_reduction == "topk" and self.topk_bins > windowed.shape[-1]:
            raise ValueError(
                f"topk_bins={self.topk_bins} exceeds selected bin count {windowed.shape[-1]}"
            )
        track_mean = windowed.mean(dim=-1)
        track_max = windowed.max(dim=-1).values
        diagnostic_k = min(self.topk_bins, windowed.shape[-1])
        track_topk = windowed.topk(diagnostic_k, dim=-1).values.mean(dim=-1)

        # Explicit bool masks per group: the skeleton did `~self.target[0]` on a Python int
        # (`~5 == -6`), which silently selected the wrong track and reduced over the batch.
        group_masks = {self.groups.target: self.groups.target_idx, **self.groups.other_group_masks}
        if self.bin_reduction == "topk":
            per_track = track_topk
        else:
            per_track = track_mean
        group_scores = {}
        for group, mask in group_masks.items():
            if not bool(mask.any()):
                raise ValueError(f"Track group {group!r} selects no tracks")
            group_scores[group] = per_track[:, :, mask].mean(dim=-1)

        target_score = group_scores[self.groups.target]  # (n_folds, B)
        other_names = list(self.groups.other_group_masks)
        other_scores = torch.stack([group_scores[name] for name in other_names], dim=-1)

        if self.offtarget_reduction == "max":
            offtarget = other_scores.max(dim=-1).values
        elif self.offtarget_reduction == "mean":
            offtarget = other_scores.mean(dim=-1)
        elif self.offtarget_reduction == "logsumexp":
            temperature = self.offtarget_temperature
            offtarget = temperature * torch.logsumexp(other_scores / temperature, dim=-1)
        else:
            raise ValueError(f"Unknown offtarget_reduction={self.offtarget_reduction!r}")

        per_fold_energy = offtarget - self.target_alpha * target_score  # (n_folds, B)

        if self.fold_reduction == "mean":
            energy = per_fold_energy.mean(dim=0)
        elif self.fold_reduction == "mean_plus_std":
            energy = per_fold_energy.mean(dim=0) + per_fold_energy.std(dim=0)
        else:
            raise ValueError(f"Unknown fold_reduction={self.fold_reduction!r}")

        group_names = [self.groups.target, *other_names]
        per_group = torch.stack([group_scores[name] for name in group_names], dim=-1).mean(dim=0)

        return EnergyResult(
            energy=energy,
            target=target_score.mean(dim=0),
            per_group=per_group,
            per_fold_energy=per_fold_energy,
            group_names=group_names,
            per_track=per_track.mean(dim=0),
            track_mean=track_mean.mean(dim=0),
            track_max=track_max.mean(dim=0),
            track_topk=track_topk.mean(dim=0),
            track_topk_ratio=(track_topk / track_mean.clamp_min(1e-8)).mean(dim=0),
        )
