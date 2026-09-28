"""Specificity objective: score a batch of one-hot sequences against a target cell-type group.

Replaces the skeleton's ``TargetFeatureEnergy``, keeping its ``clamp``/``bend`` transforms and
fixing the rest: explicit bool masks instead of an inverted Python int, an implemented
``fold_reduction``, replicate-track aggregation per group, and a softer off-target reduction than
a flat ``max``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "EnergyResult",
    "SpecificityEnergy",
    "TrackGroups",
    "contrast_energy",
    "group_track_groups",
    "max_offtarget_energy",
    "resolve_track_group_indices",
    "resolve_track_groups",
    "worst_offtarget_energy",
]

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
    exclude_tracks: set[str] | None = None,
    track_sheet: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    config: object | None = None,
) -> TrackGroups:
    """Resolve per-track group labels and build target/off-target masks.

    Resolution order: an explicit ``track_sheet`` CSV, then ``tracks.parquet`` under
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

    excluded = set(exclude_tracks or ())
    unknown = excluded.difference(track_names)
    if unknown:
        raise ValueError(
            "Excluded track(s) not present in the model: " + ", ".join(sorted(unknown))
        )
    labels = [None if name in excluded else label for name, label in zip(track_names, labels)]
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


def resolve_track_group_indices(
    track_names: list[str],
    *,
    group_by: str,
    target: str,
    exclude_tracks: set[str] | None = None,
    track_sheet: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    config: object | None = None,
) -> list[int]:
    """Positional indices of every track in ``target``'s group — see :func:`resolve_track_groups`.

    The read-only counterpart callers reach for when they just need "which tracks", not the full
    off-target mask bookkeeping ``SpecificityEnergy`` needs (e.g. attribution's group readout).
    """
    groups = resolve_track_groups(
        track_names,
        group_by=group_by,
        target=target,
        exclude_tracks=exclude_tracks,
        track_sheet=track_sheet,
        dataset_dir=dataset_dir,
        config=config,
    )
    return groups.target_idx.nonzero().flatten().tolist()


def group_track_groups(
    group_names: list[str] | None, *, target: str, exclude_groups: tuple[str, ...] = ()
) -> TrackGroups:
    """``TrackGroups`` over a region model's group axis -- no track sheet needed.

    A region model's "track" axis (``RegionSequencePredictor.track_names``) IS its group axis
    (each group has exactly one scalar output), so there is no track-to-group resolution to do
    the way :func:`resolve_track_groups` needs for a per-track model: each mask is simply
    one-hot over ``group_names``. This is pure construction, not lookup -- no track sheet,
    dataset, or model config argument.
    """
    if not group_names:
        raise ValueError(
            "group_names is empty or None; a region ensemble's group_track_groups needs "
            "group_names (RegionSequencePredictor.group_names / RegionFoldEnsemble.group_names)"
        )
    excluded = set(exclude_groups)
    unknown_excluded = excluded.difference(group_names)
    if unknown_excluded:
        raise ValueError(
            "Excluded group(s) not present in group_names: " + ", ".join(sorted(unknown_excluded))
        )
    labels: list[str | None] = [None if name in excluded else name for name in group_names]
    distinct = sorted({label for label in labels if label is not None})
    if target not in distinct:
        raise ValueError(
            f"Target group {target!r} matches no group; available groups: "
            f"{', '.join(distinct) or 'none'}"
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
    from regulonado.tracks_table import read_track_table, to_track_records

    table = read_track_table(Path(dataset_dir) / "tracks.parquet")
    records = to_track_records(table)
    if not records:
        raise ValueError(f"No included tracks in {dataset_dir}/tracks.parquet")
    by_name = {record["track_name"]: record for record in records}
    return [by_name[name].get(group_by) if name in by_name else None for name in track_names]


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
    specificity: torch.Tensor  # (B,), absolute legacy specificity diagnostic
    target: torch.Tensor  # (B,)
    per_group: torch.Tensor  # (B, n_groups), group order == group_names
    per_fold_energy: torch.Tensor  # (n_folds, B)
    per_fold_specificity: torch.Tensor  # (n_folds, B)
    group_names: list[str]  # target first, then other_group_masks in dict order
    per_track: torch.Tensor  # (B, n_tracks), averaged over folds
    track_mean: torch.Tensor  # (B, n_tracks), mean across selected bins
    track_max: torch.Tensor  # (B, n_tracks), maximum selected-bin signal
    track_topk: torch.Tensor  # (B, n_tracks), mean of ordered top-K bins
    track_topk_ratio: torch.Tensor  # (B, n_tracks), top-K mean / mean
    target_gain: torch.Tensor  # (B,), relative to the seed for selective activation
    per_group_gain: torch.Tensor  # (B, n_groups), same order as group_names
    offtarget_boost: torch.Tensor  # (B,), aggregated positive off-target gain
    n_edits: torch.Tensor  # (B,), substitutions vs the reference sequence
    edit_penalty: torch.Tensor  # (B,), the drift term already added into `energy`
    objective: str
    has_reference: bool
    gain_transform: str
    gain_pseudocount: float


class SpecificityEnergy(nn.Module):
    """Score sequence designs for absolute specificity or seed-relative activation.

    The default ``specificity`` objective preserves the historical behaviour.  In
    ``selective-activation`` mode, call :meth:`set_reference` with the unedited seed before
    scoring designs.  Lower is better for both objectives.
    """

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
        objective: Literal["specificity", "selective-activation"] = "specificity",
        offtarget_boost_weight: float = 1.0,
        offtarget_boost_tolerance: float = 0.0,
        gain_transform: Literal["raw", "log2-fold-change"] = "raw",
        gain_pseudocount: float = 1.0,
        edit_penalty_weight: float = 0.0,
        edit_budget: int = 0,
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
        self.objective = objective
        self.offtarget_boost_weight = float(offtarget_boost_weight)
        self.offtarget_boost_tolerance = float(offtarget_boost_tolerance)
        self.gain_transform = gain_transform
        self.gain_pseudocount = float(gain_pseudocount)
        self.edit_penalty_weight = float(edit_penalty_weight)
        self.edit_budget = int(edit_budget)
        self._reference_group_scores: dict[str, torch.Tensor] | None = None
        self._reference_sequence: torch.Tensor | None = None
        if self.topk_bins < 1:
            raise ValueError("topk_bins must be >= 1")
        if self.objective not in ("specificity", "selective-activation"):
            raise ValueError(f"Unknown objective={self.objective!r}")
        if self.offtarget_temperature <= 0:
            raise ValueError("offtarget_temperature must be > 0")
        if self.offtarget_boost_weight < 0:
            raise ValueError("offtarget_boost_weight must be >= 0")
        if self.offtarget_boost_tolerance < 0:
            raise ValueError("offtarget_boost_tolerance must be >= 0")
        if self.gain_transform not in ("raw", "log2-fold-change"):
            raise ValueError(f"Unknown gain_transform={self.gain_transform!r}")
        if self.gain_pseudocount <= 0:
            raise ValueError("gain_pseudocount must be > 0")
        if self.edit_penalty_weight < 0:
            raise ValueError("edit_penalty_weight must be >= 0")
        if self.edit_budget < 0:
            raise ValueError("edit_budget must be >= 0")

    def bend(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.bending_factor:
            return tensor
        exp_neg = torch.exp(-tensor)
        adjustment = self.bending_factor * (exp_neg - 1.0)
        return tensor - adjustment

    def _summarize_predictions(self, preds: torch.Tensor) -> dict[str, object]:
        finite_preds = preds[torch.isfinite(preds)]
        observed_min = (
            float(finite_preds.min().detach().cpu()) if finite_preds.numel() else math.nan
        )
        observed_max = (
            float(finite_preds.max().detach().cpu()) if finite_preds.numel() else math.nan
        )
        if finite_preds.numel() and (observed_min < self.a_min or observed_max > self.a_max):
            logger.warning(
                "Design prediction range [%g, %g] exceeds configured clamp bounds [%g, %g]; "
                "predictions will be clipped before scoring.",
                observed_min,
                observed_max,
                self.a_min,
                self.a_max,
            )
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

        return {
            "group_scores": group_scores,
            "per_track": per_track,
            "track_mean": track_mean,
            "track_max": track_max,
            "track_topk": track_topk,
        }

    def set_reference(self, one_hot_seed) -> None:
        """Cache the unedited, single-sequence seed: per-fold group scores and the sequence.

        The group scores are what ``selective-activation`` measures gain against; the sequence
        itself is what :meth:`_edit_penalty` measures drift from. Callers set the reference when
        either is in play -- see ``design.run._score_one_candidate``.
        """
        preds = self.ensemble.predict(one_hot_seed)
        if preds.shape[1] != 1:
            raise ValueError("the design reference must contain exactly one sequence")
        summary = self._summarize_predictions(preds)
        group_scores = summary["group_scores"]
        assert isinstance(group_scores, dict)
        self._reference_group_scores = {
            name: score.detach().clone() for name, score in group_scores.items()
        }
        sequence = one_hot_seed
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)
        self._reference_sequence = sequence.detach().clone()

    def _edit_penalty(self, one_hot_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """``(n_edits, penalty)`` for a batch, both ``(B,)`` and both in the energy's own units.

        ``n_edits`` is the Hamming distance to the reference sequence. Positions outside the
        search's editable span are identical to the reference by construction, so counting over
        the whole context is the same as counting over the editable span -- the energy does not
        need to know where that span is.

        A position counts as one edit when its one-hot *column* differs at all, rather than when
        ``argmax`` differs: an ``N`` base is an all-zero column, whose ``argmax`` is an arbitrary
        0 that would silently compare equal to an ``A``.

        The penalty is ``weight * relu(n_edits - edit_budget)``. With the default
        ``edit_budget=0`` that is a flat per-edit cost, so a greedy search accepts an edit only
        when it buys more than ``edit_penalty_weight`` of energy -- which makes the search
        self-terminating rather than bounded by its round count.
        """
        reference = self._reference_sequence
        if reference is None:
            raise RuntimeError(
                f"edit_penalty_weight={self.edit_penalty_weight} requires "
                "set_reference(unedited_seed) before scoring"
            )
        batch = one_hot_batch
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)
        reference = reference.to(device=batch.device)
        if batch.shape[1:] != reference.shape[1:]:
            raise ValueError(
                f"batch shape {tuple(batch.shape)} is not broadcastable against reference "
                f"shape {tuple(reference.shape)}"
            )
        n_edits = (batch != reference).any(dim=-2).sum(dim=-1).to(batch.dtype)
        penalty = self.edit_penalty_weight * torch.relu(n_edits - self.edit_budget)
        return n_edits, penalty

    def _gain(self, score: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.gain_transform == "raw":
            return score - reference
        score_shifted = score + self.gain_pseudocount
        reference_shifted = reference + self.gain_pseudocount
        if bool((score_shifted <= 0).any()) or bool((reference_shifted <= 0).any()):
            raise ValueError(
                "log2-fold-change requires group scores + gain_pseudocount to be positive"
            )
        return torch.log2(score_shifted / reference_shifted)

    def forward(self, one_hot_batch) -> EnergyResult:
        # torch.inference_mode() lives in the predictor, not here, so a future gradient
        # path through the energy is not blocked by the objective itself.
        preds = self.ensemble.predict(one_hot_batch)  # (n_folds, B, n_tracks, n_bins)
        summary = self._summarize_predictions(preds)
        group_scores = summary["group_scores"]
        assert isinstance(group_scores, dict)
        per_track = summary["per_track"]
        track_mean = summary["track_mean"]
        track_max = summary["track_max"]
        track_topk = summary["track_topk"]
        assert isinstance(per_track, torch.Tensor)
        assert isinstance(track_mean, torch.Tensor)
        assert isinstance(track_max, torch.Tensor)
        assert isinstance(track_topk, torch.Tensor)

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

        per_fold_specificity = offtarget - self.target_alpha * target_score
        group_names = [self.groups.target, *other_names]
        if self.objective == "selective-activation":
            if self._reference_group_scores is None:
                raise RuntimeError(
                    "selective-activation requires set_reference(unedited_seed) before scoring"
                )
            group_gains = {
                name: self._gain(group_scores[name], self._reference_group_scores[name])
                for name in group_names
            }
            target_gain_by_fold = group_gains[self.groups.target]
            other_gains = torch.stack([group_gains[name] for name in other_names], dim=-1)
            positive_boosts = torch.relu(other_gains - self.offtarget_boost_tolerance)
            scaled_boosts = positive_boosts / self.offtarget_temperature
            # log-mean-exp is a smooth maximum whose baseline is exactly zero when every
            # positive boost is zero.
            smooth_boost = self.offtarget_temperature * (
                torch.logsumexp(scaled_boosts, dim=-1)
                - torch.logsumexp(torch.zeros_like(scaled_boosts), dim=-1)
            )
            per_fold_energy = (
                -self.target_alpha * target_gain_by_fold
                + self.offtarget_boost_weight * smooth_boost
            )
        else:
            group_gains = {name: torch.zeros_like(group_scores[name]) for name in group_names}
            target_gain_by_fold = torch.zeros_like(target_score)
            smooth_boost = torch.zeros_like(target_score)
            per_fold_energy = per_fold_specificity

        if self.fold_reduction == "mean":
            energy = per_fold_energy.mean(dim=0)
            specificity = per_fold_specificity.mean(dim=0)
        elif self.fold_reduction == "mean_plus_std":
            energy = per_fold_energy.mean(dim=0) + per_fold_energy.std(dim=0)
            specificity = per_fold_specificity.mean(dim=0) + per_fold_specificity.std(dim=0)
        else:
            raise ValueError(f"Unknown fold_reduction={self.fold_reduction!r}")

        # Added after the fold reduction, not inside per_fold_energy: the drift term is a
        # property of the sequence alone, identical for every fold, so folding it in would make
        # `mean_plus_std` treat a constant as if it carried cross-fold disagreement.
        if self.edit_penalty_weight:
            n_edits, edit_penalty = self._edit_penalty(one_hot_batch)
            energy = energy + edit_penalty
        else:
            n_edits = torch.zeros_like(energy)
            edit_penalty = torch.zeros_like(energy)

        per_group = torch.stack([group_scores[name] for name in group_names], dim=-1).mean(dim=0)
        per_group_gain = torch.stack([group_gains[name] for name in group_names], dim=-1).mean(
            dim=0
        )

        return EnergyResult(
            energy=energy,
            specificity=specificity,
            target=target_score.mean(dim=0),
            per_group=per_group,
            per_fold_energy=per_fold_energy,
            per_fold_specificity=per_fold_specificity,
            group_names=group_names,
            per_track=per_track.mean(dim=0),
            track_mean=track_mean.mean(dim=0),
            track_max=track_max.mean(dim=0),
            track_topk=track_topk.mean(dim=0),
            track_topk_ratio=(track_topk / track_mean.clamp_min(1e-8)).mean(dim=0),
            target_gain=target_gain_by_fold.mean(dim=0),
            per_group_gain=per_group_gain,
            offtarget_boost=smooth_boost.mean(dim=0),
            n_edits=n_edits,
            edit_penalty=edit_penalty,
            objective=self.objective,
            has_reference=self._reference_group_scores is not None,
            gain_transform=self.gain_transform,
            gain_pseudocount=self.gain_pseudocount,
        )


# --------------------------------------------------------------------------- #
# Named energy factories for region-count models                             #
# --------------------------------------------------------------------------- #
#
# All three read a region ensemble's per-group softplus(eta) scores -- scale is
# log1p(rate), roughly 0-8 for realistic count magnitudes -- so any finite `a_min`/`a_max`
# clamp bound passed through **kwargs must be given in that log space, not in raw counts.
# `bend`'s default `bending_factor=0.0` is a no-op (see `SpecificityEnergy.bend`); none of
# these factories touch it.
#
# Absolute energies are NOT comparable across the three: `logsumexp` adds a constant
# `log(G-1)` relative to `mean`/`max` (a soft maximum over `n` terms of similar scale sits
# roughly `log(n)` above the plain mean), so only rankings/argmin *within* one factory's
# objective are meaningful, never a magnitude compared across factories.
def _region_energy(
    ensemble, groups: TrackGroups, bins: slice, *, fixed_offtarget_reduction, **kwargs
):
    if "offtarget_reduction" in kwargs:
        raise TypeError(
            f"offtarget_reduction is fixed by this factory (to {fixed_offtarget_reduction!r}); "
            "construct SpecificityEnergy directly if you need a different one"
        )
    return SpecificityEnergy(
        ensemble, groups, bins, offtarget_reduction=fixed_offtarget_reduction, **kwargs
    )


def contrast_energy(ensemble, groups: TrackGroups, bins: slice, **kwargs) -> SpecificityEnergy:
    """The selection metric ``contrast_pearson_<group>``, up to a positive constant.

    ``offtarget_reduction="mean"``: with the default ``target_alpha=1.0``, this energy's
    ``specificity`` equals ``-(n_groups / (n_groups - 1))`` times
    ``RegionContrastReadout(ensemble, group_index=target, centre=True)``'s centred score --
    i.e. the *same* quantity ``training/regions/metrics.py``'s ``_contrast`` computes for
    ``contrast_pearson_<group>``, just rescaled. Lower energy therefore means higher contrast,
    with identical argmin and rankings to the training-time metric. This is the default choice
    for region models: it is literally what they are selected on.
    """
    return _region_energy(ensemble, groups, bins, fixed_offtarget_reduction="mean", **kwargs)


def worst_offtarget_energy(
    ensemble, groups: TrackGroups, bins: slice, **kwargs
) -> SpecificityEnergy:
    """Soft-max over off-target groups: punishes the single worst off-target cell type.

    ``offtarget_reduction="logsumexp"``: a smooth (temperature-controlled via
    ``offtarget_temperature``) approximation to the hard worst-case in
    :func:`max_offtarget_energy`, differentiable everywhere and dominated by whichever
    off-target group currently scores highest. Use this over :func:`contrast_energy` when a
    single leaky off-target cell type is unacceptable even if the *average* off-target signal
    is low.
    """
    return _region_energy(ensemble, groups, bins, fixed_offtarget_reduction="logsumexp", **kwargs)


def max_offtarget_energy(ensemble, groups: TrackGroups, bins: slice, **kwargs) -> SpecificityEnergy:
    """Hard worst-case off-target: ``offtarget_reduction="max"``.

    Like :func:`worst_offtarget_energy` but with a true (non-differentiable-at-the-max) maximum
    instead of its smooth approximation -- use when the search method doesn't need gradients
    through the energy (e.g. greedy ISM-based search) and an exact worst-case is preferred to
    an approximation.
    """
    return _region_energy(ensemble, groups, bins, fixed_offtarget_reduction="max", **kwargs)
