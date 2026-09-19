"""Anchor-scaled target-vs-reference specificity score, computed from cached per-bin profiles.

Pure numpy: no I/O, no torch, no sklearn. This is a building block for a later training-label
generator / design objective (see the plan this module was introduced for), so its API is kept
small, importable, and tested independently of any notebook or model code.

Pipeline, matching :func:`regulonado.dataset.build.transform_signal`'s anchor-unit convention:

1. :func:`region_statistic` — per-bin bigwig profiles (padded to a common bin count) -> one
   statistic per region per track, in *stored* (pre-anchor) units.
2. :func:`to_anchor_units` — stored units -> anchor units, via ``transform_signal``.
3. :func:`group_mean` — replicate-averaged anchor units per cell-type/assay group (linear
   units — replicates are averaged before any log).
4. :func:`target_specificity_score` — log2 anchor units, target group vs. a quantile/top-k
   reduction over every other group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from regulonado.normalization import rolling_window_stat

__all__ = [
    "TargetSpecificityResult",
    "rolling_window_stat",
    "region_statistic",
    "group_index_from_records",
    "to_anchor_units",
    "group_mean",
    "target_specificity_score",
]


@dataclass(frozen=True)
class TargetSpecificityResult:
    """Output of :func:`target_specificity_score`, one row per region.

    Attributes:
        score: ``(R,)`` log2 anchor-unit units, ``target_log - reference_log``. Directly
            comparable to ``target_vs_reference_log2fc`` elsewhere in the pipeline.
        target_log: ``(R,)`` ``log2(target_group_signal + pseudocount)``.
        reference_log: ``(R,)`` the reduction (quantile or top-k mean) over non-target
            groups' ``log2(group_signal + pseudocount)``.
        group_signal: ``(R, G)`` replicate-averaged anchor units per group (linear units,
            before the log/reduction above).
        group_names: length-``G`` group labels, in the order of ``group_signal``'s columns.
        target_rank: ``(R,)`` 1-indexed rank of the target group's log2 signal among *all*
            ``G`` groups (target included), 1 = highest.
        strongest_reference: ``(R,)`` int index into ``group_names`` of the highest-signal
            non-target group.
    """

    score: np.ndarray
    target_log: np.ndarray
    reference_log: np.ndarray
    group_signal: np.ndarray
    group_names: list[str]
    target_rank: np.ndarray
    strongest_reference: np.ndarray


def region_statistic(
    profiles: np.ndarray,
    lengths: Sequence[int] | np.ndarray,
    *,
    bin_size: int,
    window_stat_bp: int,
    stat: Literal["rolling-max", "mean"] = "rolling-max",
) -> np.ndarray:
    """Reduce padded per-bin profiles ``(R, T, B)`` to one statistic per region/track ``(R, T)``.

    ``profiles[r, t, :lengths[r]]`` are the valid bins for region ``r``; bins at index
    ``>= lengths[r]`` are padding and are always excluded, regardless of ``stat`` — a ragged
    region must never have its statistic diluted by trailing padding shared with a longer
    region in the same batch.

    ``stat="rolling-max"`` (the anchor-scale convention; matches
    :func:`regulonado.normalization.track_window_stat`) takes the maximum rolling mean at
    ``width = max(1, round(window_stat_bp / bin_size))`` bins over the valid bins only.
    ``stat="mean"`` is the plain mean over the valid bins only.
    """
    profiles = np.asarray(profiles, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)
    n_regions, n_tracks, _ = profiles.shape
    if lengths.shape != (n_regions,):
        raise ValueError(f"lengths must have shape ({n_regions},), got {lengths.shape}")

    width = max(1, round(window_stat_bp / bin_size))
    out = np.empty((n_regions, n_tracks), dtype=np.float32)
    for r in range(n_regions):
        length = int(lengths[r])
        valid = profiles[r, :, :length]
        if stat == "rolling-max":
            for t in range(n_tracks):
                out[r, t] = rolling_window_stat(valid[t], width)
        elif stat == "mean":
            values = np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
            out[r, :] = values.mean(axis=1)
        else:
            raise ValueError(f"Unknown stat: {stat!r}")
    return out


def group_index_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    key: str = "group",
    assay_classes: set[str] | None = None,
    quality: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Map track-table ``records`` to a per-track group index, filtering as requested.

    ``records`` is a list of dict-like rows (e.g. from ``tracks.parquet``), one per track, in
    track order. ``group_index[t]`` is the index into the returned ``group_names`` for track
    ``t``'s ``record[key]`` value, or ``-1`` if the track is excluded.

    A track is excluded (``group_index == -1``) when:

    - ``record[key]`` is missing/``None``, or
    - ``assay_classes`` is given and ``record.get("assay_class")`` is not in it, or
    - ``quality`` is given and ``record.get("quality")`` is not in it.

    Group names are the distinct, non-excluded ``record[key]`` values, sorted alphabetically
    (stable and independent of track order) — this is the order used everywhere else in this
    module (``group_signal`` columns, ``group_names`` in :class:`TargetSpecificityResult`).
    Excluded tracks never contribute a group even if their raw ``key`` value would otherwise
    be a valid group label.
    """
    included_names: set[str] = set()
    raw_keys: list[object] = []
    keep: list[bool] = []
    for record in records:
        value = record.get(key)
        ok = value is not None
        if ok and assay_classes is not None:
            ok = record.get("assay_class") in assay_classes
        if ok and quality is not None:
            ok = record.get("quality") in quality
        raw_keys.append(value)
        keep.append(ok)
        if ok:
            included_names.add(str(value))

    group_names = sorted(included_names)
    name_to_index = {name: i for i, name in enumerate(group_names)}
    group_index = np.full(len(records), -1, dtype=np.int64)
    for t, (value, ok) in enumerate(zip(raw_keys, keep)):
        if ok:
            group_index[t] = name_to_index[str(value)]
    return group_index, group_names


def to_anchor_units(
    signal: np.ndarray,
    *,
    scale_factor: np.ndarray,
    background: np.ndarray,
    clip_hard: float | None = None,
    background_mode: Literal["subtract", "scale-only"] = "subtract",
) -> np.ndarray:
    """Convert stored-units ``signal (R, T)`` to anchor units, via ``transform_signal``.

    Delegates to :func:`regulonado.dataset.build.transform_signal` (transposed to ``(T, R)``
    to match its per-track convention) with ``apply_squash=False`` — anchor units are the
    scaled, clipped, *unsquashed* signal. This is the guarantee that a score computed here is
    the same number a training label would carry.

    ``background_mode="subtract"`` passes ``background`` through to ``transform_signal``,
    which floors the result at 0 *before* scaling (subtract, clamp, then multiply by
    ``scale_factor``). ``background_mode="scale-only"`` passes ``background=None``, skipping
    subtraction entirely (``transform_signal`` on ``background=None`` is a no-op there).

    ``clip_hard`` is an absolute anchor-unit ceiling; ``None`` disables clipping.
    """
    from regulonado.dataset.build import transform_signal

    signal = np.asarray(signal, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32) if background_mode == "subtract" else None
    apply_clip = clip_hard is not None
    ch = float(clip_hard) if clip_hard is not None else float("inf")
    transformed = transform_signal(
        signal.T,
        np.asarray(scale_factor, dtype=np.float32),
        ch,
        ch,
        background=bg,
        apply_squash=False,
        apply_clip=apply_clip,
    )
    return transformed.T


def group_mean(signal: np.ndarray, group_index: np.ndarray, n_groups: int) -> np.ndarray:
    """Linear average of tracks within each group -> ``(R, G)``.

    ``signal`` is ``(R, T)`` (e.g. anchor units). Tracks with ``group_index == -1`` are
    ignored entirely (excluded from every group's average, not folded into a catch-all
    group). A group with no member tracks gets ``nan``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    group_index = np.asarray(group_index, dtype=np.int64)
    n_regions = signal.shape[0]
    out = np.full((n_regions, n_groups), np.nan, dtype=np.float64)
    for g in range(n_groups):
        members = group_index == g
        if members.any():
            out[:, g] = signal[:, members].mean(axis=1)
    return out


def target_specificity_score(
    signal: np.ndarray,
    *,
    scale_factor: np.ndarray,
    background: np.ndarray,
    group_index: np.ndarray,
    group_names: Sequence[str],
    target_group: str,
    background_mode: Literal["subtract", "scale-only"] = "subtract",
    pseudocount: float = 0.1,
    reduction: Literal["quantile", "top-k"] = "quantile",
    quantile: float = 0.9,
    top_k: int = 3,
    clip_hard: float | None = None,
) -> TargetSpecificityResult:
    """Score each region for how much higher the target group's signal is than the rest.

    Pipeline: stored-units ``signal (R, T)`` -> anchor units (:func:`to_anchor_units`) ->
    replicate-averaged linear group signal (:func:`group_mean`) -> ``log2(x + pseudocount)``.

    ``score = target_log - reduce({group_log[g] : g != target})``, in **log2 units**,
    directly comparable to ``target_vs_reference_log2fc`` elsewhere in the pipeline.

    ``reduction="quantile"`` reduces the non-target groups' logs with
    ``np.quantile(..., quantile, method="linear")`` (explicit — this is not numpy's default
    on all versions). E.g. ``quantile=0.9`` over 53 reference groups sits between the 47th and
    48th largest values (interpolated). ``quantile=1.0`` is exactly the max, i.e.
    ``score = target_log - max(reference_log)``.

    ``reduction="top-k"`` reduces with the mean of the ``top_k`` largest non-target group
    logs; ``top_k=1`` is identical to ``quantile=1.0``.

    Because ``background_mode="subtract"`` clamps at 0 before the pseudocount (see
    :func:`to_anchor_units`), every group log is floored at ``log2(pseudocount)`` — group
    signals genuinely below the background floor are indistinguishable from 0, by
    construction. ``background_mode="scale-only"`` has no such floor.

    ``target_rank`` is 1-indexed, 1 = highest ``log2`` signal among *all* ``len(group_names)``
    groups including the target itself (so a target that is not even the strongest group
    among all groups gets rank > 1). ``strongest_reference`` is the index into ``group_names``
    of the highest-signal group *excluding* the target.

    Raises:
        ValueError: if ``target_group`` is not in ``group_names``.
    """
    group_names = list(group_names)
    if target_group not in group_names:
        raise ValueError(f"target_group {target_group!r} not found in group_names: {group_names}")
    target_idx = group_names.index(target_group)
    n_groups = len(group_names)

    anchor = to_anchor_units(
        signal,
        scale_factor=scale_factor,
        background=background,
        clip_hard=clip_hard,
        background_mode=background_mode,
    )
    group_signal = group_mean(anchor, group_index, n_groups)
    group_log = np.log2(group_signal + pseudocount)

    target_log = group_log[:, target_idx]
    reference_mask = np.ones(n_groups, dtype=bool)
    reference_mask[target_idx] = False
    reference_log_all = group_log[:, reference_mask]
    reference_names = [name for i, name in enumerate(group_names) if i != target_idx]

    if reduction == "quantile":
        reference_log = np.quantile(reference_log_all, quantile, axis=1, method="linear")
    elif reduction == "top-k":
        k = min(top_k, reference_log_all.shape[1])
        top = np.sort(reference_log_all, axis=1)[:, -k:]
        reference_log = top.mean(axis=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    score = target_log - reference_log

    order = np.argsort(-group_log, axis=1)
    target_rank = np.argmax(order == target_idx, axis=1) + 1

    strongest_ref_local = np.argmax(reference_log_all, axis=1)
    strongest_reference = np.array(
        [group_names.index(reference_names[i]) for i in strongest_ref_local], dtype=np.int64
    )

    return TargetSpecificityResult(
        score=score,
        target_log=target_log,
        reference_log=reference_log,
        group_signal=group_signal,
        group_names=list(group_names),
        target_rank=target_rank.astype(np.int64),
        strongest_reference=strongest_reference,
    )
