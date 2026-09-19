"""Group-contrast label and reduced-specificity-score for the group-contrast head.

Pure torch: no I/O, no config objects. Computes a ``[B, G, L]`` label from a per-track
target tensor already in anchor units (``counts / exposure`` under
``data.label_space: counts``, ``data.exposure: anchor`` — see
:mod:`regulonado.training.label_space`), plus a differentiable reduction of any
``[B, C, L]`` channel tensor (label or model output) to the validated target-vs-reference
specificity score.

Pipeline, matching :mod:`regulonado.target_specificity`'s numpy reference implementation
(see the cross-check in ``tests/test_group_contrast.py``):

1. :func:`smooth_bins` — rolling **mean** (never max) over ``smoothing_bins``, stride 1,
   same length as the input.
2. Subtract per-track ``background`` (anchor units) and clamp at 0, when given.
3. :func:`group_replicate_weights` gives the ``[G, T]`` matrix that averages replicate
   tracks within a group **in linear anchor units** — this is what makes the label
   invariant to duplicating a replicate track (see
   ``test_target_specificity.test_score_invariant_to_duplicating_a_reference_track`` and
   its mirror here).
4. ``log2(x + pseudocount)``.
5. Subtract the per-bin gauge (median or mean) over groups. Every pairwise channel
   difference, and :func:`reduce_target_score`'s target-vs-reference difference, is exactly
   unchanged by the choice of gauge or its value: the gauge is a per-bin, per-group-set
   constant added to every group equally, and both a channel difference and a quantile
   over channels shift by exactly that constant, which then cancels in the subtraction.

The validated statistic (``notebooks/results/target-specificity/summary.md``:
HL-60 rank 1 at 87% of curated enhancers, AUC 0.964) is the **rolling-mean max over a
region**, i.e. ``reduce_target_score(...).max()`` over bins within a region, computed with
``smoothing_bins`` chosen so the smoothing window covers the same span as
``target_specificity.region_statistic``'s ``window_stat_bp`` at the dataset's ``bin_size``
(``round(window_stat_bp / bin_size)``, e.g. ``round(1000 / 32) == 31``, which is this
module's default). ``tests/test_group_contrast.py``'s decisive
cross-check ties this module's output to that validated number exactly (to float32
precision): see the module docstring there for why an *exact* match is achievable despite
:func:`smooth_bins` being a same-length (edge-padded) rolling mean while
:func:`target_specificity.region_statistic`'s rolling-max is a shorter, unpadded
(``mode="valid"``) one — the two agree bin-for-bin away from the array edges, and the test
places the compared maximum away from any edge.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

__all__ = [
    "group_replicate_weights",
    "smooth_bins",
    "group_contrast_labels",
    "reduce_target_score",
]


def group_replicate_weights(group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Replicate-averaging weights ``[G, T]`` from a per-track group assignment.

    ``group_index`` is ``[T]`` int, ``group_index[t]`` the index (``0..n_groups-1``) of
    track ``t``'s group, or ``-1`` to exclude the track from every group (the same
    convention as :func:`regulonado.target_specificity.group_index_from_records`). Row
    ``g`` is ``1 / count`` on each member track of group ``g`` and 0 elsewhere, so it sums
    to 1 — applying it to a ``[..., T, ...]`` tensor along the track axis is a plain
    (linear-space) average of that group's replicate tracks. A group with no member tracks
    gets an all-zero row.
    """
    group_index = torch.as_tensor(group_index, dtype=torch.long)
    n_tracks = group_index.shape[0]
    weights = torch.zeros((n_groups, n_tracks), dtype=torch.float32)
    for g in range(n_groups):
        members = group_index == g
        count = int(members.sum())
        if count:
            weights[g, members] = 1.0 / count
    return weights


def smooth_bins(signal: torch.Tensor, width: int) -> torch.Tensor:
    """Rolling mean of ``signal`` over its last dim, stride 1, same length as the input.

    ``width=1`` is the identity. For ``width > 1``, edges are replicate-padded
    (``(width - 1) // 2`` on the left, the remainder on the right) so the output has the
    same length as the input; interior bins whose window doesn't touch the padding are
    identical to an unpadded (``mode="valid"``) rolling mean at the corresponding position
    — the property the decisive cross-check in ``tests/test_group_contrast.py`` relies on.
    Differentiable w.r.t. ``signal``.
    """
    if width <= 1:
        return signal
    *lead, length = signal.shape
    flat = signal.reshape(-1, 1, length)
    pad_left = (width - 1) // 2
    pad_right = width - 1 - pad_left
    padded = F.pad(flat, (pad_left, pad_right), mode="replicate")
    kernel = torch.full((1, 1, width), 1.0 / width, dtype=signal.dtype, device=signal.device)
    smoothed = F.conv1d(padded, kernel)
    return smoothed.reshape(*lead, length)


def group_contrast_labels(
    normalised_target: torch.Tensor,
    *,
    group_weights: torch.Tensor,
    background: torch.Tensor | None = None,
    smoothing_bins: int = 31,
    pseudocount: float = 0.1,
    gauge: Literal["median", "mean"] = "median",
    floor: float = 0.139,
    clamp: tuple[float, float] = (-6.0, 6.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauge-centred log2 group-contrast labels ``[B, G, L]`` and a validity mask ``[B, 1, L]``.

    ``normalised_target`` is ``[B, T, L]`` anchor units (``counts / exposure`` under
    ``data.label_space: counts``, ``data.exposure: anchor``). Order of operations (see the
    module docstring for why each step is where it is):

    1. :func:`smooth_bins` over ``smoothing_bins`` (rolling mean).
    2. Subtract ``background`` (``[T]``, anchor units) per track and clamp at 0, if given;
       ``background=None`` skips this step entirely ("scale-only", matching
       :func:`regulonado.target_specificity.to_anchor_units`'s ``background_mode``).
    3. Replicate-average within each group **in linear units** via ``group_weights``
       (``[G, T]``, from :func:`group_replicate_weights`) -> ``[B, G, L]``.
    4. ``log2(x + pseudocount)``.
    5. Subtract the per-bin gauge over groups: the median (``torch.quantile(..., 0.5)``,
       linear-interpolated) or the mean, per ``gauge``.

    ``clamp`` bounds the final gauge-centred labels.

    ``mask`` is 1.0 at a bin where **at least one** group's post-background linear signal
    (step 3's ``[B, G, L]`` output, before the log) exceeds ``floor`` (an anchor-unit noise
    ceiling), else 0.0. This is a per-bin, all-groups-OR mask, not a per-group one: a group
    sitting at its own floor while another group is far above it is not masked out, because
    "target high, this group at floor" is exactly the specificity signal being labelled.
    """
    if gauge not in ("median", "mean"):
        raise ValueError(f"gauge must be 'median' or 'mean', got {gauge!r}")

    smoothed = smooth_bins(normalised_target, smoothing_bins)  # [B, T, L]
    if background is not None:
        bg = torch.as_tensor(background, dtype=smoothed.dtype, device=smoothed.device)
        smoothed = (smoothed - bg.reshape(1, -1, 1)).clamp_min(0.0)

    weights = group_weights.to(dtype=smoothed.dtype, device=smoothed.device)
    group_linear = torch.einsum("gt,btl->bgl", weights, smoothed)  # [B, G, L]

    group_log = torch.log2(group_linear + pseudocount)
    if gauge == "median":
        centre = torch.quantile(group_log, 0.5, dim=1, keepdim=True, interpolation="linear")
    else:
        centre = group_log.mean(dim=1, keepdim=True)
    labels = (group_log - centre).clamp(min=clamp[0], max=clamp[1])

    mask = (group_linear > floor).any(dim=1, keepdim=True).to(dtype=normalised_target.dtype)
    return labels, mask


def reduce_target_score(
    channels: torch.Tensor, *, target_index: int, quantile: float = 0.9
) -> torch.Tensor:
    """``channels[:, target_index] - quantile(channels[:, others])`` -> ``[B, L]``.

    ``channels`` is ``[B, C, L]`` (group-contrast labels, or a model's predicted channels
    of the same shape — this function is applied to both). The reference reduction is
    ``torch.quantile(..., quantile, dim=1, interpolation="linear")`` over every channel
    except ``target_index``, matching
    :func:`regulonado.target_specificity.target_specificity_score`'s ``reduction="quantile"``
    bin-for-bin. ``quantile=1.0`` is exactly the max over the other channels. Differentiable
    w.r.t. ``channels`` (through both the target term and the quantile's subgradient).

    Raises:
        ValueError: if ``channels`` has fewer than 2 channels (no "other" channels to
            reduce over).
    """
    n_channels = channels.shape[1]
    if n_channels < 2:
        raise ValueError(
            f"reduce_target_score needs at least 2 channels (target + >=1 other), got {n_channels}"
        )
    other_indices = [i for i in range(n_channels) if i != target_index]
    other = channels[:, other_indices, :]
    target = channels[:, target_index, :]
    reference = torch.quantile(other, quantile, dim=1, interpolation="linear")
    return target - reference
