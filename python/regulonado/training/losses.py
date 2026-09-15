from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from regulonado.metrics import paired_group_masks


def squash(y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(y) * (torch.sqrt(torch.abs(y).clamp(min=0) + 1) - 1) + eps * y


def scaled_poisson_multinomial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_factors: torch.Tensor | None = None,
    clip_hard: torch.Tensor | None = None,
    poisson_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
) -> torch.Tensor:
    seq_len = target.shape[-1]
    _, n_tracks = pred.shape[:2]

    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon

    if clip_hard is not None:
        clip_hard_tensor = torch.as_tensor(clip_hard, dtype=y_true.dtype, device=y_true.device)
        if clip_hard_tensor.ndim == 1:
            clip_hard_tensor = clip_hard_tensor.reshape(1, n_tracks, 1)
        y_true = torch.minimum(y_true, clip_hard_tensor)

    if scale_factors is not None:
        scale_tensor = torch.as_tensor(scale_factors, dtype=y_pred.dtype, device=y_pred.device)
        if scale_tensor.ndim == 1:
            scale_tensor = scale_tensor.reshape(1, n_tracks, 1)
        y_true = y_true * scale_tensor
        y_pred = y_pred * scale_tensor

    s_true = y_true.sum(dim=-1, keepdim=True)
    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred.clamp(min=epsilon)

    poisson_term = (
        F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0.0, reduction="mean") / seq_len
    )
    multinomial_term = -(y_true * torch.log(p_pred.clamp(min=epsilon))).sum(dim=-1) / seq_len
    combined_loss = multinomial_term + poisson_weight * poisson_term
    if rescale:
        combined_loss = combined_loss * 2.0 / (1.0 + poisson_weight)
    return combined_loss.mean()


def poisson_multinomial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    poisson_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
    weight_range: float = 0.0,
    weight_exp: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Poisson-multinomial loss.

    Args:
        reduction: "mean" (default) returns the usual scalar, averaged over both batch
            and track dimensions. "none" instead averages over the batch dimension only
            and returns a per-track loss vector ``[T]`` — used by callers that apply
            their own per-track weighting (e.g. learnable track loss weights) before
            the final reduction.
    """
    seq_len = target.shape[-1]
    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon
    s_true = y_true.sum(dim=-1, keepdim=True)
    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred
    if weight_range:
        positions = torch.linspace(-1, 1, seq_len, device=pred.device, dtype=pred.dtype)
        weights = (1.0 + weight_range * positions.abs()).pow(weight_exp)
    else:
        weights = 1.0
    # [B, T, 1] -> [B, T]: per-(example, track) Poisson NLL on the summed total count.
    poisson_term = (
        F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0.0, reduction="none").squeeze(-1)
        / seq_len
    )
    multinomial_term = -(y_true * torch.log(p_pred) * weights).sum(dim=-1) / seq_len
    combined_loss = multinomial_term + poisson_weight * poisson_term  # [B, T]
    if rescale:
        combined_loss = combined_loss * 2.0 / (1.0 + poisson_weight)
    if reduction == "none":
        return combined_loss.mean(dim=0)  # [T]
    return combined_loss.mean()


def poisson_multinomial_binwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    poisson_weight: float = 1.0,
    epsilon: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multinomial profile term + a PER-BIN Poisson NLL term.

    Unlike ``poisson_multinomial_loss``, whose Poisson term is on the summed total count
    (``s_pred`` vs ``s_true``) and therefore only constrains *how much* signal a window gets,
    here the Poisson term is computed per bin (``pred`` vs ``target``).  A true-zero bin with
    prediction ``lambda`` costs ``~lambda`` with a constant gradient of 1, so misplaced /
    "fake" peaks are penalised directly rather than being free as long as the total matches.
    The multinomial term is retained purely as a profile-shape regulariser; the per-bin
    penalisation comes entirely from the Poisson term.

    Args:
        pred: model predictions [B, T, L].
        target: ground-truth signal [B, T, L].
        poisson_weight: weight on the per-bin Poisson term relative to the multinomial term.
            The per-bin Poisson is the workhorse here, so this defaults to 1.0 (vs 0.2 for the
            total-count variant).
        reduction: "mean" (default) returns the usual scalar. "none" instead averages over
            the batch and bin dimensions only and returns a per-track loss vector ``[T]`` —
            used by callers that apply their own per-track weighting before the final
            reduction.
    """
    seq_len = target.shape[-1]
    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon
    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred
    multinomial_term = -(y_true * torch.log(p_pred)).sum(dim=-1) / seq_len  # [B, T]
    # Per-bin Poisson NLL, kept at [B, T, L] so it can be reduced per track below.
    poisson_term = F.poisson_nll_loss(
        y_pred, target.float(), log_input=False, eps=0.0, full=False, reduction="none"
    ).mean(dim=-1)  # [B, T]
    combined_loss = multinomial_term + poisson_weight * poisson_term  # [B, T]
    if reduction == "none":
        return combined_loss.mean(dim=0)  # [T]
    return combined_loss.mean()


def kendall_track_weighted_loss(per_track_loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Combine a per-track loss vector into a scalar with learnable per-track uncertainty.

    Homoscedastic uncertainty weighting (Kendall et al., 2018): each track's loss is
    divided by a learned variance ``exp(log_var)`` and the log-variance is added back as a
    regulariser, so a track can only shrink its effective weight by paying a matching log
    penalty — this stops the trivial solution of driving every weight to zero. Requires
    ``per_track_loss`` to still carry a track dimension (see the ``reduction="none"``
    option on :func:`poisson_multinomial_loss` / :func:`poisson_multinomial_binwise_loss`).
    """
    precision = torch.exp(-log_var)
    return (per_track_loss * precision + log_var).mean()


def transfer_calibration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    profile_weight: float = 1.0,
    total_weight: float = 0.5,
    bin_weight: float = 0.1,
    bin_mode: str = "signal_weighted",
    bin_signal_power: float = 1.0,
    bin_threshold: float = 0.0,
    topk_bin_weight: float = 0.0,
    topk_bin_count: int = 0,
    topk_huber_delta: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    seq_len = target.shape[-1]
    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon
    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred.clamp(min=epsilon)

    multinomial_term = -(y_true * torch.log(p_pred.clamp(min=epsilon))).sum(dim=-1).mean() / seq_len
    total_term = F.mse_loss(torch.log1p(s_pred), torch.log1p(y_true.sum(dim=-1, keepdim=True)), reduction="mean")
    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)
    bin_error = (log_pred - log_true).square()
    if bin_mode == "signal_weighted":
        weights = y_true.clamp_min(0).pow(bin_signal_power)
        bin_term = (bin_error * weights).sum() / weights.sum().clamp_min(epsilon)
    elif bin_mode == "threshold_masked":
        mask = y_true >= bin_threshold
        bin_term = bin_error.masked_select(mask).mean() if mask.any() else bin_error.mean() * 0.0
    elif bin_mode == "mean":
        bin_term = bin_error.mean()
    else:
        raise ValueError(f"Unknown transfer calibration bin_mode={bin_mode!r}")
    topk_term = y_pred.new_zeros(())

    if topk_bin_weight > 0 and topk_bin_count > 0:
        k = min(topk_bin_count, y_true.shape[-1])
        topk_indices = torch.topk(y_true, k=k, dim=-1).indices
        topk_true = torch.gather(y_true, dim=-1, index=topk_indices)
        topk_pred = torch.gather(y_pred, dim=-1, index=topk_indices)
        topk_term = F.huber_loss(
            torch.log1p(topk_pred),
            torch.log1p(topk_true),
            delta=topk_huber_delta,
            reduction="mean",
        )

    return (
        profile_weight * multinomial_term
        + total_weight * total_term
        + bin_weight * bin_term
        + topk_bin_weight * topk_term
    )


def topk_additive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk_fraction: float = 0.04,
    topk_weight: float = 1.0,
    poisson_weight: float = 0.2,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Poisson-multinomial loss with an additive top-K term.

    Computes the base poisson_multinomial loss over all bins, then adds a second
    pass of the same loss restricted to the top-K bins by target signal, weighted
    by ``topk_weight``.  The top-K pass contributes an extra gradient push on the
    most active bins without touching the calibration of quiescent bins.

    Args:
        pred: model predictions [B, T, L].
        target: ground-truth signal [B, T, L].
        topk_fraction: fraction of bins selected as "peaks" (e.g. 0.04 = top 4%).
        topk_weight: scalar weight on the additive top-K term.
        poisson_weight: weight on the Poisson total-count sub-term (shared by both passes).
    """
    L = pred.shape[-1]
    k = max(1, int(round(topk_fraction * L)))

    base_loss = poisson_multinomial_loss(
        pred, target, poisson_weight=poisson_weight, epsilon=epsilon
    )

    topk_idx = target.topk(k, dim=-1).indices          # [B, T, k]
    pred_k   = pred.gather(-1, topk_idx)
    target_k = target.gather(-1, topk_idx)
    peak_loss = poisson_multinomial_loss(
        pred_k, target_k, poisson_weight=poisson_weight, epsilon=epsilon
    )

    return base_loss + topk_weight * peak_loss


def topk_reweight_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk_fraction: float = 0.04,
    topk_weight: float = 4.0,
    poisson_weight: float = 0.2,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Poisson-multinomial loss with per-bin rank weighting on the multinomial term.

    The multinomial cross-entropy contribution from each bin is multiplied by
    ``topk_weight`` if that bin falls in the top-K by target signal, and 1.0
    otherwise.  The Poisson total-count term is unaffected (it operates on the
    per-track sum, not per-bin values).  This is a smooth, single-pass loss with
    no discontinuity at the K boundary when ``topk_weight`` is not too large.

    Args:
        pred: model predictions [B, T, L].
        target: ground-truth signal [B, T, L].
        topk_fraction: fraction of bins classified as "peaks" for upweighting.
        topk_weight: multiplier applied to the multinomial term at peak bins.
        poisson_weight: weight on the Poisson total-count sub-term.
    """
    L = pred.shape[-1]
    k = max(1, int(round(topk_fraction * L)))

    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon

    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred.clamp(min=epsilon)
    s_true = y_true.sum(dim=-1, keepdim=True)

    # Per-bin multinomial cross-entropy: -y_true * log(p_pred) / L
    per_bin_ce = -(y_true * torch.log(p_pred.clamp(min=epsilon)))   # [B, T, L]

    # Build a weight mask: topk_weight for peak bins, 1.0 elsewhere.
    topk_idx = y_true.topk(k, dim=-1).indices                       # [B, T, k]
    bin_weights = torch.ones_like(per_bin_ce)
    bin_weights.scatter_(-1, topk_idx, topk_weight)

    multinomial_term = (bin_weights * per_bin_ce).sum(dim=-1) / L   # [B, T]

    poisson_term = (
        F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0.0, reduction="mean") / L
    )

    return (multinomial_term.mean() + poisson_weight * poisson_term)


def log1p_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    pred_log = torch.log1p(torch.clamp_min(pred.float(), 0.0) + epsilon)
    target_log = torch.log1p(torch.clamp_min(target.float(), 0.0) + epsilon)
    return F.huber_loss(pred_log, target_log, delta=delta, reduction="mean")


def poisson_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Per-bin Poisson NLL with no multinomial normalisation.

    Unlike poisson_multinomial_loss, this loss has no normalisation step — each
    bin is penalised independently.  The gradient at bin i is ``1 - target_i /
    pred_i``, which is strongly negative when the model underpredicts a peak,
    giving direct magnitude correction.  This is suitable as a primary training
    loss on squash-transformed or raw count targets.
    """
    return F.poisson_nll_loss(
        pred.float().clamp(min=epsilon),
        target.float(),
        log_input=False,
        full=False,
        reduction="mean",
    )


def paired_binwise_log2fc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    condition_ids: torch.Tensor,
    *,
    shared_track_index: torch.Tensor | None = None,
    pseudocount: float = 1.0,
    delta: float = 0.5,
) -> torch.Tensor:
    pair_masks = paired_group_masks(condition_ids, shared_track_index)
    if not pair_masks:
        return pred.new_zeros(())

    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for baseline_mask, perturbed_mask in pair_masks:
        pred_baseline = pred[:, baseline_mask].mean(dim=1)
        pred_perturbed = pred[:, perturbed_mask].mean(dim=1)
        target_baseline = target[:, baseline_mask].mean(dim=1)
        target_perturbed = target[:, perturbed_mask].mean(dim=1)
        pred_lfc = torch.log2(pred_perturbed + pseudocount) - torch.log2(
            pred_baseline + pseudocount
        )
        target_lfc = torch.log2(target_perturbed + pseudocount) - torch.log2(
            target_baseline + pseudocount
        )
        pred_chunks.append(pred_lfc)
        target_chunks.append(target_lfc)

    return F.huber_loss(
        torch.cat(pred_chunks, dim=-1),
        torch.cat(target_chunks, dim=-1),
        delta=delta,
        reduction="mean",
    )


def contrast_family_weights(
    families: Sequence[object | None],
    groups: Sequence[object | None],
) -> torch.Tensor:
    """Group-balanced averaging weights ``[F, T]`` for cross-track specificity.

    Tracks sharing a family label (for example ``assay_class``) are compared with one
    another. Each distinct group (cell type) in a family receives equal total weight,
    split evenly across its replicate tracks, so heavily replicated cell types do not
    dominate the family mean. Tracks missing either label are left out, and families with
    fewer than two groups are omitted because they contain no cross-group contrast.
    """
    if len(families) != len(groups):
        raise ValueError(
            "families and groups need one label per track, "
            f"got {len(families)} and {len(groups)}"
        )
    n_tracks = len(families)
    members_by_family: dict[str, list[tuple[int, str]]] = {}
    for index, (family, group) in enumerate(zip(families, groups)):
        if family is not None and group is not None:
            members_by_family.setdefault(str(family), []).append((index, str(group)))

    rows: list[torch.Tensor] = []
    for family in sorted(members_by_family):
        members = members_by_family[family]
        group_sizes: dict[str, int] = {}
        for _, group in members:
            group_sizes[group] = group_sizes.get(group, 0) + 1
        if len(group_sizes) < 2:
            continue
        row = torch.zeros(n_tracks, dtype=torch.float32)
        for index, group in members:
            row[index] = 1.0 / (len(group_sizes) * group_sizes[group])
        rows.append(row)
    if not rows:
        return torch.zeros((0, n_tracks), dtype=torch.float32)
    return torch.stack(rows)


def track_contrast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    family_weights: torch.Tensor,
    *,
    region_bins: int = 16,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Count-weighted KL between observed and predicted signal allocation across tracks.

    Bins are summed into ``region_bins``-wide regions. Within each region and contrast
    family (rows of ``family_weights``, see :func:`contrast_family_weights`), the observed
    allocation ``p_t ∝ w_t · y_t`` over member tracks is compared with the predicted
    allocation ``q_t ∝ w_t · ŷ_t``. The loss is zero whenever predictions are proportional
    to targets within a family, so it constrains cell-type differences and leaves the
    shared magnitude to the base loss. Regions are weighted by observed family signal, as
    in the multinomial profile term.
    """
    batch, n_tracks, length = pred.shape
    n_regions = length // region_bins
    if family_weights.shape[0] == 0 or n_regions == 0:
        return pred.sum() * 0.0
    usable = n_regions * region_bins
    region_pred = (
        pred.float()[..., :usable].reshape(batch, n_tracks, n_regions, region_bins).sum(-1)
    )
    region_true = (
        target.float()[..., :usable].reshape(batch, n_tracks, n_regions, region_bins).sum(-1)
    )
    weights = family_weights.to(device=pred.device, dtype=region_pred.dtype)[None, :, :, None]
    weighted_true = weights * region_true.clamp_min(0.0)[:, None]
    weighted_pred = weights * region_pred.clamp_min(0.0)[:, None] + epsilon * (weights > 0)
    family_mass = weighted_true.sum(dim=2)  # [B, F, R]
    p_true = weighted_true / family_mass.unsqueeze(2).clamp_min(epsilon)
    log_q = torch.log(weighted_pred.clamp_min(epsilon * epsilon)) - torch.log(
        weighted_pred.sum(dim=2, keepdim=True)
    )
    kl = (torch.xlogy(p_true, p_true) - p_true * log_q).sum(dim=2)
    return (family_mass * kl).sum() / family_mass.sum().clamp_min(epsilon)
