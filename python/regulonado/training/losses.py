from __future__ import annotations

import torch
import torch.nn.functional as F


def squash(y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(y) * (torch.sqrt(torch.abs(y).clamp(min=0) + 1) - 1) + eps * y


def _paired_group_masks(
    condition_ids: torch.Tensor,
    shared_track_index: torch.Tensor | None = None,
    *,
    baseline_condition_id: int = 0,
    perturbed_condition_id: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    condition_ids = condition_ids.to(dtype=torch.long)
    if shared_track_index is None:
        baseline_mask = condition_ids == baseline_condition_id
        perturbed_mask = condition_ids == perturbed_condition_id
        if baseline_mask.any() and perturbed_mask.any():
            return [(baseline_mask, perturbed_mask)]
        return []

    shared_track_index = shared_track_index.to(device=condition_ids.device, dtype=torch.long)
    pair_masks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group_id in torch.unique(shared_track_index, sorted=True):
        in_group = shared_track_index == group_id
        baseline_mask = in_group & (condition_ids == baseline_condition_id)
        perturbed_mask = in_group & (condition_ids == perturbed_condition_id)
        if baseline_mask.any() and perturbed_mask.any():
            pair_masks.append((baseline_mask, perturbed_mask))
    return pair_masks


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
) -> torch.Tensor:
    seq_len = target.shape[-1]
    y_true = target.float() + epsilon
    y_pred = pred.float() + epsilon
    s_true = y_true.sum(dim=-1, keepdim=True)
    s_pred = y_pred.sum(dim=-1, keepdim=True)
    p_pred = y_pred / s_pred
    poisson_term = (
        F.poisson_nll_loss(s_pred, s_true, log_input=False, eps=0.0, reduction="mean") / seq_len
    )
    multinomial_term = -(y_true * torch.log(p_pred)).sum(dim=-1) / seq_len
    combined_loss = multinomial_term + poisson_weight * poisson_term
    if rescale:
        combined_loss = combined_loss * 2.0 / (1.0 + poisson_weight)
    return combined_loss.mean()


def transfer_calibration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    raw_pred: torch.Tensor | None = None,
    raw_target: torch.Tensor | None = None,
    profile_weight: float = 1.0,
    total_weight: float = 0.5,
    bin_weight: float = 0.1,
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

    raw_y_true = raw_target.float() + epsilon if raw_target is not None else y_true
    raw_y_pred = raw_pred.float() + epsilon if raw_pred is not None else y_pred
    raw_s_true = raw_y_true.sum(dim=-1, keepdim=True)
    raw_s_pred = raw_y_pred.sum(dim=-1, keepdim=True)

    multinomial_term = -(y_true * torch.log(p_pred.clamp(min=epsilon))).sum(dim=-1).mean() / seq_len
    total_term = F.mse_loss(torch.log1p(raw_s_pred), torch.log1p(raw_s_true), reduction="mean")
    bin_term = F.mse_loss(torch.log1p(raw_y_pred), torch.log1p(raw_y_true), reduction="mean")
    topk_term = y_pred.new_zeros(())

    if topk_bin_weight > 0 and topk_bin_count > 0:
        k = min(topk_bin_count, raw_y_true.shape[-1])
        topk_indices = torch.topk(raw_y_true, k=k, dim=-1).indices
        topk_true = torch.gather(raw_y_true, dim=-1, index=topk_indices)
        topk_pred = torch.gather(raw_y_pred, dim=-1, index=topk_indices)
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
    pair_masks = _paired_group_masks(condition_ids, shared_track_index)
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
