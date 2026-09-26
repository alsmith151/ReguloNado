"""Negative-log-likelihood loss for per-track region counts.

Ported from ``unique_enhancer_finding.modelling.loss.CountLikelihoodLoss`` (UEF), which
this loss reproduces numerically -- see ``tests/test_regions_loss.py``, which checks it
against hard-coded values computed by importing the UEF module directly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["COUNT_NOISE_MODELS", "CountLikelihoodLoss"]

#: keras.backend.epsilon()-style floor, used wherever a mean-based renormalisation
#: divides by a weight or size sum that could be zero.
_EPSILON = 1e-7

#: Noise models :class:`CountLikelihoodLoss` supports, with the bounds its per-track
#: noise parameter is clamped to (see the class docstring). Verbatim from UEF: an
#: unbounded learned noise buys loss by declaring a task noisy and then barely
#: constrains its predictions (the Kendall failure mode), so both ends are clamped.
COUNT_NOISE_MODELS: dict[str, tuple[float, float]] = {
    # log dispersion alpha, var = mu + alpha * mu^2: 0.1 .. 2.0.
    "nb": (-2.303, 0.693),
    # log sigma of log1p(count): 0.05 .. 7.4
    "lognormal": (-3.0, 2.0),
}


class CountLikelihoodLoss(nn.Module):
    """Negative log-likelihood of raw per-track counts, with learned per-track noise.

    Takes the per-track log mean ``log_mu`` (group log rate + fixed size-factor offset
    + replicate offset, see :class:`~regulonado.regions.model.CountHead`) and a
    per-track noise parameter, both owned by the model so ``Trainer`` optimises them.

    The noise parameter is the Kendall et al. (2018) idea -- learn each task's noise
    scale jointly with the mean, so noisy tracks are down-weighted by the likelihood
    rather than by hand-set weights:

    ``noise="nb"`` (default)
        Negative binomial, ``var = mu + alpha_t * mu**2``, parameter ``log alpha_t``.
        The natural count model: overdispersed replicate/batch noise, and a variance
        that grows with the mean, so low-count regions carry less weight without a
        pseudocount.
    ``noise="lognormal"``
        Kendall's homoscedastic form on ``log1p(count)``:
        ``0.5 * exp(-2 s_t) * (log1p(y) - log1p(mu))**2 + s_t``, parameter
        ``s_t = log sigma_t``. Kept for comparison with the log-MSE models.

    The parameter is clamped to *noise_bounds* (default :data:`COUNT_NOISE_MODELS`).
    ``noise_shrinkage`` adds ``shrinkage * var(log_noise)``, a mild pull of every track
    toward the across-track mean, so no single track's noise wanders to a bound on
    little evidence.

    Labels may contain ``NaN``: those elements (extreme single-track counts masked
    upstream) are skipped by the likelihood and left out of the group pooling for the
    contrast term.

    ``contrast_weight`` adds the group-level specificity term: MSE between each
    region's centred ``log1p(contrast_multiplier * rate)`` across groups, predicted
    (``exp(eta)``) vs observed (pooled replicate counts over pooled size factors). It
    is the loss counterpart of ``contrast_pearson`` on the same scale the metric uses,
    and ``contrast_task_weights`` (per group) concentrates it on the target.
    """

    def __init__(
        self,
        noise: str = "nb",
        task_weights: Tensor | None = None,
        contrast_weight: float = 0.0,
        contrast_multiplier: float = 1.0,
        contrast_task_weights: Tensor | None = None,
        noise_shrinkage: float = 0.0,
        noise_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if noise not in COUNT_NOISE_MODELS:
            raise ValueError(f"noise must be one of {sorted(COUNT_NOISE_MODELS)}; got {noise!r}")
        self.noise = noise
        self.noise_bounds = tuple(noise_bounds) if noise_bounds else COUNT_NOISE_MODELS[noise]
        self.noise_shrinkage = noise_shrinkage
        self.contrast_weight = contrast_weight
        self.contrast_multiplier = contrast_multiplier
        self.task_weights: Tensor | None
        self.contrast_task_weights: Tensor | None
        for name, weights in (
            ("task_weights", task_weights),
            ("contrast_task_weights", contrast_task_weights),
        ):
            if weights is not None:
                self.register_buffer(name, torch.as_tensor(weights, dtype=torch.float32))
            else:
                setattr(self, name, None)

    def clamp_noise(self, log_noise: Tensor) -> Tensor:
        low, high = self.noise_bounds
        return log_noise.clamp(min=low, max=high)

    def nll(self, y_true: Tensor, log_mu: Tensor, log_noise: Tensor) -> Tensor:
        """Elementwise negative log-likelihood ``[batch, n_tracks]`` (``NaN`` where *y_true* is)."""
        log_noise = self.clamp_noise(log_noise)
        if self.noise == "nb":
            log_r = -log_noise  # r = 1 / alpha
            r = log_r.exp()
            log_r_plus_mu = torch.logaddexp(log_r.expand_as(log_mu), log_mu)
            return (
                torch.lgamma(r)
                + torch.lgamma(y_true + 1)
                - torch.lgamma(y_true + r)
                - r * (log_r - log_r_plus_mu)
                - y_true * (log_mu - log_r_plus_mu)
            )
        # log1p(exp(log_mu)) == softplus(log_mu), stable for any log_mu
        residual = torch.log1p(y_true) - nn.functional.softplus(log_mu)
        return 0.5 * torch.exp(-2 * log_noise) * residual**2 + log_noise

    def forward(
        self,
        y_true: Tensor,
        log_mu: Tensor,
        log_noise: Tensor,
        eta: Tensor | None = None,
        track_groups: Tensor | None = None,
        log_size_factors: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        """Scalar loss for ``[batch, n_tracks]`` counts.

        *eta* (``[batch, n_groups]`` group log rates), *track_groups* and
        *log_size_factors* are needed only for the contrast term.
        """
        y_true = y_true.float()
        log_mu = log_mu.float()
        observed = torch.isfinite(y_true)
        y_filled = torch.where(observed, y_true, torch.zeros_like(y_true))
        mask = observed.to(log_mu.dtype)

        elementwise = self.nll(y_filled, log_mu, log_noise.float()) * mask
        weight = mask
        if self.task_weights is not None:
            weights = self.task_weights.to(elementwise.dtype)
            weights = weights / weights.mean()
            elementwise = elementwise * weights
            weight = weight * weights

        region_weight = None
        if sample_weight is not None:
            region_weight = sample_weight.to(elementwise.dtype).flatten()
            region_weight = region_weight / region_weight.mean().clamp(min=_EPSILON)
            elementwise = elementwise * region_weight.unsqueeze(-1)
            weight = weight * region_weight.unsqueeze(-1)
        # Weighted mean over observed elements. Task and region weights each average
        # 1, so with nothing masked weight.sum() == numel and this is exactly the
        # unmasked mean.
        loss = elementwise.sum() / weight.sum().clamp(min=_EPSILON)

        if self.noise_shrinkage:
            clamped = self.clamp_noise(log_noise.float())
            loss = loss + self.noise_shrinkage * ((clamped - clamped.mean()) ** 2).mean()

        if self.contrast_weight:
            if eta is None or track_groups is None or log_size_factors is None:
                raise ValueError("the contrast term needs eta, track_groups and log_size_factors")
            n_groups = eta.shape[-1]
            batch = y_true.shape[0]
            pooled = y_true.new_zeros(batch, n_groups).index_add_(1, track_groups, y_filled)
            size = y_true.new_zeros(batch, n_groups).index_add_(
                1, track_groups, mask * log_size_factors.exp()
            )
            log_m = torch.log(torch.as_tensor(self.contrast_multiplier, dtype=eta.dtype))
            pred_c = nn.functional.softplus(eta.float() + log_m)
            true_c = torch.log1p(self.contrast_multiplier * pooled / size.clamp(min=1e-12))
            # A group with every replicate masked in a region has no observed rate
            # there; stand in the prediction, so it adds no error.
            true_c = torch.where(size > 0, true_c, pred_c.detach())
            error = (
                (pred_c - pred_c.mean(dim=-1, keepdim=True))
                - (true_c - true_c.mean(dim=-1, keepdim=True))
            ) ** 2
            if self.contrast_task_weights is not None:
                weights = self.contrast_task_weights.to(error.dtype)
                error = error * weights / weights.mean()
            if region_weight is not None:
                error = error * region_weight.unsqueeze(-1)
            loss = loss + self.contrast_weight * error.mean()
        return loss
