"""Region-level count head trained on cached backbone embeddings.

``RegionCountModel`` sits on top of a frozen backbone's cached per-region features
(:mod:`regulonado.embeddings`, added separately): a small learned head -- attention
pooling, an MLP and a per-track count head -- predicts one NB (or log-normal) rate per
group (cell type) from those features, trained against raw per-track BAM counts
(:mod:`regulonado.counts`, added separately).

``CountHead`` is ported near-verbatim from
``unique_enhancer_finding.modelling.model.CountHead`` (UEF), and the soft eta cap
follows UEF ``model.py``'s ``_trunk`` (around line 548): ``eta_max - softplus(eta_max -
eta)``, identity well below the cap, never above it -- without it, cell types with
large learned dispersion drift to 100-500x overshoots, since NB barely penalises
over-prediction there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from regulonado.regions.loss import COUNT_NOISE_MODELS, CountLikelihoodLoss

__all__ = [
    "AttentionPool",
    "CountHead",
    "RegionCountConfig",
    "RegionCountModel",
    "RegionCountOutput",
]


class AttentionPool(nn.Module):
    """Learned attention pooling over the ``K`` bins of a cached region, ``[B, K, D] -> [B, D]``.

    A single-head scoring linear layer produces one logit per bin; bins are combined by
    their softmax weight. Simpler than a full multi-head attention block, but the pool
    is still learned rather than a fixed mean/max, so the head can weight the bins that
    best summarise a 1 kb scored region instead of a plain average of cached positions.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """``features``: ``[batch, k, feature_dim]`` -> pooled ``[batch, feature_dim]``."""
        logits = self.score(features).squeeze(-1)  # [B, K]
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bk,bkd->bd", weights, features)


class CountHead(nn.Module):
    """Map group log rates to per-track log means, and hold the per-track noise.

    Ported from ``unique_enhancer_finding.modelling.model.CountHead``.

    ``log_mu[:, t] = eta[:, g(t)] + o_t + delta_t``, where

    - ``eta`` is the trunk's output, one log rate per group (cell type) in
      size-factor-normalised units -- what the model predicts, and what specificity is
      computed on;
    - ``o_t`` is the track's fixed ``log_size_factor`` (count-unit anchor scaling, from
      :mod:`regulonado.counts`), a buffer;
    - ``delta_t`` is a learnable replicate offset, centred within each group so it sums
      to zero there. Centring keeps it identifiable -- a group-wide shift belongs to
      ``eta`` -- while it absorbs what the anchor factor missed between replicates
      (depth, PE vs SE, batch). A group with one track always gets ``delta = 0``.

    ``log_noise`` is the per-track noise parameter of
    :class:`~regulonado.regions.loss.CountLikelihoodLoss` (log NB dispersion, or log
    sigma). Neither parameter is weight-decayed (see
    :meth:`RegionCountModel.count_head_parameter_names`): shrinking an offset or a
    noise scale toward zero is not regularisation of the sequence model.
    """

    def __init__(
        self,
        track_groups: list[int],
        log_size_factors: list[float],
        n_groups: int,
        noise: str = "nb",
        replicate_offsets: bool = True,
    ) -> None:
        super().__init__()
        if len(track_groups) != len(log_size_factors):
            raise ValueError("track_groups and log_size_factors must have one entry per track")
        # Validate on the Python list: from_pretrained builds modules on the meta
        # device, where tensor comparisons cannot be read back.
        if min(track_groups) < 0 or max(track_groups) >= n_groups:
            raise ValueError(f"track_groups must index 0..{n_groups - 1}")
        groups = torch.as_tensor(track_groups, dtype=torch.long)
        self.n_groups = n_groups
        self.register_buffer("track_groups", groups)
        self.register_buffer(
            "log_size_factors", torch.as_tensor(log_size_factors, dtype=torch.float32)
        )
        self.register_buffer(
            "group_sizes", torch.zeros(n_groups).index_add_(0, groups, torch.ones(len(groups)))
        )
        n_tracks = len(track_groups)
        self.replicate_offsets = nn.Parameter(torch.zeros(n_tracks)) if replicate_offsets else None
        low, high = COUNT_NOISE_MODELS[noise]
        # Start at alpha ~ 0.22 for NB (moderately overdispersed) and at sigma = 1 for
        # the log-normal; both inside the clamp bounds.
        initial = -1.5 if noise == "nb" else 0.0
        self.log_noise = nn.Parameter(torch.full((n_tracks,), min(max(initial, low), high)))

    @property
    def n_tracks(self) -> int:
        return int(self.track_groups.numel())

    def centred_offsets(self) -> Tensor:
        """``delta_t`` minus its group mean, ``[n_tracks]`` (zeros without offsets)."""
        if self.replicate_offsets is None:
            return self.log_size_factors.new_zeros(self.n_tracks)
        delta = self.replicate_offsets
        group_mean = delta.new_zeros(self.n_groups).index_add(0, self.track_groups, delta)
        group_mean = group_mean / self.group_sizes
        return delta - group_mean[self.track_groups]

    def forward(self, eta: Tensor) -> Tensor:
        """``eta`` ``[batch, n_groups]`` -> per-track ``log_mu`` ``[batch, n_tracks]``."""
        return eta[:, self.track_groups] + self.log_size_factors + self.centred_offsets()


class RegionCountConfig(PretrainedConfig):
    """Config for :class:`RegionCountModel`, saved as ``config.json`` by ``save_pretrained``.

    Stores everything needed to reconstruct the head architecture, wire it to a cached
    backbone's embeddings, and reproduce the training-time loss -- no external metadata
    files needed at inference time.

    Parameters
    ----------
    backbone_name
        Name of the backbone the cached features came from (e.g. ``"alphagenome"``,
        ``"borzoi"``); informational -- the head itself is backbone-agnostic and only
        depends on ``k``/``d``.
    backbone_checkpoint
        Pretrained checkpoint identifier for the backbone, for provenance.
    k
        Number of cached bins per region the head pools over.
    d
        Feature dimension of each cached bin.
    n_tracks, n_groups
        Number of per-track label columns and output groups (cell types).
    track_groups
        Per-track group index, ``[n_tracks]``, values in ``0..n_groups - 1``.
    log_size_factors
        Per-track fixed log size factor (count-unit anchor scaling), ``[n_tracks]``.
    track_names, group_names
        Optional display names, one per track / group; used by metrics dict keys.
    hidden
        Width of the MLP between the pooled feature and the group log rates.
    dropout
        Dropout applied in the pooling MLP.
    eta_max
        Soft cap on the predicted group log rate: ``eta_max - softplus(eta_max -
        eta)``, identity well below the cap, never above it. ``None`` disables it.
    count_noise
        ``"nb"`` or ``"lognormal"`` (see :data:`~regulonado.regions.loss.COUNT_NOISE_MODELS`).
    count_replicate_offsets
        Whether :class:`CountHead` learns a centred per-track replicate offset.
    count_freeze_noise
        Freeze :class:`CountHead`'s ``log_noise`` (no gradient), for a later stage that
        keeps the noise scale from an earlier run.
    count_noise_shrinkage
        Passed through to :class:`~regulonado.regions.loss.CountLikelihoodLoss`.
    loss_contrast_weight, loss_contrast_multiplier, task_weights, contrast_task_weights
        Passed through to :class:`~regulonado.regions.loss.CountLikelihoodLoss`; see
        its docstring.
    """

    model_type = "regulonado_region_count"

    def __init__(
        self,
        backbone_name: str | None = None,
        backbone_checkpoint: str | None = None,
        k: int = 1,
        d: int = 1536,
        n_tracks: int | None = None,
        n_groups: int | None = None,
        track_groups: list[int] | None = None,
        log_size_factors: list[float] | None = None,
        track_names: list[str] | None = None,
        group_names: list[str] | None = None,
        hidden: int = 512,
        dropout: float = 0.1,
        eta_max: float | None = None,
        count_noise: str = "nb",
        count_replicate_offsets: bool = True,
        count_freeze_noise: bool = False,
        count_noise_shrinkage: float = 0.0,
        loss_contrast_weight: float = 0.0,
        loss_contrast_multiplier: float | None = None,
        task_weights: list[float] | None = None,
        contrast_task_weights: list[float] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        # Default to a single track/group rather than raising: transformers'
        # ``PretrainedConfig`` machinery (``save_pretrained`` -> generation-parameter
        # diffing) constructs ``self.__class__()`` with no arguments internally, so a
        # required-argument validation error here breaks every save, not just a
        # genuinely missing config. Real callers always pass explicit lists.
        if track_groups is None:
            track_groups = [0]
        if log_size_factors is None:
            log_size_factors = [0.0] * len(track_groups)
        if len(log_size_factors) != len(track_groups):
            raise ValueError("track_groups and log_size_factors must have one entry per track")
        resolved_n_tracks = len(track_groups)
        if n_tracks is not None and n_tracks != resolved_n_tracks:
            raise ValueError(f"n_tracks={n_tracks} disagrees with {resolved_n_tracks} track_groups")
        resolved_n_groups = (max(track_groups) + 1) if n_groups is None else n_groups
        if max(track_groups) >= resolved_n_groups or min(track_groups) < 0:
            raise ValueError(f"track_groups must index 0..{resolved_n_groups - 1}")
        if track_names is not None and len(track_names) != resolved_n_tracks:
            raise ValueError("track_names must have one entry per track")
        if group_names is not None and len(group_names) != resolved_n_groups:
            raise ValueError("group_names must have one entry per group")
        if count_noise not in COUNT_NOISE_MODELS:
            raise ValueError(f"count_noise must be one of {sorted(COUNT_NOISE_MODELS)}")
        for name, weights, expected in (
            ("task_weights", task_weights, resolved_n_tracks),
            ("contrast_task_weights", contrast_task_weights, resolved_n_groups),
        ):
            if weights is not None and len(weights) != expected:
                raise ValueError(f"{name} has {len(weights)} entries; expected {expected}")

        self.backbone_name = backbone_name
        self.backbone_checkpoint = backbone_checkpoint
        self.k = k
        self.d = d
        self.n_tracks = resolved_n_tracks
        self.n_groups = resolved_n_groups
        self.track_groups = [int(g) for g in track_groups]
        self.log_size_factors = [float(o) for o in log_size_factors]
        self.track_names = [str(name) for name in track_names] if track_names is not None else None
        self.group_names = [str(name) for name in group_names] if group_names is not None else None
        self.hidden = hidden
        self.dropout = dropout
        self.eta_max = eta_max
        self.count_noise = count_noise
        self.count_replicate_offsets = count_replicate_offsets
        self.count_freeze_noise = count_freeze_noise
        self.count_noise_shrinkage = count_noise_shrinkage
        self.loss_contrast_weight = loss_contrast_weight
        self.loss_contrast_multiplier = loss_contrast_multiplier
        self.task_weights = task_weights
        self.contrast_task_weights = contrast_task_weights


@dataclass
class RegionCountOutput(ModelOutput):
    """``RegionCountModel.forward`` output.

    ``logits`` are per-group expected rates ``exp(eta)``, ``[batch, n_groups]`` -- the
    same quantity :class:`~regulonado.regions.metrics.GroupedCountMetrics` pools raw
    per-track labels to via :func:`~regulonado.regions.metrics.group_count_rates`, so
    ``eval_pred = (predictions, labels)`` from a ``transformers.Trainer`` eval loop can
    be fed to ``GroupedCountMetrics`` directly. ``loss`` is ``None`` at inference
    (``labels`` not given) or in a ``ModelOutput``/``nn.DataParallel`` context, where
    ``None`` fields are dropped instead of gathered.
    """

    loss: Tensor | None = None
    logits: Tensor | None = None


class RegionCountModel(PreTrainedModel):
    """Attention pool + MLP + :class:`CountHead`, trained on cached backbone features.

    ``forward`` follows the HF convention -- ``logits``, plus ``loss`` when ``labels``
    is passed -- so ``transformers.Trainer`` can drive it without a custom
    ``compute_loss``.
    """

    config_class = RegionCountConfig
    main_input_name = "features"

    #: See ``UefRegressionModel.accepts_loss_kwargs`` (UEF ``model.py``):
    #: ``CountLikelihoodLoss`` is already a mean over regions and tracks, so
    #: ``Trainer``'s ``num_processes`` rescaling for token-level losses does not apply.
    accepts_loss_kwargs = False

    def __init__(self, config: RegionCountConfig) -> None:
        super().__init__(config)
        self.pool = AttentionPool(config.d)
        self.norm = nn.LayerNorm(config.d)
        self.mlp = nn.Sequential(
            nn.Linear(config.d, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.n_groups),
        )
        self.count_head = CountHead(
            track_groups=config.track_groups,
            log_size_factors=config.log_size_factors,
            n_groups=config.n_groups,
            noise=config.count_noise,
            replicate_offsets=config.count_replicate_offsets,
        )
        self.count_head.log_noise.requires_grad_(not config.count_freeze_noise)
        self.loss_fn = self.build_loss()
        self.post_init()

    def build_loss(self) -> CountLikelihoodLoss:
        """The loss the config asks for; called again when a later stage re-weights it."""
        config = self.config
        return CountLikelihoodLoss(
            noise=config.count_noise,
            task_weights=torch.as_tensor(config.task_weights) if config.task_weights else None,
            contrast_weight=config.loss_contrast_weight,
            contrast_multiplier=(
                1.0 if config.loss_contrast_multiplier is None else config.loss_contrast_multiplier
            ),
            contrast_task_weights=(
                torch.as_tensor(config.contrast_task_weights)
                if config.contrast_task_weights
                else None
            ),
            noise_shrinkage=config.count_noise_shrinkage,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """No-op: submodules keep their constructor init.

        ``post_init`` otherwise re-initialises every submodule with transformers'
        generic scheme (normal(0.02) weights, zero bias), which would discard
        :class:`CountHead`'s empirical noise seeding. See
        ``RegulonadoModel._init_weights`` for the same pattern.
        """

    def _trunk(self, features: Tensor) -> Tensor:
        """Pooled features through the MLP to group log rates, with the soft eta cap applied."""
        pooled = self.norm(self.pool(features))
        eta = self.mlp(pooled)
        eta_max = self.config.eta_max
        if eta_max is not None:
            eta = eta_max - nn.functional.softplus(eta_max - eta)
        return eta

    def forward(
        self,
        features: Tensor,
        labels: Tensor | None = None,
        sample_weight: Tensor | None = None,
        **kwargs: object,
    ) -> RegionCountOutput:
        """``features``: ``[batch, k, d]`` (cast to float32 regardless of cache dtype).

        ``labels``, when given: ``[batch, n_tracks]`` raw per-track counts, ``NaN``
        where masked.
        """
        eta = self._trunk(features.float())
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                y_true=labels,
                log_mu=self.count_head(eta),
                log_noise=self.count_head.log_noise,
                eta=eta,
                track_groups=self.count_head.track_groups,
                log_size_factors=self.count_head.log_size_factors,
                sample_weight=sample_weight,
            )
        return RegionCountOutput(loss=loss, logits=eta.exp())

    def count_head_parameter_names(self) -> list[str]:
        """Fully-qualified names of :class:`CountHead`'s learnable parameters.

        For a training runner to exclude from weight decay: shrinking a replicate
        offset or a noise scale toward zero is not regularisation of the sequence
        model (see :class:`CountHead`'s docstring).
        """
        return [name for name, _ in self.count_head.named_parameters(prefix="count_head")]
