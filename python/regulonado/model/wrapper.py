from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from regulonado.model.config import RegulonadoConfig

if TYPE_CHECKING:
    pass


def one_hot_sequence_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """One-hot encode uint8 sequence tokens (A0 C1 G2 T3, N/pad>=4) to shape [B, 4, L].

    Bases 4 and above (N and out-of-contig padding) have no set bit — an all-zero
    column, matching the one-hot padding convention the backbone already expects.
    """
    bases = torch.arange(4, device=tokens.device).view(1, 4, 1)
    return (tokens.unsqueeze(1) == bases).float()


class BackboneAdapter(Iterable):
    """Protocol satisfied by BorzoiBackboneAdapter and EnformerBackboneAdapter."""

    feature_dim: int

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]: ...


@dataclass(slots=True)
class FreezePolicy:
    """Policy for freezing and unfreezing model components during training.

    Controls which parts of the backbone are trainable. By default, the entire
    backbone is frozen (backbone-as-feature-extractor). Stages can be
    selectively unfrozen from the output end or by module name.

    Parameters
    ----------
    freeze_backbone : bool, optional
        If True (default), freeze all backbone parameters. Then selectively
        unfreeze using other fields.
    unfreeze_backbone_stages_from_output_end : int, optional
        Number of backbone stages (blocks) to unfreeze starting from the end.
        By default 0 (all frozen).
    unfreeze_module_names : tuple[str, ...], optional
        Names of specific modules to unfreeze (e.g., ("transformer.10",)).
        Empty by default.
    """

    freeze_backbone: bool = True
    unfreeze_backbone_stages_from_output_end: int = 0
    unfreeze_module_names: tuple[str, ...] = field(default_factory=tuple)


class RegulonadoModel(PreTrainedModel):
    """Backbone + prediction head, loadable via ``from_pretrained``.

    When constructed directly (during training), pass pre-built ``backbone`` and ``head``.
    When loaded for inference, ``from_pretrained(checkpoint_dir)`` reconstructs architecture
    from ``config.json`` then loads merged weights from ``model.safetensors``.
    """

    config_class = RegulonadoConfig

    def __init__(
        self,
        config: RegulonadoConfig | None = None,
        backbone: nn.Module | None = None,
        head: nn.Module | None = None,
    ):
        if config is None:
            config = RegulonadoConfig(feature_dim=int(getattr(backbone, "feature_dim", 1920)))
        super().__init__(config)
        if backbone is not None:
            self.backbone = backbone
        else:
            from regulonado.model.adapters import build_backbone_architecture

            self.backbone = build_backbone_architecture(
                config.backbone_type,
                config.config_overrides,
                config.target_length,
                config.pretrained_name,
            )
        if head is not None:
            self.head = head
        else:
            self.head = _build_head(config)
        # Carried on the model (per the composite-head plan) so compute_loss/metrics code
        # can split concatenated [B, T + G, L] logits without guessing from shape alone.
        # Plain single-output heads don't set this attribute, so it falls back to n_tracks
        # (the whole of logits) -- a no-op split, keeping that path unchanged.
        self.track_channel_count: int = getattr(self.head, "track_channel_count", config.n_tracks)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Leave every module with the initialisation its constructor gave it.

        ``post_init`` otherwise applies transformers' generic init (normal(0.02) weights,
        zero biases) to the head, erasing the empirical output-bias seeding, zeroed output
        weights, and zero-initialised metadata modulation layers. The Borzoi/Enformer
        backbone is a nested ``PreTrainedModel`` and keeps its own init dispatch.
        """

    def forward(self, input_ids: torch.Tensor, **head_kwargs: torch.Tensor | None) -> torch.Tensor:
        # Workers hand off uint8 tokens (A0 C1 G2 T3, N/pad>=4); one-hot encoding happens
        # here, on the GPU. A float input is already one-hot (attribution/design/predict
        # passes construct it directly) and passes through unchanged.
        if input_ids.dtype == torch.uint8:
            input_ids = one_hot_sequence_tokens(input_ids)
        features = self.backbone.forward_features(input_ids)
        # Match the head's parameter dtype rather than hardcoding float32: under
        # autocast training the head's master weights are float32 (so this is a
        # no-op vs. .float()), but a model loaded at bf16/fp16 for inference
        # keeps features and head weights on the same dtype.
        head_dtype = next(self.head.parameters(), features).dtype
        return self.head(features.to(head_dtype), **head_kwargs)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> RegulonadoModel:
        """Set train/eval mode, keeping BatchNorm in frozen backbone layers in eval mode.

        ``requires_grad=False`` stops weight updates but not BatchNorm's running-stat
        updates, and in train mode BatchNorm also normalises with per-batch statistics
        rather than the pretrained ones. Frozen layers therefore behave as pretrained
        only in eval mode. BatchNorm layers whose affine parameters are trainable keep
        normal train-mode behaviour.
        """
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self

    def apply_freeze_policy(self, policy: FreezePolicy) -> None:
        if policy.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        blocks = list(self.backbone.iter_named_blocks())
        if policy.unfreeze_backbone_stages_from_output_end > 0:
            for _, module in blocks[-policy.unfreeze_backbone_stages_from_output_end :]:
                for p in module.parameters():
                    p.requires_grad = True

        if policy.unfreeze_module_names:
            allowed = set(policy.unfreeze_module_names)
            for name, module in self.backbone.named_modules():
                if name in allowed:
                    for p in module.parameters():
                        p.requires_grad = True


def _build_head(config: RegulonadoConfig) -> nn.Module:
    from regulonado.model.heads import CompositeTrackGroupHead, build_transfer_learning_head

    shared_track_index = config.condition_shared_track_index or None
    head_kwargs: dict = {
        "in_ch": config.feature_dim,
        "hidden": config.head_hidden,
        "n_tracks": config.n_tracks,
        "use_track_metadata": config.use_track_metadata,
        "num_conditions": config.num_conditions if config.use_track_metadata else 0,
        "num_cell_lines": config.num_cell_lines if config.use_track_metadata else 0,
        "num_assay_types": config.num_assay_types if config.use_track_metadata else 0,
        "num_targets": config.num_targets if config.use_track_metadata else 0,
        "metadata_hidden": config.metadata_hidden,
        "condition_shared_track_index": shared_track_index if config.use_track_metadata else None,
        "dropout": config.head_dropout,
    }
    if config.head_type == "residual_film":
        head_kwargs["refinement_kernel"] = config.refinement_kernel
    if config.head_type == "transfer_mlp" and config.mlp_hidden is not None:
        head_kwargs["mlp_hidden"] = config.mlp_hidden
    head_kwargs["output_bias_init"] = config.output_bias_init
    head_kwargs["zero_output_weights"] = config.zero_output_weights
    track_head = build_transfer_learning_head(
        head_type=config.head_type,
        activation_type=config.activation_type,
        **head_kwargs,
    )
    if config.group_contrast_n_groups <= 0:
        return track_head

    # Strictly opt-in: only reached when group_contrast_n_groups > 0. The group head
    # always uses activation_type="identity" regardless of the per-track head's
    # activation_type -- its target is a signed log2 contrast, not a rate.
    group_head = build_transfer_learning_head(
        head_type="group_contrast",
        activation_type="identity",
        in_ch=config.feature_dim,
        hidden=config.group_contrast_hidden,
        n_groups=config.group_contrast_n_groups,
        mlp_hidden=config.group_contrast_mlp_hidden,
        dropout=config.group_contrast_dropout,
    )
    return CompositeTrackGroupHead(
        track_head=track_head,
        group_head=group_head,
        track_channel_count=config.n_tracks,
    )
