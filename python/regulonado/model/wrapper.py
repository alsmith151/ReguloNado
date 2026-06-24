from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from regulonado.model.config import RegulonadoConfig

if TYPE_CHECKING:
    pass


class BackboneAdapter(Iterable):
    """Protocol satisfied by BorzoiBackboneAdapter and EnformerBackboneAdapter."""

    feature_dim: int

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]: ...


@dataclass(slots=True)
class FreezePolicy:
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
                config.backbone_type, config.config_overrides, config.target_length
            )
        if head is not None:
            self.head = head
        else:
            self.head = _build_head(config)

    def forward(self, input_ids: torch.Tensor, **head_kwargs: torch.Tensor | None) -> torch.Tensor:
        features = self.backbone.forward_features(input_ids)
        return self.head(features.float(), **head_kwargs)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def apply_freeze_policy(self, policy: FreezePolicy) -> None:
        if policy.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        blocks = list(self.backbone.iter_named_blocks())
        if policy.unfreeze_backbone_stages_from_output_end > 0:
            for _, module in blocks[-policy.unfreeze_backbone_stages_from_output_end:]:
                for p in module.parameters():
                    p.requires_grad = True

        if policy.unfreeze_module_names:
            allowed = set(policy.unfreeze_module_names)
            for name, module in self.backbone.named_modules():
                if name in allowed:
                    for p in module.parameters():
                        p.requires_grad = True


def _build_head(config: RegulonadoConfig) -> nn.Module:
    from regulonado.model.heads import build_perturb_head

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
    return build_perturb_head(
        head_type=config.head_type,
        activation_type=config.activation_type,
        **head_kwargs,
    )
