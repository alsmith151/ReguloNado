from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import torch
import torch.nn as nn


class BackboneAdapter(Protocol):
    feature_dim: int

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]: ...


@dataclass(slots=True)
class FreezePolicy:
    freeze_backbone: bool = True
    unfreeze_backbone_stages_from_output_end: int = 0
    unfreeze_module_names: tuple[str, ...] = field(default_factory=tuple)


class HeadedSequenceModel(nn.Module):
    def __init__(self, backbone: BackboneAdapter, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor, **head_kwargs: torch.Tensor | None) -> torch.Tensor:
        features = self.backbone.forward_features(input_ids)
        return self.head(features.float(), **head_kwargs)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def apply_freeze_policy(self, policy: FreezePolicy) -> None:
        if policy.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        blocks = list(self.backbone.iter_named_blocks())
        if policy.unfreeze_backbone_stages_from_output_end > 0:
            for _, module in blocks[-policy.unfreeze_backbone_stages_from_output_end :]:
                for parameter in module.parameters():
                    parameter.requires_grad = True

        if policy.unfreeze_module_names:
            allowed = set(policy.unfreeze_module_names)
            for name, module in self.backbone.named_modules():
                if name in allowed:
                    for parameter in module.parameters():
                        parameter.requires_grad = True