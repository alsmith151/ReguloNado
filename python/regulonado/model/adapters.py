from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from enformer_pytorch import Enformer
from enformer_pytorch.config_enformer import EnformerConfig

BackboneType = Literal["borzoi", "enformer"]


@dataclass(slots=True)
class BackboneSpec:
    backbone_type: BackboneType
    pretrained_name: str | None = None
    feature_dim: int | None = None
    target_length: int | None = None
    config_overrides: dict[str, Any] | None = None
    # Opt-in guard: building a backbone without pretrained weights yields a randomly
    # initialised network. That is almost never intended for transfer learning, so it
    # must be requested explicitly rather than happening as a silent fallback.
    allow_random_init: bool = False


def _require_pretrained_or_explicit_random(spec: BackboneSpec, example: str) -> None:
    """Guard against silently training on a randomly initialised backbone.

    Raises when no ``pretrained_name`` is set and random initialisation was not
    explicitly opted into via ``allow_random_init``.
    """
    if not spec.allow_random_init:
        raise ValueError(
            f"Backbone {spec.backbone_type!r} has no 'pretrained_name', which would "
            "silently train on a randomly initialised backbone (no pretrained motif "
            "representations). Set backbone.pretrained_name to a checkpoint "
            f"(e.g. {example!r}), or set backbone.allow_random_init=true to "
            "deliberately train from scratch."
        )


class BaseBackboneAdapter(nn.Module):
    feature_dim: int

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        """Yield ordered trainable backbone stages from early layers toward the output side.

        Freeze-policy settings that unfreeze stages "from the end" operate on this ordered
        sequence. For example, Borzoi returns transformer blocks followed by
        ``final_joined_convs``, and Enformer returns transformer blocks followed by
        ``final_pointwise``.
        """
        raise NotImplementedError

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BorzoiBackboneAdapter(BaseBackboneAdapter):
    def __init__(self, model: Borzoi):
        super().__init__()
        self.model = model
        self.feature_dim = 1920

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        autocast_dtype = torch.get_autocast_dtype("cuda") if input_ids.is_cuda else torch.float32
        with torch.amp.autocast(
            device_type="cuda",
            enabled=input_ids.is_cuda,
            dtype=autocast_dtype,
        ):
            features = self.model.get_embs_after_crop(input_ids)
            features = self.model.final_joined_convs(features)
        return features

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        if hasattr(self.model, "transformer") and isinstance(
            self.model.transformer, (nn.ModuleList, nn.Sequential)
        ):
            for index, block in enumerate(self.model.transformer):
                yield f"transformer.{index}", block
        if hasattr(self.model, "final_joined_convs"):
            yield "final_joined_convs", self.model.final_joined_convs

    @classmethod
    def from_spec(cls, spec: BackboneSpec) -> "BorzoiBackboneAdapter":
        if spec.pretrained_name:
            model = Borzoi.from_pretrained(spec.pretrained_name)
            return cls(model)

        _require_pretrained_or_explicit_random(spec, example="johahi/borzoi-replicate-0")
        overrides = dict(spec.config_overrides or {})
        config = BorzoiConfig(**overrides)
        return cls(Borzoi(config=config))


class EnformerBackboneAdapter(BaseBackboneAdapter):
    def __init__(self, model: Enformer):
        super().__init__()
        self.model = model
        self.feature_dim = int(model.config.dim)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (batch, channels, length), got {tuple(input_ids.shape)}"
            )
        sequence_major = input_ids.transpose(1, 2)
        features = self.model(sequence_major, return_only_embeddings=True)
        if features.ndim != 3:
            raise ValueError(
                f"Expected Enformer embeddings to be rank-3, got {tuple(features.shape)}"
            )
        return features.transpose(1, 2)

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        if hasattr(self.model, "transformer") and isinstance(
            self.model.transformer, (nn.ModuleList, nn.Sequential)
        ):
            for index, block in enumerate(self.model.transformer):
                yield f"transformer.{index}", block
        if hasattr(self.model, "final_pointwise"):
            yield "final_pointwise", self.model.final_pointwise

    @classmethod
    def from_spec(cls, spec: BackboneSpec) -> "EnformerBackboneAdapter":
        if spec.pretrained_name:
            model = Enformer.from_pretrained(spec.pretrained_name)
            return cls(model)

        _require_pretrained_or_explicit_random(spec, example="EleutherAI/enformer-official-rough")
        overrides = dict(spec.config_overrides or {})
        if spec.target_length is not None:
            overrides.setdefault("target_length", spec.target_length)
        config = EnformerConfig(**overrides)
        return cls(Enformer(config))


def build_backbone_adapter(spec: BackboneSpec) -> BaseBackboneAdapter:
    if spec.backbone_type == "borzoi":
        return BorzoiBackboneAdapter.from_spec(spec)
    if spec.backbone_type == "enformer":
        return EnformerBackboneAdapter.from_spec(spec)
    raise ValueError(f"Unsupported backbone type {spec.backbone_type!r}")
