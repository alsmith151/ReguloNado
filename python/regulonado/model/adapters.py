from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi as _Borzoi
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



class Borzoi(_Borzoi):
    """Thin subclass that adapts upstream Borzoi to transformers v5 weight loading.

    transformers v5 requires post_init() to mark modules with _is_hf_initialized so
    that _initialize_missing_keys() does not re-initialize weights after from_pretrained
    loads the checkpoint.  The upstream Borzoi.__init__ omits this call.

    Additionally, transformers >=5.12 only honours the per-parameter
    ``_is_hf_initialized`` flag (set on each tensor by the checkpoint loader) inside
    ``_initialize_weights`` when ``is_remote_code=True``.  borzoi_pytorch is an
    installed package, not Hub remote code, so that flag is otherwise ignored and
    ``_init_weights`` re-runs ``xavier_normal_`` over already-loaded modules — silently
    clobbering the pretrained BatchNorm affine params and conv/head biases (~100
    tensors) and producing NaN activations at the first transformer block.  We force
    the remote-code code path so loaded weights are preserved while genuinely missing
    keys (whose tensors lack the flag) are still initialized normally.
    """

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def _initialize_weights(self, module, is_remote_code: bool = False):
        return super()._initialize_weights(module, is_remote_code=True)


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
        # FlashZoi's flash_attn kernels require uniform half precision and CANNOT run
        # under torch.autocast: flash_attn builds its rotary cos/sin cache with
        # torch.outer *inside* the autocast region, which corrupts it to NaN on every
        # torch/flash_attn version tested (torch 2.6-2.8, flash 2.7-2.8). So we keep the
        # flash backbone in bf16 and run it with autocast disabled (see forward_features).
        # Casting here — at build time, before the optimizer is created and before any
        # DDP wrap — keeps param dtypes consistent for the optimizer and DDP reducer.
        # The prediction head stays fp32 (the wrapper feeds it features.float()).
        if getattr(model, "flashed", False):
            self.model = self.model.to(torch.bfloat16)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.model.parameters()).dtype
        if input_ids.is_cuda and param_dtype in (torch.bfloat16, torch.float16):
            # Half-precision flash backbone: run with autocast explicitly disabled so the
            # rotary cache is computed in fp32 (finite) rather than under autocast (NaN).
            with torch.autocast(device_type="cuda", enabled=False):
                features = self.model.get_embs_after_crop(input_ids.to(param_dtype))
                features = self.model.final_joined_convs(features)
            return features
        # Non-flash / CPU path: plain fp32, no autocast (Borzoi's own attention is
        # numerically stable in fp32; autocast fp16 here overflows the conv tower).
        features = self.model.get_embs_after_crop(input_ids.to(param_dtype))
        return self.model.final_joined_convs(features)

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


def build_backbone_architecture(
    backbone_type: str,
    config_overrides: dict[str, Any],
    target_length: int | None,
    pretrained_name: str | None = None,
) -> BaseBackboneAdapter:
    """Build backbone architecture with random weights — no pretrained download.

    Used by RegulonadoModel.__init__ so that from_pretrained can reconstruct the exact
    architecture before loading merged weights from the checkpoint.
    """
    if backbone_type == "borzoi":
        overrides = dict(config_overrides or {})
        if not overrides and pretrained_name:
            # Legacy checkpoints may not persist backbone config_overrides in config.json.
            # Fetching the pretrained config keeps HF from_pretrained architecture-compatible
            # (e.g. flashed=True for flashzoi) without relying on custom load paths.
            overrides = BorzoiConfig.from_pretrained(pretrained_name).to_dict()
        borzoi_config = BorzoiConfig(**overrides)
        return BorzoiBackboneAdapter(Borzoi(config=borzoi_config))
    if backbone_type == "enformer":
        overrides = dict(config_overrides or {})
        if not overrides and pretrained_name:
            overrides = EnformerConfig.from_pretrained(pretrained_name).to_dict()
        if target_length is not None:
            overrides.setdefault("target_length", target_length)
        enformer_config = EnformerConfig(**overrides)
        return EnformerBackboneAdapter(Enformer(enformer_config))
    raise ValueError(f"Unsupported backbone type {backbone_type!r}")
