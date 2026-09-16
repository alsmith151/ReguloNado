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
    """Specification for building a sequence model backbone.

    Specifies which pretrained model to load or whether to initialize randomly.
    Requires either a ``pretrained_name`` or explicit ``allow_random_init=True``
    to prevent silent training on randomly-initialized weights (a common bug).

    Parameters
    ----------
    backbone_type : BackboneType
        Model type: "borzoi" or "enformer".
    pretrained_name : str | None, optional
        HuggingFace model ID (e.g., "johahi/borzoi-replicate-0"). If None,
        random initialization is used only if allow_random_init=True.
    feature_dim : int | None, optional
        Output feature dimension (inferred from model if not provided).
    target_length : int | None, optional
        Prediction length for Enformer-style models (default from dataset).
    config_overrides : dict[str, Any] | None, optional
        Config parameters to override when building from scratch.
    allow_random_init : bool, optional
        If False (default), raises ValueError when pretrained_name is None.
        Set True to deliberately train from scratch.
    """

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

    The flag is passed positionally because its name changed across 5.x releases
    (``is_remote_code`` in 5.12, ``is_custom_code`` by 5.17). Passing it by keyword
    raised ``TypeError`` on 5.17, and falling back to the flagless call re-initialised
    ~100 loaded tensors — NaN at the first flash-attention block.
    """

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def _initialize_weights(self, module, _is_custom_code: bool = True):
        return super()._initialize_weights(module, True)


class BaseBackboneAdapter(nn.Module):
    """Base class for backbone adapters.

    Adapters wrap sequence models (Borzoi, Enformer) and expose a unified
    interface for feature extraction and staged unfreezing.

    Attributes
    ----------
    feature_dim : int
        Output feature dimension.
    """

    feature_dim: int

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        """Yield ordered trainable backbone stages from early layers toward
        the output side.

        Freeze-policy settings that unfreeze stages "from the end" operate on
        this ordered sequence. For example, Borzoi returns transformer blocks,
        then its ``unet1``/``unet0`` upsampling stages, then
        ``final_joined_convs``, and Enformer returns transformer
        blocks followed by ``final_pointwise``.

        Yields
        ------
        tuple[str, nn.Module]
            (module_name, module) pairs in order from early to late layers.
        """
        raise NotImplementedError

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract sequence features.

        Parameters
        ----------
        input_ids : torch.Tensor
            One-hot encoded sequence, shape [batch, channels, length].

        Returns
        -------
        torch.Tensor
            Sequence features, shape [batch, feature_dim, length].
        """
        raise NotImplementedError


class BorzoiBackboneAdapter(BaseBackboneAdapter):
    def __init__(self, model: Borzoi):
        super().__init__()
        self.model = model
        self.feature_dim = 1920
        # Parameters stay float32 (the checkpoint dtype is whatever it was saved in, so
        # cast explicitly). Casting the backbone to bf16 instead makes AdamW update bf16
        # weights directly: bf16's relative resolution is ~0.8%, so any step smaller than
        # ~0.4% of |w| rounds to zero and larger ones are quantised. float32 weights are
        # the master copy; forward_features supplies the bf16 compute via autocast.
        self.model = self.model.float()

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.model.parameters()).dtype
        input_ids = input_ids.to(param_dtype)
        if input_ids.is_cuda:
            # Always bf16, never torch.get_autocast_dtype("cuda"): outside an enclosing
            # autocast region that returns float16, which overflows Borzoi's conv tower.
            # flash_attn needs half-precision q/k/v, which autocast's Linear casts give it;
            # its rotary cache is built in float32 (torch.outer is not an autocast op) and
            # cast to the q dtype. This also overrides an enclosing fp16 autocast.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.model.get_embs_after_crop(input_ids)
                return self.model.final_joined_convs(features)
        # CPU path (non-flash Borzoi only; flash_attn has no CPU kernels): plain float32.
        features = self.model.get_embs_after_crop(input_ids)
        return self.model.final_joined_convs(features)

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        """Stages in data-flow order: transformer blocks, the two U-Net upsampling
        stages (each merging a conv-tower skip connection back in), then
        ``final_joined_convs``. Each U-Net stage groups its upsampling conv, the
        horizontal conv on its skip branch, and the separable conv after the merge.
        """
        if hasattr(self.model, "transformer") and isinstance(
            self.model.transformer, (nn.ModuleList, nn.Sequential)
        ):
            for index, block in enumerate(self.model.transformer):
                yield f"transformer.{index}", block
        for level in (1, 0):
            names = (f"upsampling_unet{level}", f"horizontal_conv{level}", f"separable{level}")
            if all(hasattr(self.model, name) for name in names):
                yield f"unet{level}", nn.ModuleList(getattr(self.model, name) for name in names)
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
    """Build a backbone adapter from specification.

    Instantiates the appropriate adapter (Borzoi or Enformer) based on the
    spec's backbone_type. Loads pretrained weights if specified, or
    initializes randomly if allowed.

    Parameters
    ----------
    spec : BackboneSpec
        Specification including backbone type and pretrained model name.

    Returns
    -------
    BaseBackboneAdapter
        Initialized adapter wrapping the backbone model.

    Raises
    ------
    ValueError
        If backbone_type is not recognized, or if no pretrained model is
        specified and allow_random_init is False.

    Examples
    --------
    >>> spec = BackboneSpec(
    ...     backbone_type="borzoi",
    ...     pretrained_name="johahi/borzoi-replicate-0"
    ... )
    >>> adapter = build_backbone_adapter(spec)  # doctest: +SKIP
    """
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
