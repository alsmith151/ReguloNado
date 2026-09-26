from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi as _Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from borzoi_pytorch.pytorch_borzoi_transformer import Attention as BorzoiAttention
from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed
from enformer_pytorch import Enformer
from enformer_pytorch.config_enformer import EnformerConfig
from enformer_pytorch.modeling_enformer import SEQUENCE_LENGTH as ENFORMER_SEQUENCE_LENGTH

BackboneType = Literal["borzoi", "enformer", "alphagenome"]

# AlphaGenome checkpoints are hosted here (not gated), as .safetensors files named
# model_all_folds.safetensors, model_fold_0.safetensors, ..., model_fold_3.safetensors.
# See https://huggingface.co/gtca/alphagenome_pytorch. Weights are distributed under a
# non-commercial licence -- see that repo's LICENSE before using them outside research.
ALPHAGENOME_HUB_REPO_ID = "gtca/alphagenome_pytorch"


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

    ``from_pretrained`` also replaces non-persistent buffers with uninitialised memory
    and relies on ``_init_weights`` to refill them. flash_attn's ``RotaryEmbedding``
    keeps ``inv_freq`` that way (it is not in the checkpoint) and upstream
    ``_init_weights`` ignores it, leaving garbage rotary frequencies — NaN out of
    ``rotary_emb`` in every FlashZoi forward. ``_init_weights`` recomputes them. Non-flash
    Borzoi's relative-position ``Attention.positions`` buffer is lost the same way —
    garbage relative-position logits, NaN on MPS — and is recomputed alongside.
    """

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def _initialize_weights(self, module, _is_custom_code: bool = True):
        return super()._initialize_weights(module, True)

    def _init_weights(self, module):
        super()._init_weights(module)
        if hasattr(module, "_compute_inv_freq") and isinstance(
            getattr(module, "inv_freq", None), torch.Tensor
        ):
            with torch.no_grad():
                module.inv_freq.copy_(module._compute_inv_freq(device=module.inv_freq.device))
            # Drop any cos/sin tables built from the uninitialised frequencies.
            module._seq_len_cached = 0
            module._cos_cached = None
        if isinstance(module, BorzoiAttention):
            with torch.no_grad():
                module.positions.copy_(
                    get_positional_embed(
                        module.positions.shape[0] // 2 + 1,
                        module.num_rel_pos_features,
                        module.positions.device,
                    )
                )


class BaseBackboneAdapter(nn.Module):
    """Base class for backbone adapters.

    Adapters wrap sequence models (Borzoi, Enformer) and expose a unified
    interface for feature extraction and staged unfreezing.

    Attributes
    ----------
    feature_dim : int
        Output feature dimension.
    output_bin_size : int
        Width, in base pairs, of one output bin (e.g. 32 for Borzoi/Flashzoi, 128 for
        Enformer and AlphaGenome).
    fixed_input_length : int | None
        The one input length (bp) this adapter accepts, or ``None`` when any multiple
        of ``input_multiple`` works (AlphaGenome is fully convolutional and flexible;
        Borzoi and Enformer are deployed at one fixed context length).
    input_multiple : int
        Input lengths must be a multiple of this many base pairs. Equal to
        ``output_bin_size`` for every current adapter.
    """

    feature_dim: int
    output_bin_size: int
    fixed_input_length: int | None
    input_multiple: int

    def output_span(self, input_length: int) -> tuple[int, int]:
        """Part of ``input_length`` covered by this adapter's feature map.

        Some backbones (Borzoi, Enformer) centre-crop their feature map to a
        fixed number of bins, discarding an equal margin from each end of the
        input; others (AlphaGenome) keep the full span. This lets the embedding
        cache line a region's target bins up with the right slice of the feature
        map for any backbone, without special-casing each one.

        Parameters
        ----------
        input_length : int
            Input sequence length in base pairs. Must be a multiple of
            ``input_multiple``, and must equal ``fixed_input_length`` when that
            is not ``None``.

        Returns
        -------
        tuple[int, int]
            ``(offset_bp, n_bins)``: ``offset_bp`` is the number of input base
            pairs excluded from the start (and, symmetrically, the end) of the
            input before the feature map begins; ``n_bins`` is the feature
            map's length in bins.
        """
        raise NotImplementedError

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

    def iter_locon_conv_candidates(self) -> Iterable[str]:
        """Conv1d-bearing module names eligible for LoCon, in data-flow order.

        Only implemented for :class:`BorzoiBackboneAdapter`; see there for the
        eight-entry candidate list.
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
        # Borzoi's main path pools by 2x five times (conv_dna + the four strided steps of
        # res_tower) before the U-Net skip that final_joined_convs sits on -- 32 bp/bin,
        # independent of config (depth, dim, ...). Fixed by architecture, not config.
        self.output_bin_size = 32
        self.input_multiple = self.output_bin_size
        # Borzoi/Flashzoi checkpoints are trained and deployed at this context length.
        # The network is fully convolutional and would run at other multiples of 32, but
        # only this length is supported here -- matches RegulonadoConfig.context_length.
        self.fixed_input_length = 524_288
        # Parameters stay float32 (the checkpoint dtype is whatever it was saved in, so
        # cast explicitly). Casting the backbone to bf16 instead makes AdamW update bf16
        # weights directly: bf16's relative resolution is ~0.8%, so any step smaller than
        # ~0.4% of |w| rounds to zero and larger ones are quantised. float32 weights are
        # the master copy; forward_features supplies the bf16 compute via autocast.
        self.model = self.model.float()

    def output_span(self, input_length: int) -> tuple[int, int]:
        n_bins = int(self.model.config.bins_to_return)
        total_bins = input_length // self.output_bin_size
        offset_bins = (total_bins - n_bins) // 2
        return offset_bins * self.output_bin_size, n_bins

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

    def iter_locon_conv_candidates(self) -> Iterable[str]:
        """Conv1d-bearing module names eligible for LoCon, in data-flow order.

        Mirrors Baskerville's reference-implementation ``conv_layers`` list
        (``transfer.py``, ``add_locon``): ``conv_dna``, the five strided
        ``res_tower`` ``ConvBlock``s, the ``unet1`` ``ConvBlock`` immediately
        before attention, and the final ``final_joined_convs`` ``ConvBlock``.
        This exact eight-entry order was confirmed empirically against the
        authors' TensorFlow implementation; do not reorder or extend it here.

        Introspects the live module tree (``hasattr``/``len`` checks) rather
        than assuming these attributes exist, so the method degrades safely —
        yielding a partial or empty sequence — on dummy modules built for
        tests instead of raising.

        Yields
        ------
        str
            Dotted module path, relative to ``self.model``, of each
            Conv1d-bearing block, e.g. ``"res_tower.6"``. Append
            ``".conv_layer"`` to reach the actual ``nn.Conv1d`` — none of
            these eight blocks are ``separable`` ConvBlocks, so
            ``.conv_layer`` is always a single ``nn.Conv1d``, never a
            ``Sequential`` of depthwise/pointwise convs.
        """
        if hasattr(self.model, "conv_dna") and hasattr(self.model.conv_dna, "conv_layer"):
            yield "conv_dna"

        res_tower = getattr(self.model, "res_tower", None)
        if isinstance(res_tower, (nn.Sequential, nn.ModuleList)):
            for index in (0, 2, 4, 6, 8):
                if index < len(res_tower) and hasattr(res_tower[index], "conv_layer"):
                    yield f"res_tower.{index}"

        unet1 = getattr(self.model, "unet1", None)
        if (
            isinstance(unet1, (nn.Sequential, nn.ModuleList))
            and len(unet1) > 1
            and hasattr(unet1[1], "conv_layer")
        ):
            yield "unet1.1"

        final_joined_convs = getattr(self.model, "final_joined_convs", None)
        if (
            isinstance(final_joined_convs, (nn.Sequential, nn.ModuleList))
            and len(final_joined_convs) > 0
            and hasattr(final_joined_convs[0], "conv_layer")
        ):
            yield "final_joined_convs.0"

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
        # enformer_pytorch's final_pointwise block projects the transformer's `dim`-wide
        # trunk up to `2 * dim` right before `return_only_embeddings=True` returns it
        # (see modeling_enformer.py's `twice_dim`) -- confirmed against a live forward
        # pass, which returns (B, L, 2 * dim), not (B, L, dim) as this previously assumed.
        self.feature_dim = int(model.config.dim) * 2
        # Enformer downsamples by 2x `num_downsamples` times (7 by default -> 128 bp/bin);
        # read from config since num_downsamples is a real EnformerConfig override. Falls
        # back to the default for dummy/stub configs (e.g. in tests) that omit it.
        self.output_bin_size = 2 ** int(getattr(model.config, "num_downsamples", 7))
        self.input_multiple = self.output_bin_size
        # Enformer is deployed at this context length by convention (enformer_pytorch's
        # own SEQUENCE_LENGTH constant), matching the crop tests are trained against.
        self.fixed_input_length = ENFORMER_SEQUENCE_LENGTH

    def output_span(self, input_length: int) -> tuple[int, int]:
        n_bins = int(self.model.config.target_length)
        total_bins = input_length // self.output_bin_size
        offset_bins = (total_bins - n_bins) // 2
        return offset_bins * self.output_bin_size, n_bins

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


def _resolve_alphagenome_weights_path(name: str) -> str:
    """Resolve an AlphaGenome checkpoint name to a local weights file path.

    ``name`` may be a path to an existing local ``.pth``/``.safetensors`` file, or one
    of the named folds hosted at ``ALPHAGENOME_HUB_REPO_ID`` on the Hugging Face Hub
    (``"all_folds"``, ``"fold_0"``, ..., ``"fold_3"``), which is downloaded (and cached
    by huggingface_hub) on first use.
    """
    local_path = Path(name)
    if local_path.exists():
        return str(local_path)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=ALPHAGENOME_HUB_REPO_ID, filename=f"model_{name}.safetensors")


class AlphaGenomeBackboneAdapter(BaseBackboneAdapter):
    """Adapter for AlphaGenome (via the ``alphagenome-pytorch`` optional extra).

    Unlike Borzoi/Enformer, AlphaGenome is fully convolutional over its 128 bp output
    resolution with no fixed context length or centre-crop, so ``forward_features``
    accepts any input length that is a multiple of 128 and returns a feature map
    covering the whole input (``output_span`` offset is always 0).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.feature_dim = 3072
        self.output_bin_size = 128
        self.input_multiple = self.output_bin_size
        self.fixed_input_length = None
        # See BorzoiBackboneAdapter: keep float32 master weights, bf16 compute only
        # under autocast (forward_features), so AdamW updates aren't quantised by bf16.
        self.model = self.model.float()

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, 4, L] one-hot -> [B, 3072, L/128] AlphaGenome 128 bp embeddings.

        ``L`` must be a multiple of 128 (``input_multiple``); AlphaGenome's own
        SAME-padded pooling would otherwise silently round the output length up,
        breaking the 1:1 bin alignment the embedding cache relies on.
        """
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (batch, channels, length), got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[-1] % self.input_multiple != 0:
            raise ValueError(
                f"AlphaGenome input length {input_ids.shape[-1]} is not a multiple of "
                f"{self.input_multiple}"
            )
        param_dtype = next(self.model.parameters()).dtype
        sequence_major = input_ids.transpose(1, 2).to(param_dtype)  # NCL -> NLC (B, L, 4)
        organism_index = torch.zeros(
            sequence_major.shape[0], dtype=torch.long, device=sequence_major.device
        )
        if input_ids.is_cuda:
            # Always bf16, as Borzoi's forward_features does -- see there for why not
            # torch.get_autocast_dtype("cuda").
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model.encode(
                    sequence_major, organism_index, resolutions=(128,), channels_last=False
                )
        else:
            # CPU/MPS: float32. MPS has no kernel for one op AlphaGenome's rotary
            # embeddings use (`aten::logspace`), so running on MPS needs the process
            # started with PYTORCH_ENABLE_MPS_FALLBACK=1 (checked once at process
            # start, so it can't be set here) to route just that op to CPU.
            outputs = self.model.encode(
                sequence_major, organism_index, resolutions=(128,), channels_last=False
            )
        return outputs["embeddings_128bp"]

    def output_span(self, input_length: int) -> tuple[int, int]:
        if input_length % self.input_multiple != 0:
            raise ValueError(
                f"AlphaGenome input length {input_length} is not a multiple of "
                f"{self.input_multiple}"
            )
        return 0, input_length // self.output_bin_size

    def iter_named_blocks(self) -> Iterable[tuple[str, nn.Module]]:
        """Top-level trunk modules in data-flow order.

        AlphaGenome's block structure (encoder down-blocks, transformer tower,
        decoder up-blocks, output embedders) doesn't map onto Borzoi/Enformer's
        "transformer blocks then a short output tail" shape, so unlike those
        adapters this yields whole top-level submodules rather than individual
        transformer layers. FreezePolicy's "unfreeze last N" still works against
        this ordering; it just unfreezes whole stages rather than single blocks.
        """
        for name, module in self.model.named_children():
            yield name, module

    def iter_locon_conv_candidates(self) -> Iterable[str]:
        """No LoCon support yet for AlphaGenome; empty by design (see plan)."""
        return iter(())

    @classmethod
    def from_spec(cls, spec: BackboneSpec) -> "AlphaGenomeBackboneAdapter":
        from alphagenome_pytorch import AlphaGenome

        if spec.pretrained_name:
            weights_path = _resolve_alphagenome_weights_path(spec.pretrained_name)
            model = AlphaGenome.from_pretrained(weights_path)
            return cls(model)

        _require_pretrained_or_explicit_random(spec, example="all_folds")
        overrides = dict(spec.config_overrides or {})
        return cls(AlphaGenome(**overrides))


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
    if spec.backbone_type == "alphagenome":
        return AlphaGenomeBackboneAdapter.from_spec(spec)
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
    if backbone_type == "alphagenome":
        from alphagenome_pytorch import AlphaGenome

        return AlphaGenomeBackboneAdapter(AlphaGenome(**dict(config_overrides or {})))
    raise ValueError(f"Unsupported backbone type {backbone_type!r}")
