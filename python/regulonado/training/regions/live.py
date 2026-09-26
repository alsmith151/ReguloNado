"""Live-trunk region training: the backbone runs every step on each region's own window.

The cached path (:mod:`regulonado.embeddings`) runs a frozen trunk once and trains the
head on stored features. Here the trunk runs on every batch, which is what lets it be
fine-tuned -- fully, its last few blocks, or through low-rank adapters -- and makes short
inputs practical: AlphaGenome's CNN encoder (``backbone.features: encoder``) on a few kb
per region, or the full trunk on a shortened context.

Geometry matches the cache exactly: a region's ``K`` bins start at the bin holding its
``target_start`` (:func:`region_bin_layout`), so a head trained on cached embeddings
warm-starts a live run bin-for-bin. Each window is laid out so those bins sit in the middle
of the backbone's output, with equal context either side.

Fine-tuning (:class:`TrunkFinetuneConfig`, the ``trunk:`` section of
``train_regions.yaml``):

- ``frozen``: no trunk parameter trains; only the head does.
- ``full``: every trunk parameter trains (at ``trainer.backbone_learning_rate``).
- ``last_blocks``: the last ``unfreeze_last`` blocks of the adapter's
  :meth:`~regulonado.model.adapters.BaseBackboneAdapter.iter_named_blocks` train.
- ``adapters`` (AlphaGenome): low-rank adapters on a frozen trunk. ``lora``/``ia3``/
  ``houlsby`` come from ``alphagenome-pytorch``'s fine-tuning extension
  (``prepare_for_transfer``) and adapt the transformer tower; ``locon`` adapts
  convolutions -- the only option for ``features: encoder`` -- via :class:`SameLocon` here,
  since the port's ``Locon`` does not reproduce ``StandardizedConv1d``'s manual "same"
  padding and changes the output length.

The trunk always runs in eval mode (running-statistics normalisation, no dropout), trained
or not, so its features match what the cache stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from regulonado.counts.dataset import RegionCountData
from regulonado.model.adapters import BaseBackboneAdapter
from regulonado.sequence import Genome, fetch_window, reverse_complement_onehot

__all__ = [
    "FINETUNE_MODES",
    "LiveTrunk",
    "SameLocon",
    "SequenceRegionDataset",
    "TrunkFinetuneConfig",
    "TrunkWindow",
    "prepare_trunk",
    "region_bin_layout",
]

#: ``trunk.finetune`` choices.
FINETUNE_MODES = ("frozen", "full", "last_blocks", "adapters")
#: ``trunk.adapters`` choices (AlphaGenome only).
ADAPTER_KINDS = ("lora", "ia3", "houlsby", "locon")


def region_bin_layout(target_width: int, bin_size: int, pool_to: int | None) -> tuple[int, int]:
    """``(k, pool_factor)`` for a region of *target_width* bp: the same ``K`` pooled bins
    of ``pool_to`` bp (default: the backbone's own *bin_size*) the embedding cache stores."""
    if pool_to is not None and (pool_to % bin_size != 0 or pool_to < bin_size):
        raise ValueError(f"pool_to {pool_to} must be a positive multiple of {bin_size}")
    effective = pool_to or bin_size
    return math.ceil(target_width / effective) + 1, effective // bin_size


@dataclass(frozen=True)
class TrunkWindow:
    """Where each region's window sits and which output bins are its ``K`` bins.

    ``input_length`` bp go in; the backbone's output covers ``[output_offset_bp,
    output_offset_bp + n_output_bins * bin_size)`` of it; the region's ``k * pool_factor``
    raw bins start ``first_bin`` bins into that output.
    """

    input_length: int
    bin_size: int
    pool_factor: int
    k: int
    output_offset_bp: int
    n_output_bins: int
    first_bin: int

    @classmethod
    def for_adapter(
        cls,
        adapter: BaseBackboneAdapter,
        input_length: int,
        target_width: int,
        pool_to: int | None = None,
    ) -> "TrunkWindow":
        """Lay a region's bins out centrally in *adapter*'s output for *input_length* bp.

        Raises:
            ValueError: if *input_length* is not an input the adapter accepts, or leaves no
                room for the region's bins.
        """
        fixed = adapter.fixed_input_length
        if fixed is not None and input_length != fixed:
            raise ValueError(
                f"data.input_length must be {fixed} for this backbone (fixed input length)"
            )
        if input_length % adapter.input_multiple != 0:
            raise ValueError(
                f"data.input_length {input_length} is not a multiple of {adapter.input_multiple}"
            )
        k, pool_factor = region_bin_layout(target_width, adapter.output_bin_size, pool_to)
        offset_bp, n_bins = adapter.output_span(input_length)
        raw_bins = k * pool_factor
        if raw_bins > n_bins:
            raise ValueError(
                f"data.input_length {input_length} gives {n_bins} output bins; the region "
                f"needs {raw_bins} ({k} x {pool_factor}) -- use a longer input"
            )
        return cls(
            input_length=input_length,
            bin_size=adapter.output_bin_size,
            pool_factor=pool_factor,
            k=k,
            output_offset_bp=offset_bp,
            n_output_bins=n_bins,
            first_bin=(n_bins - raw_bins) // 2,
        )

    def window_start(self, target_start: int) -> int:
        """Genomic start of the window for a region whose target starts at *target_start*."""
        effective = self.bin_size * self.pool_factor
        first_raw_bin_bp = (target_start // effective) * effective
        return first_raw_bin_bp - self.output_offset_bp - self.first_bin * self.bin_size

    def as_dict(self) -> dict[str, int]:
        return {
            "input_length": self.input_length,
            "bin_size": self.bin_size,
            "pool_factor": self.pool_factor,
            "k": self.k,
            "output_offset_bp": self.output_offset_bp,
            "n_output_bins": self.n_output_bins,
            "first_bin": self.first_bin,
        }


class LiveTrunk(nn.Module):
    """``[B, 4, L]`` one-hot windows -> ``[B, K, D]`` region features, like the cache's.

    Rows flagged ``rc`` were reverse-complemented by the dataset; their feature maps are
    flipped back along the bin axis before slicing, so every row's bins are in forward
    genomic order (as in the cache's RC pass).
    """

    def __init__(self, adapter: BaseBackboneAdapter, window: TrunkWindow) -> None:
        super().__init__()
        self.adapter = adapter
        self.window = window

    def train(self, mode: bool = True) -> "LiveTrunk":
        # Always eval: running-statistics normalisation and no dropout, whether or not its
        # parameters train, so the features stay those the cache would have stored.
        super().train(False)
        return self

    def forward(self, sequence: Tensor, rc: Tensor | None = None) -> Tensor:
        features = self.adapter.forward_features(sequence)  # [B, D, n_output_bins]
        if rc is not None and bool(rc.any()):
            flip = rc.to(device=features.device, dtype=torch.bool)
            features = torch.where(flip[:, None, None], features.flip(-1), features)
        window = self.window
        raw = features[:, :, window.first_bin : window.first_bin + window.k * window.pool_factor]
        batch, d, _ = raw.shape
        pooled = raw.reshape(batch, d, window.k, window.pool_factor).mean(dim=-1)
        return pooled.transpose(1, 2)  # [B, K, D]


@dataclass
class TrunkFinetuneConfig:
    """The ``trunk:`` section of ``train_regions.yaml`` (live-trunk runs only).

    See the module docstring for ``finetune`` modes. ``adapters`` lists adapter kinds for
    ``finetune: adapters``; the ``lora_*``/``ia3_*``/``houlsby_*`` fields are passed to
    ``alphagenome-pytorch``'s ``TransferConfig``, and ``locon_*`` to :class:`SameLocon`.
    ``unfreeze_norm`` also trains normalisation layers under adapters.
    ``gradient_checkpointing`` recomputes the trunk's blocks in backward instead of storing
    their activations (AlphaGenome), trading compute for memory at long inputs.
    """

    finetune: str = "frozen"
    unfreeze_last: int = 0
    adapters: list[str] = field(default_factory=list)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    ia3_targets: list[str] = field(default_factory=lambda: ["k_proj", "v_proj"])
    ia3_ff_targets: list[str] = field(default_factory=list)
    houlsby_latent_dim: int = 8
    houlsby_placement: str = "block"
    houlsby_targets: list[str] = field(default_factory=lambda: ["mha", "mlp"])
    locon_rank: int = 4
    locon_alpha: int = 1
    locon_targets: list[str] = field(default_factory=lambda: ["down_blocks"])
    unfreeze_norm: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.finetune not in FINETUNE_MODES:
            raise ValueError(
                f"trunk.finetune must be one of {FINETUNE_MODES}; got {self.finetune!r}"
            )
        unknown = [kind for kind in self.adapters if kind not in ADAPTER_KINDS]
        if unknown:
            raise ValueError(f"trunk.adapters: unknown {unknown}; choose from {ADAPTER_KINDS}")
        if self.finetune == "adapters" and not self.adapters:
            raise ValueError("trunk.finetune: adapters needs trunk.adapters, e.g. [lora]")
        if self.finetune != "adapters" and self.adapters:
            raise ValueError("trunk.adapters only applies with trunk.finetune: adapters")
        if self.finetune == "last_blocks" and self.unfreeze_last < 1:
            raise ValueError("trunk.finetune: last_blocks needs trunk.unfreeze_last >= 1")


class SameLocon(nn.Module):
    """LoCon (low-rank adapter for a conv) that keeps a "same"-padded conv's output length.

    ``conv(x) + scale * up(down(pad(x)))``: ``down`` is a rank-``r`` conv with the wrapped
    conv's kernel, padded exactly as the wrapped conv pads itself (AlphaGenome's
    ``StandardizedConv1d`` pads "same" by hand inside ``forward``), and ``up`` a 1x1 conv
    initialised to zero, so the adapted model starts identical to the pretrained one.
    """

    def __init__(self, conv: nn.Conv1d, rank: int = 4, alpha: int = 1) -> None:
        super().__init__()
        if conv.stride[0] != 1 or conv.groups != 1:
            raise ValueError("SameLocon supports stride-1, ungrouped convolutions only")
        self.original_layer = conv
        kernel = conv.kernel_size[0]
        pad_total = conv.dilation[0] * (kernel - 1)
        self.pad = (pad_total // 2, pad_total - pad_total // 2)
        self.down = nn.Conv1d(
            conv.in_channels, rank, kernel, dilation=conv.dilation[0], bias=False
        )
        self.up = nn.Conv1d(rank, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        for parameter in conv.parameters():
            parameter.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.original_layer(x) + self.scale * self.up(self.down(F.pad(x, self.pad)))


def _apply_same_locon(module: nn.Module, targets: list[str], rank: int, alpha: int) -> int:
    """Wrap every "same"-padded ``Conv1d`` whose name contains a target; returns the count."""
    wrapped = 0
    for name, child in list(module.named_modules()):
        if not name or not any(target in name for target in targets):
            continue
        if not isinstance(child, nn.Conv1d) or getattr(child, "pad_mode", None) != "same":
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        if isinstance(parent, SameLocon):
            continue
        setattr(parent, attr, SameLocon(child, rank=rank, alpha=alpha))
        wrapped += 1
    return wrapped


def prepare_trunk(adapter: BaseBackboneAdapter, cfg: TrunkFinetuneConfig) -> dict[str, int]:
    """Freeze/adapt *adapter* in place per *cfg*; returns trainable/total parameter counts.

    Raises:
        ValueError: for adapters on a backbone other than AlphaGenome, or adapters that
            matched no module (e.g. ``lora`` on ``features: encoder``, which has no
            attention -- use ``locon``).
    """
    if cfg.gradient_checkpointing:
        setter = getattr(adapter, "set_gradient_checkpointing", None)
        if setter is None:
            raise ValueError(
                f"trunk.gradient_checkpointing is not available for {type(adapter).__name__}"
            )
        setter(True)

    for parameter in adapter.parameters():
        parameter.requires_grad = cfg.finetune == "full"

    if cfg.finetune == "last_blocks":
        blocks = list(adapter.iter_named_blocks())
        if cfg.unfreeze_last > len(blocks):
            raise ValueError(
                f"trunk.unfreeze_last {cfg.unfreeze_last} exceeds the trunk's {len(blocks)} "
                f"blocks: {[name for name, _ in blocks]}"
            )
        for _name, block in blocks[-cfg.unfreeze_last :]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    if cfg.finetune == "adapters":
        from regulonado.model.adapters import AlphaGenomeBackboneAdapter

        if not isinstance(adapter, AlphaGenomeBackboneAdapter):
            raise ValueError("trunk.finetune: adapters is implemented for AlphaGenome only")
        before = sum(p.numel() for p in adapter.parameters())
        port_kinds = [kind for kind in cfg.adapters if kind != "locon"]
        if port_kinds:
            from alphagenome_pytorch.extensions.finetuning.transfer import (
                TransferConfig,
                prepare_for_transfer,
            )

            prepare_for_transfer(
                adapter.model,
                TransferConfig(
                    mode=port_kinds,
                    lora_rank=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_targets=list(cfg.lora_targets),
                    ia3_targets=list(cfg.ia3_targets),
                    ia3_ff_targets=list(cfg.ia3_ff_targets),
                    houlsby_latent_dim=cfg.houlsby_latent_dim,
                    houlsby_placement=cfg.houlsby_placement,
                    houlsby_targets=list(cfg.houlsby_targets),
                ),
            )
        if "locon" in cfg.adapters:
            if not _apply_same_locon(
                adapter.model, list(cfg.locon_targets), cfg.locon_rank, cfg.locon_alpha
            ):
                raise ValueError(f"trunk.locon_targets {cfg.locon_targets} matched no conv layer")
        if sum(p.numel() for p in adapter.parameters()) == before:
            raise ValueError(
                f"trunk.adapters {cfg.adapters} matched no module of this trunk "
                f"(features: {adapter.features}); the encoder has no attention, so use locon"
            )
        if cfg.unfreeze_norm:
            from alphagenome_pytorch.extensions.finetuning.adapters import unfreeze_norm_layers

            unfreeze_norm_layers(adapter.model)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    total = sum(p.numel() for p in adapter.parameters())
    return {"trainable": trainable, "total": total}


class SequenceRegionDataset(Dataset):
    """One split of a :class:`RegionCountData` as one-hot windows for a :class:`LiveTrunk`.

    Yields ``{"sequence": [4, input_length] float16, "rc": bool, "labels": [n_tracks]}``
    (plus ``"sample_weight"``). In train mode, *enable_rc_aug* reverse-complements half the
    windows and *shift_max* jitters each window by up to that many bp either way (the ``K``
    bins read stay the same output bins, so the target moves by the shift within them).
    """

    def __init__(
        self,
        data: RegionCountData,
        genome: Genome,
        window: TrunkWindow,
        split: str | None = None,
        *,
        train: bool = False,
        sample_weights: np.ndarray | None = None,
        enable_rc_aug: bool = True,
        shift_max: int = 0,
    ) -> None:
        if split is not None:
            data = data.split(split)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)
            if sample_weights.shape != (data.n_regions,):
                raise ValueError(
                    f"sample_weights has shape {sample_weights.shape}; "
                    f"expected ({data.n_regions},) for split {split!r}"
                )
        self.data = data
        self.genome = genome
        self.window = window
        self.train = train
        self.sample_weights = sample_weights
        self.enable_rc_aug = enable_rc_aug
        self.shift_max = int(shift_max)
        self._chroms = data.regions["chrom"].cast(pl.Utf8).to_list()
        self._starts = np.array(
            [window.window_start(int(s)) for s in data.regions["target_start"].to_list()],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return self.data.n_regions

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = int(self._starts[idx])
        rc = False
        if self.train:
            if self.shift_max:
                start += int(torch.randint(-self.shift_max, self.shift_max + 1, ()).item())
            rc = self.enable_rc_aug and bool(torch.rand(()) < 0.5)
        end = start + self.window.input_length
        onehot = fetch_window(self.genome, self._chroms[idx], start, end)
        if rc:
            onehot = reverse_complement_onehot(onehot)
        item: dict[str, Any] = {
            "sequence": torch.from_numpy(onehot).to(torch.float16),
            "rc": torch.tensor(rc),
            "labels": torch.from_numpy(self.data.counts[idx].astype(np.float32, copy=False)),
        }
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]))
        return item
