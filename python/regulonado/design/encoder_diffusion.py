"""Masked DNA reconstruction on the user's frozen, trained AlphaGenome encoder."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=8)
def _digest(path, size, mtime):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encoder_digest(checkpoint):
    path = Path(checkpoint).resolve() / "model.safetensors"
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def freeze_standardized_convolutions(module):
    """Precompute frozen weight standardisation; preserve odd-kernel SAME padding."""
    from alphagenome_pytorch.convolutions import StandardizedConv1d

    for name, child in list(module.named_children()):
        if isinstance(child, StandardizedConv1d):
            if child.kernel_size[0] % 2 == 0 or child.stride != (1,) or child.dilation != (1,):
                raise ValueError(
                    "Frozen convolution conversion requires odd stride-one undilated kernels"
                )
            replacement = nn.Conv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                padding="same" if child.pad_mode == "same" else 0,
                groups=child.groups,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                weight = child.weight.float()
                mean = weight.mean((1, 2), keepdim=True)
                variance = weight.var((1, 2), keepdim=True, unbiased=False)
                factor = (
                    torch.rsqrt(
                        torch.maximum(
                            variance * (child.in_channels * child.kernel_size[0]),
                            torch.tensor(1e-4, device=weight.device),
                        )
                    )
                    * child.scale
                )
                replacement.weight.copy_((weight - mean) * factor)
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
            replacement.requires_grad_(False)
            setattr(module, name, replacement)
        else:
            freeze_standardized_convolutions(child)


def load_encoder(checkpoint):
    from regulonado.training.regions.inference import load_region_count_model

    oracle = load_region_count_model(checkpoint, device="cpu")
    if oracle.config.trunk["backbone"].get("features") != "encoder":
        raise ValueError("Encoder diffusion requires an AlphaGenome encoder checkpoint")
    backbone = oracle.trunk.adapter.model
    backbone.requires_grad_(False)
    backbone.eval()
    freeze_standardized_convolutions(backbone)
    return backbone


class ConditionalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(width, 2 * width)
        self.conv = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation)
        self.out = nn.Conv1d(width, width, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden, condition):
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        value = self.norm(hidden.transpose(1, 2)) * (1 + scale[:, None]) + shift[:, None]
        return hidden + self.out(F.silu(self.conv(value.transpose(1, 2))))


class EncoderDenoiser(nn.Module):
    """Frozen multi-resolution features plus a conditional feature-pyramid decoder."""

    channels = {1: 768, 2: 896, 4: 1024, 8: 1152, 16: 1280, 32: 1408, 64: 1536}

    def __init__(self, config):
        super().__init__()
        self.config = config
        if (
            config.encoder_sha256
            and encoder_digest(config.encoder_checkpoint) != config.encoder_sha256
        ):
            raise ValueError("The encoder checkpoint does not match the recorded SHA256")
        self.backbone = load_encoder(config.encoder_checkpoint)
        self.projections = nn.ModuleDict(
            {
                str(scale): nn.Conv1d(channels, config.width, 1)
                for scale, channels in self.channels.items()
            }
        )
        self.tokens = nn.Embedding(7, config.width, padding_idx=5)
        self.positions = nn.Embedding(config.max_length, config.width)
        self.condition = nn.Sequential(
            nn.Linear(2 * len(config.activity_columns) + 1, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.blocks = nn.ModuleList(
            [ConditionalBlock(config.width, 2**i) for i in range(config.layers)]
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output = nn.Conv1d(config.width, 4, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, tokens, time, values, observed):
        length = tokens.shape[1]
        if length > self.config.max_length:
            raise ValueError("Sequence exceeds denoiser max_length")
        padded = math.ceil(length / 128) * 128
        one_hot = F.one_hot(tokens.clamp_max(3), 4).float() * tokens.lt(4)[..., None]
        one_hot = F.pad(one_hot, (0, 0, 0, padded - length))
        # no_grad rather than inference_mode: decoder weight gradients save these features.
        dtype = getattr(torch, self.config.encoder_compute_dtype)
        mixed = self.config.encoder_compute_dtype != "float32" and one_hot.device.type in (
            "mps",
            "cuda",
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=one_hot.device.type,
                dtype=dtype if mixed else torch.float16,
                enabled=mixed,
            ),
        ):
            _, skips = self.backbone(one_hot)
        hidden = None
        for scale, project in self.projections.items():
            feature = project(skips[f"bin_size_{scale}"].float())
            feature = F.interpolate(feature, size=padded, mode="linear", align_corners=False)[
                ..., :length
            ]
            hidden = feature if hidden is None else hidden + feature
        hidden = hidden / math.sqrt(len(self.projections))
        hidden = hidden + self.tokens(tokens).transpose(1, 2)
        hidden = hidden + self.positions(torch.arange(length, device=tokens.device)).T[None]
        numeric = torch.cat(
            [values.masked_fill(~observed, 0), observed.float(), time[:, None]], dim=-1
        )
        condition = self.condition(numeric)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = self.output_norm(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output(hidden).transpose(1, 2)
