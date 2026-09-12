#!/usr/bin/env python
"""Quick check that flash_attn loads and Borzoi/FlashZoi runs a forward pass on GPU."""
import sys

import torch

print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")

import flash_attn  # noqa: E402

print(f"flash_attn {flash_attn.__version__}")

from regulonado.model.adapters import BackboneSpec, build_backbone_adapter  # noqa: E402

pretrained_name = sys.argv[1] if len(sys.argv) > 1 else "johahi/flashzoi-replicate-0"
spec = BackboneSpec(backbone_type="borzoi", pretrained_name=pretrained_name)

print(f"loading {pretrained_name} ...")
adapter = build_backbone_adapter(spec).cuda().eval()

x = torch.randint(0, 4, (1, 524288), device="cuda")
x = torch.nn.functional.one_hot(x, num_classes=4).permute(0, 2, 1).float()

with torch.no_grad():
    out = adapter.forward_features(x)

print(f"forward ok, output shape: {out.shape}")
