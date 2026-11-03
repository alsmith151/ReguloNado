import pytest
import torch
from regulonado.design import encoder_diffusion as module
from regulonado.design.diffusion import (
    Denoiser,
    DiffusionConfig,
    load_checkpoint,
    sample,
    train_diffusion,
)
from torch import nn
from torch.nn import functional as F


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(4, 768, 1)

    def forward(self, sequence):
        base = self.embed(sequence.transpose(1, 2))
        skips = {}
        for scale, channels in module.EncoderDenoiser.channels.items():
            value = F.avg_pool1d(base, scale) if scale > 1 else base
            skips[f"bin_size_{scale}"] = F.pad(value, (0, 0, 0, channels - 768))
        return F.avg_pool1d(skips["bin_size_64"], 2), skips


def test_frozen_encoder_trains_only_decoder_and_roundtrips(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        module, "load_encoder", lambda path: ToyEncoder().eval().requires_grad_(False)
    )
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    (encoder / "model.safetensors").write_bytes(b"test weights identity")
    config = DiffusionConfig(
        architecture="alphagenome_encoder",
        encoder_checkpoint=str(encoder),
        activity_columns=["HL60"],
        max_length=16,
        width=16,
        layers=2,
        heads=2,
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    model = Denoiser(config)
    model.train()
    assert not model.network.backbone.training
    assert all(not parameter.requires_grad for parameter in model.network.backbone.parameters())
    values = model(
        torch.full((2, 12), 4), torch.ones(2), torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool)
    )
    assert values.shape == (2, 12, 4)
    values.sum().backward()
    assert model.network.output.weight.grad is not None
    assert model.network.backbone.embed.weight.grad is None
    table = tmp_path / "training.tsv"
    pd.DataFrame(
        dict(
            chrom=["chr1", "chr2", "chr8"],
            sequence=["AAAACCCC", "ACGTACGA", "TGCATTGC"],
            HL60=[1.0, 2.0, 3.0],
        )
    ).to_csv(table, sep="\t", index=False)
    path = train_diffusion(table, tmp_path / "model", config)
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["format_version"] == 2
    assert not any(key.startswith("network.backbone.") for key in saved["model"])
    loaded, metadata = load_checkpoint(path, "cpu")
    assert metadata["config"]["encoder_sha256"]
    assert len(sample(loaded, [1.0], [True], mode="de_novo", length=8, count=2, steps=2)) == 2
    (encoder / "model.safetensors").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(path, "cpu")


def test_frozen_convolution_conversion_preserves_outputs():
    ag = pytest.importorskip("alphagenome_pytorch.convolutions")
    original = ag.StandardizedConv1d(4, 8, 5).eval()
    container = nn.Sequential(original)
    x = torch.randn(2, 4, 32)
    expected = container(x).detach()
    module.freeze_standardized_convolutions(container)
    assert isinstance(container[0], nn.Conv1d)
    torch.testing.assert_close(container(x), expected, atol=1e-6, rtol=1e-5)
