"""Backbone-adapter geometry: ``output_span`` against real feature-map lengths.

Borzoi and Enformer centre-crop their feature map to a fixed number of bins; these
tests build small random-init versions of the real modules (few transformer layers,
few bins) so the check runs fast on CPU, and confirm ``output_span`` predicts both the
feature map's length and where it starts. AlphaGenome has no crop (offset is always 0)
but does need a minimum input length for its internal pairwise attention, so its
geometry test also uses the real module, at the smallest input that works.
"""

from __future__ import annotations

import os

import pytest
import torch
from regulonado.model.adapters import (
    AlphaGenomeBackboneAdapter,
    BorzoiBackboneAdapter,
    EnformerBackboneAdapter,
)


def test_borzoi_output_span_matches_real_feature_map():
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    torch.manual_seed(0)
    config = BorzoiConfig(
        depth=1, dim=32, heads=2, attn_dim_key=8, attn_dim_value=8, bins_to_return=32
    )
    adapter = BorzoiBackboneAdapter(Borzoi(config))

    assert adapter.output_bin_size == 32
    assert adapter.input_multiple == 32
    assert adapter.fixed_input_length == 524_288

    input_length = 3072  # 96 bins of 32 bp; 96 - 32 = 64, evenly split each side
    offset_bp, n_bins = adapter.output_span(input_length)

    features = adapter.forward_features(torch.randn(1, 4, input_length))
    assert features.shape == (1, adapter.feature_dim, n_bins)
    assert n_bins == 32
    # Symmetric centre crop: the same margin is discarded from each end.
    assert offset_bp == (input_length - n_bins * adapter.output_bin_size) // 2 == 1024


def test_borzoi_output_span_offset_scales_with_symmetric_padding():
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    torch.manual_seed(0)
    config = BorzoiConfig(
        depth=1, dim=32, heads=2, attn_dim_key=8, attn_dim_value=8, bins_to_return=32
    )
    adapter = BorzoiBackboneAdapter(Borzoi(config))

    offset_small, n_bins_small = adapter.output_span(3072)
    offset_large, n_bins_large = adapter.output_span(3072 + 2 * 320)  # +10 bins each side

    assert n_bins_large == n_bins_small
    assert offset_large - offset_small == 320


def test_enformer_output_span_matches_real_feature_map():
    from enformer_pytorch import Enformer
    from enformer_pytorch.config_enformer import EnformerConfig

    torch.manual_seed(0)
    config = EnformerConfig(depth=1, target_length=8)
    adapter = EnformerBackboneAdapter(Enformer(config))

    assert adapter.output_bin_size == 128  # 2 ** num_downsamples (default 7)
    assert adapter.input_multiple == 128
    assert adapter.fixed_input_length == 196_608

    input_length = 2048  # 16 bins of 128 bp; 16 - 8 = 8, evenly split each side
    offset_bp, n_bins = adapter.output_span(input_length)

    sequence_major_input = torch.randn(1, 4, input_length)
    features = adapter.forward_features(sequence_major_input)
    assert features.shape == (1, adapter.feature_dim, n_bins)
    assert n_bins == 8
    assert offset_bp == (input_length - n_bins * adapter.output_bin_size) // 2 == 512


def test_enformer_output_span_offset_scales_with_symmetric_padding():
    from enformer_pytorch import Enformer
    from enformer_pytorch.config_enformer import EnformerConfig

    torch.manual_seed(0)
    config = EnformerConfig(depth=1, target_length=8)
    adapter = EnformerBackboneAdapter(Enformer(config))

    offset_small, n_bins_small = adapter.output_span(2048)
    offset_large, n_bins_large = adapter.output_span(2048 + 2 * 256)  # +2 bins each side

    assert n_bins_large == n_bins_small
    assert offset_large - offset_small == 256


def test_alphagenome_output_span_is_uncropped():
    pytest.importorskip("alphagenome_pytorch")
    adapter_cls = AlphaGenomeBackboneAdapter

    assert adapter_cls.__mro__  # sanity: class importable without instantiating a model

    from alphagenome_pytorch import AlphaGenome

    torch.manual_seed(0)
    adapter = AlphaGenomeBackboneAdapter(AlphaGenome())

    assert adapter.feature_dim == 3072
    assert adapter.output_bin_size == 128
    assert adapter.input_multiple == 128
    assert adapter.fixed_input_length is None

    # 2048 bp is the smallest input AlphaGenome's pairwise attention accepts (its pair
    # embeddings pool the trunk by a further 16x, so the trunk needs >= 16 positions).
    input_length = 2048
    offset_bp, n_bins = adapter.output_span(input_length)
    assert (offset_bp, n_bins) == (0, 16)

    features = adapter.forward_features(torch.randn(1, 4, input_length))
    assert features.shape == (1, adapter.feature_dim, n_bins)


def test_alphagenome_output_span_rejects_non_multiple_of_128():
    pytest.importorskip("alphagenome_pytorch")
    from alphagenome_pytorch import AlphaGenome

    adapter = AlphaGenomeBackboneAdapter(AlphaGenome())
    with pytest.raises(ValueError, match="multiple of 128"):
        adapter.output_span(2049)


def test_alphagenome_iter_locon_conv_candidates_is_empty():
    pytest.importorskip("alphagenome_pytorch")
    from alphagenome_pytorch import AlphaGenome

    adapter = AlphaGenomeBackboneAdapter(AlphaGenome())
    assert list(adapter.iter_locon_conv_candidates()) == []


def test_alphagenome_iter_named_blocks_yields_top_level_modules():
    pytest.importorskip("alphagenome_pytorch")
    from alphagenome_pytorch import AlphaGenome

    adapter = AlphaGenomeBackboneAdapter(AlphaGenome())
    names = [name for name, _ in adapter.iter_named_blocks()]
    assert names == [name for name, _ in adapter.model.named_children()]
    assert "tower" in names


def _accelerator_available() -> bool:
    if torch.cuda.is_available():
        return True
    # MPS has no kernel for one op AlphaGenome's rotary embeddings use, so it only
    # works there when the process was started with PYTORCH_ENABLE_MPS_FALLBACK=1
    # (that flag is latched at process start, so it can't be set from inside the test).
    return torch.backends.mps.is_available() and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "1"


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="AlphaGenome smoke test needs CUDA, or MPS with PYTORCH_ENABLE_MPS_FALLBACK=1",
)
def test_alphagenome_smoke_on_accelerator():
    """End-to-end pretrained-weight smoke test, skipped when weights aren't cached.

    Downloading the ~900 MB ``all_folds`` checkpoint here would make CI flaky and slow,
    so this only runs when it's already present in the local Hugging Face cache
    (``local_files_only=True``); otherwise it skips rather than hitting the network.
    """
    pytest.importorskip("alphagenome_pytorch")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from regulonado.model.adapters import ALPHAGENOME_HUB_REPO_ID

    try:
        weights_path = huggingface_hub.hf_hub_download(
            repo_id=ALPHAGENOME_HUB_REPO_ID,
            filename="model_all_folds.safetensors",
            local_files_only=True,
        )
    except Exception:
        pytest.skip("AlphaGenome pretrained weights are not cached locally")

    from alphagenome_pytorch import AlphaGenome

    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AlphaGenome.from_pretrained(weights_path, device=device)
    adapter = AlphaGenomeBackboneAdapter(model).to(device)

    input_length = 131_072
    x = torch.randn(1, 4, input_length, device=device)
    with torch.no_grad():
        features = adapter.forward_features(x)

    offset_bp, n_bins = adapter.output_span(input_length)
    assert (offset_bp, n_bins) == (0, input_length // 128)
    assert features.shape == (1, adapter.feature_dim, n_bins)
