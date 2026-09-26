"""Live-trunk region training: window geometry matches the cache, RC flip, fine-tune modes,
AlphaGenome encoder adapters, checkpoint contents, and a CPU end-to-end run.

The geometry tests reuse ``test_embeddings_cache``'s position-decoding stub adapter and
palindromic synthetic genome, so each feature decodes to its absolute genomic bin: a live
trunk's ``[K, D]`` features must equal what ``embed_regions`` cached for the same region,
bin for bin.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from regulonado.counts.dataset import RegionCountData
from regulonado.embeddings.cache import EmbeddingStore, embed_regions
from regulonado.model.adapters import BaseBackboneAdapter
from regulonado.sequence import open_genome
from regulonado.training.regions import live as live_module
from regulonado.training.regions.live import (
    LiveTrunk,
    SequenceRegionDataset,
    TrunkFinetuneConfig,
    TrunkWindow,
    prepare_trunk,
)
from regulonado.training.regions.model import RegionCountConfig, RegionCountModel
from regulonado.training.regions.runner import _load_warm_start, run_training
from test_embeddings_cache import (
    BIN_SIZE,
    _FlexiblePositionAdapter,
    _position_encoded_sequence,
    _regions_frame,
    _tile_row,
    _write_fasta,
)
from test_regions_runner import _base_cfg, _toy_region_data


@pytest.fixture
def genome(tmp_path):
    path = tmp_path / "genome.fa"
    _write_fasta(path, {"chrTile": _position_encoded_sequence(625 * BIN_SIZE, BIN_SIZE)})
    return open_genome(path)


def _region_data(regions: pl.DataFrame) -> RegionCountData:
    n = regions.height
    tracks = pl.DataFrame({"track_name": ["a"], "group": ["g"], "log_size_factor": [0.0]})
    return RegionCountData.from_arrays(
        regions.with_columns(pl.lit("train").alias("split")),
        np.zeros((n, 1), dtype=np.float32),
        ["a"],
        tracks,
    )


def _live_features(dataset, trunk, indices) -> np.ndarray:
    batch = [dataset[i] for i in indices]
    sequence = torch.stack([item["sequence"] for item in batch])
    rc = torch.stack([item["rc"] for item in batch])
    return trunk(sequence.float(), rc).numpy()


@pytest.mark.parametrize("pool_to", [None, 64])
def test_live_trunk_reads_exactly_the_cached_bins(tmp_path, genome, pool_to):
    # Targets at bin-aligned and unaligned offsets, near a tile boundary and mid-tile.
    regions = _regions_frame(
        [_tile_row("chrTile", start) for start in (1000, 2031, 4077, 4100, 9001, 15000)]
    )
    adapter = _FlexiblePositionAdapter(BIN_SIZE)
    out_dir = tmp_path / "emb"
    embed_regions(
        regions, genome, adapter, out_dir, backbone="stub", context=4096, stride=2048,
        pool_to=pool_to,
    )
    cached = EmbeddingStore(out_dir).get_many(np.arange(regions.height)).astype(np.float32)

    window = TrunkWindow.for_adapter(adapter, 2048, 1000, pool_to)
    dataset = SequenceRegionDataset(_region_data(regions), genome, window)
    live = _live_features(dataset, LiveTrunk(adapter, window), range(regions.height))

    assert live.shape == cached.shape
    np.testing.assert_array_equal(live, cached)


def test_reverse_complemented_windows_flip_back_to_forward_bins(genome, monkeypatch):
    regions = _regions_frame([_tile_row("chrTile", start) for start in (1000, 4077)])
    adapter = _FlexiblePositionAdapter(BIN_SIZE)
    window = TrunkWindow.for_adapter(adapter, 2048, 1000)
    data = _region_data(regions)
    forward = _live_features(
        SequenceRegionDataset(data, genome, window), LiveTrunk(adapter, window), [0, 1]
    )

    monkeypatch.setattr(live_module.torch, "rand", lambda *args: torch.tensor(0.0))
    flipped_dataset = SequenceRegionDataset(data, genome, window, train=True)
    assert all(bool(flipped_dataset[i]["rc"]) for i in (0, 1))
    flipped = _live_features(flipped_dataset, LiveTrunk(adapter, window), [0, 1])
    np.testing.assert_array_equal(flipped, forward)


def test_window_must_fit_the_region_and_the_backbone():
    adapter = _FlexiblePositionAdapter(BIN_SIZE)
    with pytest.raises(ValueError, match="use a longer input"):
        TrunkWindow.for_adapter(adapter, 512, 1000)
    with pytest.raises(ValueError, match="not a multiple of 32"):
        TrunkWindow.for_adapter(adapter, 2050, 1000)


# --------------------------------------------------------------------------- #
# Fine-tune modes
# --------------------------------------------------------------------------- #


class _ConvAdapter(BaseBackboneAdapter):
    """A small trainable flexible backbone: two conv blocks, 32 bp bins."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.output_bin_size = BIN_SIZE
        self.input_multiple = BIN_SIZE
        self.fixed_input_length = None
        self.feature_dim = feature_dim
        self.embed = torch.nn.Conv1d(4, 8, BIN_SIZE, stride=BIN_SIZE)
        self.block = torch.nn.Conv1d(8, feature_dim, 3, padding=1)

    def output_span(self, input_length: int) -> tuple[int, int]:
        return 0, input_length // BIN_SIZE

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.block(torch.relu(self.embed(input_ids.float())))

    def iter_named_blocks(self):
        yield "embed", self.embed
        yield "block", self.block


def _trainable(adapter) -> set[str]:
    return {name for name, p in adapter.named_parameters() if p.requires_grad}


def test_finetune_modes_choose_what_trains():
    adapter = _ConvAdapter()
    assert prepare_trunk(adapter, TrunkFinetuneConfig())["trainable"] == 0
    prepare_trunk(adapter, TrunkFinetuneConfig(finetune="full"))
    assert _trainable(adapter) == {name for name, _ in adapter.named_parameters()}
    prepare_trunk(adapter, TrunkFinetuneConfig(finetune="last_blocks", unfreeze_last=1))
    assert _trainable(adapter) == {"block.weight", "block.bias"}
    with pytest.raises(ValueError, match="AlphaGenome only"):
        prepare_trunk(adapter, TrunkFinetuneConfig(finetune="adapters", adapters=["lora"]))
    with pytest.raises(ValueError, match="needs trunk.adapters"):
        TrunkFinetuneConfig(finetune="adapters")
    with pytest.raises(ValueError, match="unknown"):
        TrunkFinetuneConfig(finetune="adapters", adapters=["prefix"])


def test_alphagenome_encoder_locon_starts_identical_and_trains_only_adapters():
    pytest.importorskip("alphagenome_pytorch")
    from alphagenome_pytorch import AlphaGenome
    from regulonado.model.adapters import AlphaGenomeBackboneAdapter

    torch.manual_seed(0)
    adapter = AlphaGenomeBackboneAdapter(AlphaGenome(), features="encoder").eval()
    assert adapter.feature_dim == 1536
    x = torch.zeros(1, 4, 1024)
    x[:, torch.randint(0, 4, (1024,)), torch.arange(1024)] = 1.0
    with torch.no_grad():
        before = adapter.forward_features(x)
    assert before.shape == (1, 1536, 8)

    counts = prepare_trunk(
        adapter,
        TrunkFinetuneConfig(finetune="adapters", adapters=["locon"], gradient_checkpointing=True),
    )
    assert 0 < counts["trainable"] < counts["total"] / 50
    assert all(
        ".down." in name or ".up." in name for name in _trainable(adapter)
    ), sorted(_trainable(adapter))[:5]
    assert adapter.model.gradient_checkpointing
    with torch.no_grad():
        after = adapter.forward_features(x)
    torch.testing.assert_close(after, before)

    with pytest.raises(ValueError, match="use locon"):
        prepare_trunk(
            AlphaGenomeBackboneAdapter(AlphaGenome(), features="encoder"),
            TrunkFinetuneConfig(finetune="adapters", adapters=["lora"]),
        )


# --------------------------------------------------------------------------- #
# Heads and checkpoints
# --------------------------------------------------------------------------- #


def _config(**kwargs) -> RegionCountConfig:
    return RegionCountConfig(
        k=5, d=4, track_groups=[0, 0, 1], log_size_factors=[0.0, 0.1, -0.1], hidden=8, **kwargs
    )


def test_live_model_warm_starts_from_a_cached_head_and_saves_only_what_trains(tmp_path):
    torch.manual_seed(0)
    cached = RegionCountModel(_config())
    cached.save_pretrained(tmp_path / "cached")

    adapter = _ConvAdapter()
    prepare_trunk(adapter, TrunkFinetuneConfig(finetune="last_blocks", unfreeze_last=1))
    window = TrunkWindow.for_adapter(adapter, 256, 100)
    live = RegionCountModel(_config(), trunk=LiveTrunk(adapter, window))
    _load_warm_start(live, tmp_path / "cached")
    for name, tensor in cached.state_dict().items():
        torch.testing.assert_close(live.state_dict()[name], tensor)

    live.save_pretrained(tmp_path / "live")
    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "live" / "model.safetensors"))
    trunk_keys = {key for key in saved if key.startswith("trunk.")}
    assert trunk_keys == {"trunk.adapter.block.weight", "trunk.adapter.block.bias"}

    live.train()
    assert not live.trunk.training


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_live_trunk_trains_end_to_end_on_cpu(tmp_path, monkeypatch):
    data = _toy_region_data()
    dataset_dir = tmp_path / "dataset"
    data.write(dataset_dir)
    fasta = tmp_path / "genome.fa"
    rng = np.random.default_rng(0)
    _write_fasta(fasta, {"chr1": "".join(rng.choice(list("ACGT"), size=40_000))})

    import regulonado.model.adapters as adapters_module

    built: list[_ConvAdapter] = []
    initial: dict[str, torch.Tensor] = {}

    def build(spec):
        built.append(_ConvAdapter())
        initial.update({k: v.clone() for k, v in built[-1].state_dict().items()})
        return built[-1]

    monkeypatch.setattr(adapters_module, "build_backbone_adapter", build)
    cfg = _base_cfg(dataset_dir, "", tmp_path / "out")
    cfg["data"].update(fasta=str(fasta), input_length=2048, shift_max=16, pool_to=None)
    cfg["data"]["embeddings_dir"] = ""
    cfg["model"]["pooling"] = "per_group"
    cfg["backbone"] = {"name": "stub", "pretrained_name": "none", "features": "trunk"}
    cfg["trunk"] = {"finetune": "last_blocks", "unfreeze_last": 1}
    cfg["trainer"]["backbone_learning_rate"] = 1.0e-3

    summary = run_training(cfg)
    adapter = built[0]
    assert summary["backbone"] == "stub"
    assert summary["history"]["train/loss"]
    # Only the unfrozen last block moved; the frozen first block is untouched.
    assert not torch.equal(adapter.block.weight.cpu(), initial["block.weight"])
    assert torch.equal(adapter.embed.weight.cpu(), initial["embed.weight"])
    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert "trunk.adapter.block.weight" in saved
    assert "trunk.adapter.embed.weight" not in saved
