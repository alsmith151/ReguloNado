"""Embedding cache: tiling geometry, pooling, RC, resume, round-trip and manifest checks.

Uses two stub adapters (fixed-input, like Borzoi/Enformer, and flexible, like AlphaGenome)
whose ``forward_features`` decodes each output bin's *absolute genomic bin index* straight
back out of the one-hot input, rather than doing any real feature extraction. This is made
possible by a synthetic FASTA built from ``_encode_block``: each ``bin_size``-wide block of
sequence is constructed to be its own reverse-complement palindrome (see the block-encoding
proof in that function's docstring) that decodes to the block's absolute bin index -- so the
same decode gives the same answer whether it is read forward or via the RC pass, which is
exactly what makes it possible to test that :mod:`regulonado.embeddings.cache` flips the RC
pass's bins back into forward genomic order correctly (see ``cache.py``'s module docstring).

This lets every test assert exact expected bin indices instead of just shapes, for: normal
in-tile regions, a region whose K bins straddle two tiles (needs an extra window), a window
that pads past both ends of a short contig, ``--pool-to`` averaging, the RC flip, resumable
per-chromosome skipping, the parquet round trip (via ``EmbeddingStore``), and manifest
mismatch detection.
"""

from __future__ import annotations

import polars as pl
import pytest
import torch
from regulonado.embeddings.cache import (
    EmbeddingStore,
    embed_regions,
    read_manifest,
    validate_manifest,
)
from regulonado.model.adapters import BaseBackboneAdapter
from regulonado.sequence import open_genome

BASES = "ACGT"
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _encode_block(bin_index: int, bin_size: int) -> str:
    """A ``bin_size``-bp block that decodes to *bin_index* and is its own RC-palindrome.

    The first half of the block encodes ``bin_index`` two bits per base (standard
    ``A=0 C=1 G=2 T=3``); the second half is the reverse-complement of the first half.
    Writing ``f`` for the first half, the block is ``f + rc(f)``. Since ``rc`` (complement
    then reverse) is an involution and commutes with itself, ``rc(f + rc(f)) == rc(rc(f)) +
    rc(f) == f + rc(f)`` -- the block is unchanged by reverse-complementing, so decoding the
    first ``bin_size // 2`` bases gives the same ``bin_index`` whether the block is read as
    originally written or after a reverse-complement pass. That is what lets the RC tests
    below assert the cache's forward and RC features decode to the *same* bin index once
    correctly flipped back into forward order.
    """
    half = bin_size // 2
    value = bin_index & ((1 << (2 * half)) - 1)
    first_half = "".join(BASES[(value >> (2 * i)) & 0b11] for i in range(half))
    second_half = "".join(_COMPLEMENT[b] for b in reversed(first_half))
    return first_half + second_half


def _position_encoded_sequence(length: int, bin_size: int) -> str:
    assert length % bin_size == 0
    return "".join(_encode_block(i, bin_size) for i in range(length // bin_size))


def _write_fasta(path, chrom_sequences: dict[str, str]) -> None:
    with open(path, "w") as handle:
        for chrom, sequence in chrom_sequences.items():
            handle.write(f">{chrom}\n{sequence}\n")


def _decode_positions(
    input_ids: torch.Tensor, bin_size: int, offset_bp: int, n_bins: int
) -> torch.Tensor:
    """Decode each of *n_bins* output bins' absolute genomic bin index from one-hot input."""
    half = bin_size // 2
    base_codes = input_ids.argmax(dim=1)  # [B, L], 0..3 (all-zero padding decodes as 0)
    powers = (4 ** torch.arange(half)).float()
    out = torch.zeros(input_ids.shape[0], n_bins)
    for j in range(n_bins):
        start = offset_bp + j * bin_size
        codes = base_codes[:, start : start + half].float()
        out[:, j] = (codes * powers).sum(dim=1)
    return out


class _FixedPositionAdapter(BaseBackboneAdapter):
    """Stub mimicking a fixed-input, centre-cropping backbone (Borzoi/Enformer-shaped)."""

    def __init__(
        self, bin_size: int, fixed_input_length: int, crop_bins: int, feature_dim: int = 2
    ):
        super().__init__()
        self.output_bin_size = bin_size
        self.input_multiple = bin_size
        self.fixed_input_length = fixed_input_length
        self.feature_dim = feature_dim
        self._crop_bins = crop_bins
        self.forward_calls = 0

    def output_span(self, input_length: int) -> tuple[int, int]:
        total_bins = input_length // self.output_bin_size
        offset_bins = (total_bins - self._crop_bins) // 2
        return offset_bins * self.output_bin_size, self._crop_bins

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        offset_bp, n_bins = self.output_span(input_ids.shape[-1])
        positions = _decode_positions(input_ids, self.output_bin_size, offset_bp, n_bins)
        features = torch.zeros(input_ids.shape[0], self.feature_dim, n_bins)
        features[:, 0, :] = positions
        return features


class _FlexiblePositionAdapter(BaseBackboneAdapter):
    """Stub mimicking a flexible, fully convolutional backbone (AlphaGenome-shaped)."""

    def __init__(self, bin_size: int, feature_dim: int = 2):
        super().__init__()
        self.output_bin_size = bin_size
        self.input_multiple = bin_size
        self.fixed_input_length = None
        self.feature_dim = feature_dim
        self.forward_calls = 0

    def output_span(self, input_length: int) -> tuple[int, int]:
        if input_length % self.output_bin_size != 0:
            raise ValueError(f"{input_length} is not a multiple of {self.output_bin_size}")
        return 0, input_length // self.output_bin_size

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        offset_bp, n_bins = self.output_span(input_ids.shape[-1])
        positions = _decode_positions(input_ids, self.output_bin_size, offset_bp, n_bins)
        features = torch.zeros(input_ids.shape[0], self.feature_dim, n_bins)
        features[:, 0, :] = positions
        return features


# Fixed-geometry constants shared across tests: bin=32bp, 256-bin (8192bp) input, cropped
# to the central 128 bins (4096bp kept span) -- comfortably larger than K (33 bins) so a
# straddling region's extra window still has room, unlike the tiny geometries real Borzoi
# tests use (this one only needs to be internally consistent, not realistic).
BIN_SIZE = 32
FIXED_INPUT_LENGTH = 8192
CROP_BINS = 128
KEEP_BP = CROP_BINS * BIN_SIZE  # 4096


def _fixed_adapter() -> _FixedPositionAdapter:
    return _FixedPositionAdapter(BIN_SIZE, FIXED_INPUT_LENGTH, CROP_BINS)


def _tile_row(chrom: str, target_start: int, *, end: int = 20000, width: int = 1000) -> dict:
    """One region row on *chrom*, target ``[target_start, target_start + width)``."""
    return {
        "chrom": chrom,
        "start": 0,
        "end": end,
        "target_start": target_start,
        "target_end": target_start + width,
    }


def _regions_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col("target_start").cast(pl.Int64),
        pl.col("target_end").cast(pl.Int64),
    )


@pytest.fixture
def fixed_genome(tmp_path):
    """One short contig (padding on both ends) and one long contig (real tiling)."""
    chrom_pad_length = 94 * BIN_SIZE  # 3008bp: shorter than a full kept span
    chrom_tile_length = 625 * BIN_SIZE  # 20000bp: several tiles
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(
        fasta_path,
        {
            "chrPad": _position_encoded_sequence(chrom_pad_length, BIN_SIZE),
            "chrTile": _position_encoded_sequence(chrom_tile_length, BIN_SIZE),
        },
    )
    return open_genome(fasta_path)


def test_region_bins_cover_target_with_contig_end_padding(tmp_path, fixed_genome):
    # A single region whose 33-bin span sits at the very start of a contig shorter than one
    # full kept span (4096bp): the window backing it (context 8192bp, offset 2048bp before
    # the kept span) runs off both ends of the 3008bp contig, exercising fetch_window's
    # zero-padding on both sides without affecting this region's (in-bounds) decoded bins.
    regions = _regions_frame(
        [_tile_row("chrPad", 10, end=3008)]
    )
    out_dir = tmp_path / "emb"
    adapter = _fixed_adapter()
    embed_regions(regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrPad"])

    manifest = read_manifest(out_dir)
    assert manifest.k == 33  # ceil(1000/32) + 1
    assert manifest.d == 2
    assert manifest.bin_size == 32

    store = EmbeddingStore(out_dir)
    features = store.get(0)
    assert features.shape == (33, 2)
    expected = [float(j) for j in range(33)]  # raw_first == 0 here
    assert features[:, 0].tolist() == expected


def test_straddling_region_gets_its_own_window_and_normal_regions_use_tiles(tmp_path, fixed_genome):
    regions = _regions_frame(
        [
            # In-tile, well inside tile 0's kept span [0, 4096).
            _tile_row("chrTile", 100),
            # Straddles tile 0/1's boundary at bp 4096 -> needs an extra, one-off window.
            _tile_row("chrTile", 4090),
            # In-tile, inside tile 1's kept span [4096, 8192).
            _tile_row("chrTile", 4200),
        ]
    )
    out_dir = tmp_path / "emb"
    adapter = _fixed_adapter()
    embed_regions(regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrTile"])

    store = EmbeddingStore(out_dir)
    assert sorted(store.region_rows) == [0, 1, 2]

    raw_first_a = 100 // BIN_SIZE  # 3
    assert store.get(0)[:, 0].tolist() == [float(raw_first_a + j) for j in range(33)]

    raw_first_b = 4064 // BIN_SIZE  # 127 (bp_start for K=33 bins from floor(4090/32)=127)
    assert store.get(1)[:, 0].tolist() == [float(raw_first_b + j) for j in range(33)]

    raw_first_c = 4192 // BIN_SIZE  # 131
    assert store.get(2)[:, 0].tolist() == [float(raw_first_c + j) for j in range(33)]


def test_pool_to_averages_adjacent_raw_bins(tmp_path, fixed_genome):
    regions = _regions_frame(
        [_tile_row("chrTile", 100)]
    )
    out_dir = tmp_path / "emb"
    adapter = _fixed_adapter()
    embed_regions(
        regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrTile"], pool_to=64
    )

    manifest = read_manifest(out_dir)
    assert manifest.bin_size == 64
    assert manifest.k == 17  # ceil(1000/64) + 1

    store = EmbeddingStore(out_dir)
    features = store.get(0)
    assert features.shape == (17, 2)
    # eff_first_bin = 100 // 64 = 1; raw_first = 1 * 2 = 2; pooled bin m averages raw bins
    # (raw_first + 2m, raw_first + 2m + 1).
    expected = [2 + 2 * m + 0.5 for m in range(17)]
    assert features[:, 0].tolist() == pytest.approx(expected)


def test_rc_flips_bins_back_into_forward_alignment(tmp_path, fixed_genome):
    regions = _regions_frame(
        [_tile_row("chrTile", 4090)]
    )
    out_dir = tmp_path / "emb"
    adapter = _fixed_adapter()
    embed_regions(
        regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrTile"], rc=True
    )

    manifest = read_manifest(out_dir)
    assert manifest.rc is True

    store = EmbeddingStore(out_dir)
    assert store.has_rc is True
    forward = store.get(0, rc=False)
    reverse = store.get(0, rc=True)
    # The stub's block encoding is a reverse-complement palindrome (see _encode_block), so a
    # correctly flipped-back RC pass decodes to exactly the same bin indices as forward.
    assert reverse[:, 0].tolist() == pytest.approx(forward[:, 0].tolist())

    no_rc_dir = tmp_path / "emb_no_rc"
    embed_regions(
        regions, fixed_genome, _fixed_adapter(), no_rc_dir, backbone="stub", chroms=["chrTile"]
    )
    with pytest.raises(ValueError, match="features_rc"):
        EmbeddingStore(no_rc_dir).get(0, rc=True)


def test_per_chromosome_resume_skips_finished_files(tmp_path, fixed_genome):
    regions = _regions_frame(
        [
            _tile_row("chrTile", 100),
            _tile_row("chrPad", 10, end=3008),
        ]
    )
    out_dir = tmp_path / "emb"
    adapter = _fixed_adapter()
    embed_regions(regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrTile"])
    assert adapter.forward_calls > 0
    calls_after_first = adapter.forward_calls
    chrom_path = out_dir / "chrTile.parquet"
    mtime = chrom_path.stat().st_mtime_ns

    # Rerun over both chromosomes: chrTile is already finished and must be skipped (no new
    # forward passes, file untouched); only chrPad is newly processed.
    embed_regions(
        regions, fixed_genome, adapter, out_dir, backbone="stub", chroms=["chrTile", "chrPad"]
    )
    assert chrom_path.stat().st_mtime_ns == mtime
    assert (out_dir / "chrPad.parquet").exists()
    # forward_calls grew only from chrPad's one window, not a re-run of chrTile's.
    assert adapter.forward_calls > calls_after_first


def test_manifest_rejects_mismatched_rerun(tmp_path, fixed_genome):
    regions = _regions_frame(
        [_tile_row("chrTile", 100)]
    )
    out_dir = tmp_path / "emb"
    embed_regions(
        regions, fixed_genome, _fixed_adapter(), out_dir, backbone="stub", chroms=["chrTile"]
    )

    validate_manifest(out_dir, regions)  # matches -- no error

    with pytest.raises(ValueError, match="different"):
        embed_regions(
            regions,
            fixed_genome,
            _fixed_adapter(),
            out_dir,
            backbone="stub",
            chroms=["chrTile"],
            pool_to=64,
        )

    other_regions = _regions_frame(
        [_tile_row("chrTile", 200)]
    )
    with pytest.raises(ValueError, match="different region table"):
        validate_manifest(out_dir, other_regions)


def test_flexible_geometry_context_and_stride(tmp_path):
    # AlphaGenome-shaped: no fixed input length, no built-in crop; tiling keeps only the
    # central `stride` bp of a `context`-bp window, so each kept bin has margin on both sides.
    bin_size = 16
    context = 512
    stride = 256
    chrom_length = 64 * bin_size  # 1024bp: several stride-sized tiles
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path, {"chrFlex": _position_encoded_sequence(chrom_length, bin_size)})
    genome = open_genome(fasta_path)

    regions = _regions_frame(
        [
            # Comfortably inside the first kept span [64, 320) once centred (see below).
            _tile_row("chrFlex", 100, end=chrom_length, width=100),
        ]
    )
    adapter = _FlexiblePositionAdapter(bin_size)
    out_dir = tmp_path / "emb"
    target_width = 100
    embed_regions(
        regions,
        genome,
        adapter,
        out_dir,
        backbone="stub",
        chroms=["chrFlex"],
        context=context,
        stride=stride,
    )
    manifest = read_manifest(out_dir)
    assert manifest.context == context
    assert manifest.stride == stride
    assert manifest.k == -(-target_width // bin_size) + 1  # ceil(100/16) + 1 == 8

    store = EmbeddingStore(out_dir)
    raw_first = 100 // bin_size  # 6
    features = store.get(0)
    assert features[:, 0].tolist() == [float(raw_first + j) for j in range(manifest.k)]


def test_embed_cli_with_monkeypatched_stub_adapter(tmp_path, monkeypatch):
    from regulonado.cli.app import app
    from regulonado.counts.dataset import RegionCountData
    from typer.testing import CliRunner

    regions = _regions_frame(
        [_tile_row("chrTile", 100)]
    )
    dataset_dir = tmp_path / "dataset"
    RegionCountData.from_arrays(regions, counts=[[1.0]], track_names=["t1"]).write(dataset_dir)

    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path, {"chrTile": _position_encoded_sequence(625 * BIN_SIZE, BIN_SIZE)})

    def fake_build_backbone_adapter(spec):
        return _fixed_adapter()

    monkeypatch.setattr(
        "regulonado.model.adapters.build_backbone_adapter", fake_build_backbone_adapter
    )

    out_dir = tmp_path / "emb"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "embed",
            "regions",
            str(dataset_dir),
            str(fasta_path),
            "--backbone",
            "stub",
            "--out",
            str(out_dir),
            "--chroms",
            "chrTile",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "chrTile.parquet").exists()
