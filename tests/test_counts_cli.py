"""``regulonado counts``: ``bam`` then ``gather`` through the CLI, end to end."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

pytest.importorskip("pysam")

from regulonado.cli.counts import counts_app  # noqa: E402
from regulonado.counts.bam import TrackCounts  # noqa: E402
from regulonado.counts.dataset import RegionCountData  # noqa: E402
from regulonado.tracks_table import write_track_table  # noqa: E402
from test_counts_bam import _write_se_bam  # noqa: E402

runner = CliRunner()


@pytest.fixture
def bam_dir(tmp_path):
    directory = tmp_path / "bams"
    directory.mkdir()
    # Track "a": 5 dense insertions near the region; track "b": none.
    _write_se_bam(directory / "a.bam", [{"pos": 5000 + i} for i in range(5)], length=20_000)
    _write_se_bam(directory / "b.bam", [{"pos": 15000}], length=20_000)
    return directory


@pytest.fixture
def tracks_path(tmp_path, bam_dir):
    path = tmp_path / "tracks.parquet"
    frame = pd.DataFrame(
        {
            "track_name": ["a", "b"],
            "status": ["included", "included"],
            "track_index": [0, 1],
            "bam": [str(bam_dir / "a.bam"), str(bam_dir / "b.bam")],
            "assay_class": ["ATAC", "ATAC"],
            "group": ["hl60", "hl60"],
        }
    )
    write_track_table(frame, path)
    return path


@pytest.fixture
def regions_path(tmp_path):
    path = tmp_path / "regions.parquet"
    pl.DataFrame({"chrom": ["chr1"], "start": [4900], "end": [5450]}).write_parquet(path)
    return path


@pytest.fixture
def anchor_and_background(tmp_path):
    anchor = tmp_path / "anchor.bed"
    anchor.write_text("chr1\t4900\t5450\n")
    background = tmp_path / "background.bed"
    background.write_text("chr1\t15000\t15500\n")
    return anchor, background


def test_bam_command_counts_every_track_into_the_cache_dir(
    tmp_path, regions_path, tracks_path, anchor_and_background
):
    anchor, background = anchor_and_background
    out_dir = tmp_path / "counts"
    result = runner.invoke(
        counts_app,
        [
            "bam",
            str(regions_path),
            "--tracks",
            str(tracks_path),
            "--anchor-regions",
            str(anchor),
            "--background-regions",
            str(background),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    a = TrackCounts.load(out_dir / "a.parquet")
    b = TrackCounts.load(out_dir / "b.parquet")
    assert a.counts.tolist() == [5]
    assert b.counts.tolist() == [0]


def test_bam_command_with_track_option_counts_only_that_track(
    tmp_path, regions_path, tracks_path, anchor_and_background
):
    anchor, background = anchor_and_background
    out_dir = tmp_path / "counts"
    result = runner.invoke(
        counts_app,
        [
            "bam",
            str(regions_path),
            "--tracks",
            str(tracks_path),
            "--anchor-regions",
            str(anchor),
            "--background-regions",
            str(background),
            "--out-dir",
            str(out_dir),
            "--track",
            "a",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "a.parquet").exists()
    assert not (out_dir / "b.parquet").exists()


def test_bam_command_rejects_an_unknown_track_name(
    tmp_path, regions_path, tracks_path, anchor_and_background
):
    anchor, background = anchor_and_background
    result = runner.invoke(
        counts_app,
        [
            "bam",
            str(regions_path),
            "--tracks",
            str(tracks_path),
            "--anchor-regions",
            str(anchor),
            "--background-regions",
            str(background),
            "--out-dir",
            str(tmp_path / "counts"),
            "--track",
            "nonexistent",
        ],
    )
    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_gather_command_assembles_a_region_count_dataset(
    tmp_path, regions_path, tracks_path, anchor_and_background
):
    anchor, background = anchor_and_background
    out_dir = tmp_path / "counts"
    runner.invoke(
        counts_app,
        [
            "bam",
            str(regions_path),
            "--tracks",
            str(tracks_path),
            "--anchor-regions",
            str(anchor),
            "--background-regions",
            str(background),
            "--out-dir",
            str(out_dir),
        ],
    )

    dataset_dir = tmp_path / "dataset"
    result = runner.invoke(
        counts_app,
        [
            "gather",
            str(regions_path),
            "--tracks",
            str(tracks_path),
            "--out-dir",
            str(out_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--val-chroms",
            "chr2",
        ],
    )
    assert result.exit_code == 0, result.output

    data = RegionCountData.read(dataset_dir)
    assert data.n_regions == 1
    assert data.track_names == ["a", "b"]
    assert data.counts.tolist() == [[5.0, 0.0]]
    assert data.regions["split"].to_list() == ["train"]  # chr1 not in --val-chroms
    assert "log_size_factor" in data.tracks.columns
    assert set(data.tracks.columns) >= {"track_name", "assay_class", "group"}
