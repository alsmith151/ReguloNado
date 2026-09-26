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


def _discover_bam_tracks(tmp_path, *discover_args):
    """``tracks discover --format bam`` then ``tracks assemble``: the count model's tracks."""
    from regulonado.cli.tracks import tracks_app

    discovered = tmp_path / "discovered.parquet"
    result = runner.invoke(
        tracks_app, ["discover", str(discovered), "--format", "bam", *discover_args]
    )
    assert result.exit_code == 0, result.output
    table = tmp_path / "count-tracks.parquet"
    result = runner.invoke(tracks_app, ["assemble", str(discovered), "--output", str(table)])
    assert result.exit_code == 0, result.output
    return table


def test_bam_tracks_discovered_from_a_sheet_count_end_to_end(
    tmp_path, regions_path, bam_dir, anchor_and_background
):
    """A sheet naming samples by run accession finds BAMs anywhere under --bam-dir."""
    nested = bam_dir / "study"
    nested.mkdir()
    (bam_dir / "a.bam").rename(nested / "x_SRR1_sorted.bam")
    (bam_dir / "a.bam.bai").rename(nested / "x_SRR1_sorted.bam.bai")
    sheet = tmp_path / "tracks.csv"
    sheet.write_text(
        "track_name,sample_id,group,assay\nhl60_a,SRR1,hl60,atac\nhl60_b,b,hl60,ATAC-seq\n"
    )
    tracks = _discover_bam_tracks(tmp_path, "--track-sheet", str(sheet), "--bam-dir", str(bam_dir))

    from regulonado.tracks_table import read_track_table, track_format

    table = read_track_table(tracks)
    assert track_format(table) == "bam"
    assert table["bam"].tolist() == [str(nested / "x_SRR1_sorted.bam"), str(bam_dir / "b.bam")]

    anchor, background = anchor_and_background
    out_dir = tmp_path / "counts"
    result = runner.invoke(
        counts_app,
        [
            "bam",
            str(regions_path),
            "--tracks",
            str(tracks),
            "--anchor-regions",
            str(anchor),
            "--background-regions",
            str(background),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert TrackCounts.load(out_dir / "hl60_a.parquet").counts.tolist() == [5.0]

    dataset_dir = tmp_path / "dataset"
    result = runner.invoke(
        counts_app,
        [
            "gather",
            str(regions_path),
            "--tracks",
            str(tracks),
            "--out-dir",
            str(out_dir),
            "--dataset-dir",
            str(dataset_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    meta = RegionCountData.read(dataset_dir).tracks
    assert meta["group"].to_list() == ["hl60", "hl60"]
    assert meta["assay_class"].to_list() == ["ATAC", "ATAC"]


def test_bam_dir_alone_makes_every_bam_a_track(tmp_path, bam_dir):
    tracks = _discover_bam_tracks(tmp_path, "--bam-dir", str(bam_dir))
    from regulonado.tracks_table import read_track_table

    assert read_track_table(tracks)["track_name"].tolist() == ["a", "b"]


def test_bigwig_tracks_carry_their_bam_for_a_shared_count_model(tmp_path, bam_dir):
    """--bam-dir on bigWig discovery pairs <stem>.bam, so both model types share tracks."""
    from conftest import write_bigwig
    from regulonado.cli.tracks import tracks_app
    from regulonado.counts.bam import read_count_tracks

    bigwigs = tmp_path / "bigwigs"
    bigwigs.mkdir()
    write_bigwig(bigwigs / "a.bw", [("chr1", 0, 100, 1.0)])
    discovered = tmp_path / "discovered.parquet"
    result = runner.invoke(
        tracks_app,
        ["discover", str(discovered), "--bigwig-dir", str(bigwigs), "--bam-dir", str(bam_dir)],
    )
    assert result.exit_code == 0, result.output
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("track_name,assay,group\na,atac,hl60\n")
    table = tmp_path / "tracks.parquet"
    result = runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "--output", str(table), "--annotations", str(annotations)],
    )
    assert result.exit_code == 0, result.output
    tracks = read_count_tracks(table)
    assert tracks["bam"].to_list() == [str((bam_dir / "a.bam").resolve())]


def test_bigwig_stages_refuse_a_bam_track_table(tmp_path, bam_dir):
    from regulonado.cli.tracks import tracks_app

    tracks = _discover_bam_tracks(tmp_path, "--bam-dir", str(bam_dir))
    result = runner.invoke(
        tracks_app,
        ["interval-means", str(tracks), "-o", str(tmp_path / "m.parquet"), "--intervals", "x.bed"],
    )
    assert result.exit_code != 0
    assert "discovered from bam files" in result.output


def test_count_tracks_need_a_known_assay(tmp_path, bam_dir):
    from regulonado.counts.bam import read_count_tracks

    sheet = tmp_path / "tracks.csv"
    sheet.write_text("track_name,bam,assay\na,a.bam,rna\n")
    (tmp_path / "a.bam").write_bytes((bam_dir / "a.bam").read_bytes())
    tracks = _discover_bam_tracks(tmp_path, "--track-sheet", str(sheet))
    with pytest.raises(ValueError, match="no assay_class"):
        read_count_tracks(tracks)


def test_find_bam_rejects_an_ambiguous_accession(tmp_path):
    from regulonado.tracks import find_bam

    (tmp_path / "a_SRR1.bam").touch()
    (tmp_path / "b_SRR1.bam").touch()
    with pytest.raises(ValueError, match="more than one BAM"):
        find_bam(tmp_path, "SRR1")


def test_regions_command_writes_the_rows_gather_produces(tmp_path, anchor_and_background):
    regions = tmp_path / "regions.parquet"
    pl.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [4900, 4900, 100], "end": [5450, 5450, 700]}
    ).write_parquet(regions)
    output = tmp_path / "region_set.parquet"
    result = runner.invoke(
        counts_app, ["regions", str(regions), "-o", str(output), "--test-chroms", "chr2"]
    )
    assert result.exit_code == 0, result.output
    region_set = pl.read_parquet(output)
    assert region_set.height == 2  # duplicate dropped
    assert region_set["split"].to_list() == ["train", "test"]
    assert region_set["target_end"].to_list() == [
        s + 1000 for s in region_set["target_start"].to_list()
    ]
