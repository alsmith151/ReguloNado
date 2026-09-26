"""``regulonado tracks``: discover -> qc -> assemble, end to end through the CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

pytest.importorskip("pybigtools")

from conftest import write_bigwig as _make_bigwig  # noqa: E402
from regulonado.cli.tracks import tracks_app  # noqa: E402
from regulonado.tracks_table import read_track_table  # noqa: E402

runner = CliRunner()


@pytest.fixture
def bigwig_dir(tmp_path):
    bw_dir = tmp_path / "bw"
    bw_dir.mkdir()
    _make_bigwig(bw_dir / "a.bw", [("chr1", 0, 1000, 1.0), ("chr1", 1000, 2000, 2.0)])
    _make_bigwig(bw_dir / "b.bw", [("chr1", 0, 1000, 1.5), ("chr1", 1000, 2000, 1.8)])
    _make_bigwig(bw_dir / "flat.bw", [("chr1", 0, 2000, 0.0)])
    return bw_dir


def test_discover_writes_one_row_per_track_with_a_guaranteed_name(tmp_path, bigwig_dir):
    output = tmp_path / "discovered.parquet"
    result = runner.invoke(
        tracks_app, ["discover", str(output), "--bigwig-dir", str(bigwig_dir)]
    )
    assert result.exit_code == 0, result.output

    table = read_track_table(output)
    assert set(table["track_name"]) == {"a", "b", "flat"}
    assert (table["status"] == "included").all()
    assert table.set_index("track_name").loc["a", "track_index"] == 0


def test_qc_sparsity_flags_a_constant_track(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    qc_report = tmp_path / "qc_report.parquet"
    result = runner.invoke(
        tracks_app, ["qc", str(discovered), "-o", str(qc_report), "--check", "sparsity"]
    )
    assert result.exit_code == 0, result.output

    report = pd.read_parquet(qc_report).set_index("track_name")
    assert report.loc["flat", "qc_verdict"] == "failed"
    assert report.loc["a", "qc_verdict"] == "passed"


def test_assemble_drop_degenerate_excludes_qc_failures_and_renumbers(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    qc_report = tmp_path / "qc_report.parquet"
    runner.invoke(
        tracks_app, ["qc", str(discovered), "-o", str(qc_report), "--check", "sparsity"]
    )

    assembled = tmp_path / "tracks.parquet"
    result = runner.invoke(
        tracks_app,
        [
            "assemble",
            str(discovered),
            "-o",
            str(assembled),
            "--qc-report",
            str(qc_report),
            "--drop-degenerate",
        ],
    )
    assert result.exit_code == 0, result.output

    table = read_track_table(assembled).set_index("track_name")
    assert table.loc["flat", "status"] == "qc_failed"
    assert pd.isna(table.loc["flat", "track_index"])
    # a/b renumber contiguously over the surviving two tracks.
    assert sorted(table.loc[["a", "b"], "track_index"].tolist()) == [0, 1]


def test_assemble_exclude_track_applies_even_with_qc_off(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    assembled = tmp_path / "tracks.parquet"
    result = runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(assembled), "--exclude-track", "b"],
    )
    assert result.exit_code == 0, result.output

    table = read_track_table(assembled).set_index("track_name")
    assert table.loc["b", "status"] == "excluded"
    assert table.loc["a", "status"] == "included"


def test_assemble_annotations_merges_a_group_column(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    annotations = tmp_path / "groups.csv"
    annotations.write_text("track_name,group\na,hl60\nb,k562\n")

    assembled = tmp_path / "tracks.parquet"
    result = runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(assembled), "--annotations", str(annotations)],
    )
    assert result.exit_code == 0, result.output

    table = read_track_table(assembled).set_index("track_name")
    assert table.loc["a", "group"] == "hl60"
    assert table.loc["b", "group"] == "k562"


def test_assemble_drops_tracks_the_annotations_do_not_list(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    annotations = tmp_path / "groups.csv"
    annotations.write_text("track_name,group\nb,k562\n")

    assembled = tmp_path / "tracks.parquet"
    result = runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(assembled), "--annotations", str(annotations)],
    )
    assert result.exit_code == 0, result.output
    assert "2 unannotated" in result.output

    table = read_track_table(assembled).set_index("track_name")
    assert table.loc["a", "status"] == table.loc["flat", "status"] == "unannotated"
    assert pd.isna(table.loc["a", "track_index"])
    assert table.loc["b", "status"] == "included"
    assert table.loc["b", "track_index"] == 0


def test_assemble_annotations_rejects_an_unknown_track_name(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    annotations = tmp_path / "groups.csv"
    annotations.write_text("track_name,group\na,hl60\nnonexistent,k562\n")

    result = runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(tmp_path / "tracks.parquet"), "--annotations",
         str(annotations)],
    )
    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_assemble_column_set_is_identical_with_qc_on_and_off(tmp_path, bigwig_dir):
    """Stable schema: an unrun stage leaves its columns null, not absent."""
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    qc_report = tmp_path / "qc_report.parquet"
    runner.invoke(
        tracks_app, ["qc", str(discovered), "-o", str(qc_report), "--check", "sparsity"]
    )

    no_qc = tmp_path / "no_qc.parquet"
    with_qc = tmp_path / "with_qc.parquet"
    runner.invoke(tracks_app, ["assemble", str(discovered), "-o", str(no_qc)])
    runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(with_qc), "--qc-report", str(qc_report)],
    )

    assert set(read_track_table(no_qc).columns) == set(read_track_table(with_qc).columns)


def test_assemble_is_a_strict_superset_of_every_stage_output(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    qc_report = tmp_path / "qc_report.parquet"
    runner.invoke(
        tracks_app, ["qc", str(discovered), "-o", str(qc_report), "--check", "sparsity"]
    )
    assembled = tmp_path / "tracks.parquet"
    runner.invoke(
        tracks_app,
        ["assemble", str(discovered), "-o", str(assembled), "--qc-report", str(qc_report)],
    )

    final = read_track_table(assembled).set_index("track_name")
    for stage_path in (discovered, qc_report):
        stage = pd.read_parquet(stage_path).set_index("track_name")
        assert set(stage.columns) <= set(final.columns)
        for column in stage.columns:
            pd.testing.assert_series_equal(
                stage[column], final.loc[stage.index, column], check_names=False
            )


def test_show_and_targets_read_the_assembled_table(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    assembled = tmp_path / "tracks.parquet"
    runner.invoke(tracks_app, ["assemble", str(discovered), "-o", str(assembled)])

    show_result = runner.invoke(tracks_app, ["show", str(assembled)])
    assert show_result.exit_code == 0
    assert "included" in show_result.output

    targets_path = tmp_path / "targets.txt"
    targets_result = runner.invoke(
        tracks_app, ["targets", str(assembled), "-o", str(targets_path)]
    )
    assert targets_result.exit_code == 0
    lines = targets_path.read_text().splitlines()
    assert lines[0].split("\t") == [
        "index",
        "identifier",
        "file",
        "clip",
        "clip_soft",
        "scale",
        "sum_stat",
        "description",
        "strand_pair",
    ]
    assert len(lines) == 4  # header + a, b, flat


def test_interval_means_is_shared_by_tmm_and_qc(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])

    intervals = tmp_path / "intervals.bed"
    intervals.write_text("chr1\t0\t640\nchr1\t640\t1280\n")

    means_path = tmp_path / "interval_means.parquet"
    result = runner.invoke(
        tracks_app,
        [
            "interval-means",
            str(discovered),
            "-o",
            str(means_path),
            "--intervals",
            str(intervals),
            "--n-pred-bins",
            "10",
            "--bin-size",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output

    qc_report = tmp_path / "qc_report.parquet"
    result = runner.invoke(
        tracks_app,
        [
            "qc",
            str(discovered),
            "-o",
            str(qc_report),
            "--check",
            "interval_signal",
            "--interval-means",
            str(means_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "qc_nonzero_bin_fraction" in pd.read_parquet(qc_report).columns


def _write_paired_bam(path, template_lengths, read_length=50):
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100_000}]}
    reads = []
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index, tlen in enumerate(template_lengths):
            start = 100 + index * 10
            for is_read1 in (True, False):
                read = pysam.AlignedSegment()
                read.query_name = f"pair{index}"
                read.query_sequence = "A" * read_length
                read.query_qualities = pysam.qualitystring_to_array("I" * read_length)
                read.flag = 0x1 | 0x2 | (0x40 if is_read1 else 0x80)
                read.reference_id = 0
                read.reference_start = start if is_read1 else start + tlen - read_length
                read.next_reference_id = 0
                read.next_reference_start = start + tlen - read_length if is_read1 else start
                read.template_length = tlen if is_read1 else -tlen
                read.cigartuples = [(0, read_length)]
                read.mapping_quality = 60
                reads.append(read)
        for read in sorted(reads, key=lambda r: r.reference_start):
            bam.write(read)
    pysam.index(str(path))


def test_fragment_lengths_writes_mean_template_length_per_track(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    _write_paired_bam(bam_dir / "a.bam", [100, 200, 300])
    _write_paired_bam(bam_dir / "b.bam", [150, 150, 5000])  # 5000 exceeds the cap
    _write_paired_bam(bam_dir / "flat.bam", [80])

    output = tmp_path / "fragment_lengths.csv"
    result = runner.invoke(
        tracks_app,
        ["fragment-lengths", str(discovered), "-o", str(output), "--bam-dir", str(bam_dir)],
    )
    assert result.exit_code == 0, result.output

    lengths = pd.read_csv(output).set_index("track_name")
    assert lengths.loc["a", "fragment_length"] == 200
    assert lengths.loc["b", "fragment_length"] == 150
    assert lengths.loc["a", "read_length"] == 50
    assert lengths.loc["a", "coverage_units"] == 3
    assert (lengths["fragment_length_method"] == "template_length").all()
    assert "coverage_per_unit" in lengths.columns


def _write_single_end_bam(path, fragment_length, *, n_fragments=4000, read_length=36):
    pysam = pytest.importorskip("pysam")
    rng = np.random.default_rng(0)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1_000_000}]}
    reads = []
    for index, start in enumerate(rng.integers(0, 200_000, size=n_fragments)):
        for is_reverse in (False, True):
            read = pysam.AlignedSegment()
            read.query_name = f"frag{index}_{int(is_reverse)}"
            read.query_sequence = "A" * read_length
            read.query_qualities = pysam.qualitystring_to_array("I" * read_length)
            read.flag = 0x10 if is_reverse else 0
            read.reference_id = 0
            offset = fragment_length - read_length if is_reverse else 0
            read.reference_start = int(start + offset)
            read.cigartuples = [(0, read_length)]
            read.mapping_quality = 60
            reads.append(read)
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for read in sorted(reads, key=lambda r: r.reference_start):
            bam.write(read)
    pysam.index(str(path))


@pytest.mark.parametrize(
    ("length_source", "expected", "method"),
    [("auto", 180, "strand_cross_correlation"), ("read", 36, "read_length")],
)
def test_fragment_lengths_single_end(tmp_path, bigwig_dir, length_source, expected, method):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    for stem in ("a", "b", "flat"):
        _write_single_end_bam(bam_dir / f"{stem}.bam", fragment_length=180)

    output = tmp_path / "fragment_lengths.parquet"
    result = runner.invoke(
        tracks_app,
        [
            "fragment-lengths",
            str(discovered),
            "-o",
            str(output),
            "--bam-dir",
            str(bam_dir),
            "--length-source",
            length_source,
        ],
    )
    assert result.exit_code == 0, result.output

    lengths = pd.read_parquet(output)
    assert not lengths["paired"].any()
    assert (lengths["fragment_length_method"] == method).all()
    assert lengths["fragment_length"].sub(expected).abs().max() <= 2
    assert (lengths["coverage_units"] == 8000).all()


def test_fragment_lengths_names_tracks_without_a_bam(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    result = runner.invoke(
        tracks_app,
        [
            "fragment-lengths",
            str(discovered),
            "-o",
            str(tmp_path / "out.csv"),
            "--bam-dir",
            str(bam_dir),
        ],
    )
    assert result.exit_code == 1
    assert "a, b, flat" in result.output


def test_fragment_lengths_keeps_going_past_a_bam_without_usable_reads(tmp_path, bigwig_dir):
    discovered = tmp_path / "discovered.parquet"
    runner.invoke(tracks_app, ["discover", str(discovered), "--bigwig-dir", str(bigwig_dir)])
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    _write_paired_bam(bam_dir / "a.bam", [100, 200, 300])
    _write_paired_bam(bam_dir / "b.bam", [150])
    _write_paired_bam(bam_dir / "flat.bam", [])

    output = tmp_path / "fragment_lengths.csv"
    result = runner.invoke(
        tracks_app,
        ["fragment-lengths", str(discovered), "-o", str(output), "--bam-dir", str(bam_dir)],
    )
    assert result.exit_code == 1
    assert "the BAM has no alignments" in result.output

    lengths = pd.read_csv(output).set_index("track_name")
    assert lengths.loc["a", "fragment_length"] == 200
    assert pd.isna(lengths.loc["flat", "fragment_length"])
    assert "no alignments" in lengths.loc["flat", "error"]
