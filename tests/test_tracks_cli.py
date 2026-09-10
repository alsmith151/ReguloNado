"""``regulonado tracks``: discover -> qc -> assemble, end to end through the CLI."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

pybigtools = pytest.importorskip("pybigtools")

from regulonado.cli.tracks import tracks_app  # noqa: E402
from regulonado.tracks_table import read_track_table  # noqa: E402

runner = CliRunner()


def _make_bigwig(path, values, size=2000):
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": size}, iter(values))


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
