"""``regulonado.counts.bam``: event counting, region geometry, anchor stats, parquet round-trip.

Uses tiny pysam BAMs, in the style of ``tests/test_tracks_cli.py``'s
``_write_paired_bam``/``_write_single_end_bam`` fixtures.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pysam = pytest.importorskip("pysam")

from regulonado.counts.bam import (  # noqa: E402
    BamRegionCounter,
    CountSpec,
    TrackCounts,
    _best_subwindow_counts,
    _count_overlaps,
    count_events,
    scale_bp_for,
)


def _write_se_bam(path, reads, *, contig="chr1", length=20_000, read_length=36):
    """Write a tiny single-end BAM. Each of *reads* is a dict with keys
    ``pos``, and optionally ``reverse``, ``dup``, ``mapq``."""
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": contig, "LN": length}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index, spec in enumerate(sorted(reads, key=lambda r: r["pos"])):
            read = pysam.AlignedSegment()
            read.query_name = f"read{index}"
            read.query_sequence = "A" * read_length
            read.query_qualities = pysam.qualitystring_to_array("I" * read_length)
            flag = 0x10 if spec.get("reverse") else 0
            flag |= 0x400 if spec.get("dup") else 0
            read.flag = flag
            read.reference_id = 0
            read.reference_start = spec["pos"]
            read.cigartuples = [(0, read_length)]
            read.mapping_quality = spec.get("mapq", 60)
            bam.write(read)
    pysam.index(str(path))


def _write_pe_bam(path, template_lengths, *, contig="chr1", length=20_000, read_length=50):
    """Write a tiny paired-end BAM, one proper pair per template length."""
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": contig, "LN": length}]}
    reads = []
    for index, tlen in enumerate(template_lengths):
        start = 1000 + index * 500
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
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for read in sorted(reads, key=lambda r: r.reference_start):
            bam.write(read)
    pysam.index(str(path))


# --------------------------------------------------------------------- #
# count_events: insertions (Tn5 shift), fragments (paired), filters
# --------------------------------------------------------------------- #


def test_count_events_insertions_applies_tn5_shift_on_both_strands(tmp_path):
    bam_path = tmp_path / "atac.bam"
    _write_se_bam(
        bam_path,
        [{"pos": 100, "reverse": False}, {"pos": 200, "reverse": True}],
        read_length=50,
    )
    spec = CountSpec("insertions", tn5_shift=True)
    starts, ends, n_paired, n_single = count_events(bam_path, "chr1", spec)

    # Forward: 5' end is reference_start, shifted +4 -> 104.
    # Reverse: 5' end is reference_end - 1 (=249), shifted -5 -> 244.
    assert sorted(starts.tolist()) == [104, 244]
    assert sorted(ends.tolist()) == [105, 245]
    assert (n_paired, n_single) == (0, 2)


def test_count_events_insertions_without_tn5_shift_uses_raw_5prime_ends(tmp_path):
    bam_path = tmp_path / "atac.bam"
    _write_se_bam(
        bam_path,
        [{"pos": 100, "reverse": False}, {"pos": 200, "reverse": True}],
        read_length=50,
    )
    spec = CountSpec("insertions", tn5_shift=False)
    starts, _ends, _n_paired, _n_single = count_events(bam_path, "chr1", spec)
    assert sorted(starts.tolist()) == [100, 249]


def test_count_events_drops_duplicates_and_low_mapq_reads(tmp_path):
    bam_path = tmp_path / "atac.bam"
    _write_se_bam(
        bam_path,
        [
            {"pos": 100, "reverse": False},
            {"pos": 300, "reverse": False, "dup": True},
            {"pos": 500, "reverse": False, "mapq": 10},
        ],
        read_length=50,
    )
    spec = CountSpec("insertions", tn5_shift=True, min_mapq=30, drop_duplicates=True)
    starts, _ends, _n_paired, n_single = count_events(bam_path, "chr1", spec)
    assert starts.tolist() == [104]
    assert n_single == 1


def test_count_events_keeps_duplicates_when_not_dropping(tmp_path):
    bam_path = tmp_path / "atac.bam"
    _write_se_bam(
        bam_path,
        [{"pos": 100, "reverse": False}, {"pos": 300, "reverse": False, "dup": True}],
        read_length=50,
    )
    spec = CountSpec("insertions", tn5_shift=True, drop_duplicates=False)
    starts, _ends, _n_paired, n_single = count_events(bam_path, "chr1", spec)
    assert sorted(starts.tolist()) == [104, 304]
    assert n_single == 2


def test_count_events_fragments_counts_one_event_per_pe_fragment(tmp_path):
    bam_path = tmp_path / "chip.bam"
    _write_pe_bam(bam_path, [200, 300])
    spec = CountSpec("fragments")
    starts, ends, n_paired, n_single = count_events(bam_path, "chr1", spec)
    assert starts.tolist() == [1000, 1500]
    assert ends.tolist() == [1200, 1800]
    assert (n_paired, n_single) == (2, 0)


def test_count_events_fragments_single_end_uses_aligned_span(tmp_path):
    bam_path = tmp_path / "cutrun.bam"
    _write_se_bam(bam_path, [{"pos": 100}], read_length=36)
    spec = CountSpec("fragments")
    starts, ends, _n_paired, n_single = count_events(bam_path, "chr1", spec)
    assert starts.tolist() == [100]
    assert ends.tolist() == [136]
    assert n_single == 1


# --------------------------------------------------------------------- #
# _count_overlaps / _best_subwindow_counts: pure numpy geometry
# --------------------------------------------------------------------- #


def test_count_overlaps_matches_half_open_interval_counting():
    starts = np.array([0, 5, 10])
    ends = np.array([2, 7, 12])
    counts = _count_overlaps(starts, ends, np.array([0, 6]), np.array([5, 11]))
    # [0, 5): only [0, 2) overlaps -> 1.  [6, 11): [5, 7) and [10, 12) overlap -> 2.
    np.testing.assert_array_equal(counts, [1, 2])


def test_best_subwindow_counts_finds_the_densest_slice():
    # 1000 single-bp events tiling [0, 1000): the densest 300 bp sub-window
    # holds exactly 300 of them, whichever offset it lands at.
    starts = np.arange(1000, dtype=np.int64)
    ends = starts + 1
    result = _best_subwindow_counts(starts, ends, np.array([0]), window_bp=1000, scale_bp=300)
    assert result.tolist() == [300]


# --------------------------------------------------------------------- #
# BamRegionCounter: target-window geometry, bounds check, duplicate drop
# --------------------------------------------------------------------- #


def test_region_geometry_centres_a_1kb_target_window():
    regions = pl.DataFrame({"chrom": ["chr1"], "start": [4900], "end": [5450]})
    counter = BamRegionCounter(regions, target_width=1000, chrom_sizes={"chr1": 20_000})
    row = counter.regions.row(0, named=True)
    # centre = (4900 + 5450) // 2 = 5175; target = [5175 - 500, 5175 + 500).
    assert (row["target_start"], row["target_end"]) == (4675, 5675)


def test_region_geometry_drops_out_of_bounds_and_duplicate_regions():
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [1000, 1000, 5000],  # first two are duplicates
            "end": [1200, 1200, 5300],
        }
    )
    # chrom is only 5200 bp: region 3's target window [4650, 5650) runs past it.
    counter = BamRegionCounter(regions, target_width=1000, chrom_sizes={"chr1": 5200})
    assert counter.n_regions == 1
    assert counter.n_dropped_duplicates == 1
    assert counter.n_dropped_out_of_bounds == 1
    assert (counter.regions["target_start"][0], counter.regions["target_end"][0]) == (600, 1600)


def test_count_bam_measures_anchor_and_background_count_scale(tmp_path):
    bam_path = tmp_path / "atac.bam"
    reads = [{"pos": 2000 + i} for i in range(10)]  # dense cluster: one anchor window
    reads += [{"pos": 10000}, {"pos": 10500}]  # 500 bp apart: one background window
    _write_se_bam(bam_path, reads, length=20_000, read_length=36)

    regions = pl.DataFrame({"chrom": ["chr1"], "start": [4900], "end": [5450]})
    counter = BamRegionCounter(
        regions,
        target_width=1000,
        chrom_sizes={"chr1": 20_000},
        anchor_windows=[("chr1", 2000, 2500)],
        background_windows=[("chr1", 10000, 10500)],
    )
    spec = CountSpec("insertions", tn5_shift=True)
    result = counter.count_bam(bam_path, spec, track_name="atac", scale_bp=scale_bp_for("ATAC"))

    # All 10 shifted insertions (2004..2013) fit inside one 300 bp sub-window.
    assert result.count_scale_high == 10.0
    # The two background insertions (10004, 10504) are 500 bp apart, so the
    # best 300 bp sub-window catches only one of them.
    assert result.count_scale_low == 1.0
    # No events fall in the (empty) region's target window.
    assert result.counts.tolist() == [0]


def test_anchor_stats_centres_log_size_factor_to_mean_zero_across_tracks():
    empty = np.zeros(1, dtype=np.int64)
    spec = CountSpec("fragments")
    results = [
        TrackCounts("a", empty, spec, 0, 0, 0, count_scale_high=50.0, count_scale_low=2.0),
        TrackCounts("b", empty, spec, 0, 0, 0, count_scale_high=20.0, count_scale_low=4.0),
    ]
    counter = BamRegionCounter(
        pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1000]}), chrom_sizes={"chr1": 10_000}
    )
    stats = counter.anchor_stats(results)

    log_a, log_b = np.log(48.0), np.log(16.0)
    mean = (log_a + log_b) / 2
    by_track = {row["track_name"]: row["log_size_factor"] for row in stats.iter_rows(named=True)}
    np.testing.assert_allclose(by_track["a"], log_a - mean)
    np.testing.assert_allclose(by_track["b"], log_b - mean)
    assert abs(stats["log_size_factor"].sum()) < 1e-9


def test_anchor_stats_is_nan_when_anchor_does_not_clear_background():
    empty = np.zeros(1, dtype=np.int64)
    spec = CountSpec("fragments")
    results = [TrackCounts("a", empty, spec, 0, 0, 0, count_scale_high=1.0, count_scale_low=3.0)]
    counter = BamRegionCounter(
        pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1000]}), chrom_sizes={"chr1": 10_000}
    )
    stats = counter.anchor_stats(results)
    assert np.isnan(stats["log_size_factor"][0])


# --------------------------------------------------------------------- #
# TrackCounts: parquet round-trip, resumable counting
# --------------------------------------------------------------------- #


def test_track_counts_parquet_round_trip(tmp_path):
    counts = np.array([0, 3, 10], dtype=np.int64)
    spec = CountSpec("insertions", tn5_shift=True)
    original = TrackCounts(
        track_name="atac1",
        counts=counts,
        spec=spec,
        n_events=13,
        n_paired=0,
        n_single=13,
        count_scale_high=5.0,
        count_scale_low=1.0,
    )
    path = original.save(tmp_path / "atac1.parquet")
    loaded = TrackCounts.load(path)

    assert loaded.track_name == "atac1"
    np.testing.assert_array_equal(loaded.counts, counts)
    assert loaded.counts.dtype == np.int64
    assert loaded.spec == spec
    assert (loaded.n_events, loaded.n_paired, loaded.n_single) == (13, 0, 13)
    assert (loaded.count_scale_high, loaded.count_scale_low) == (5.0, 1.0)


def test_count_tracks_resumes_from_an_existing_cache_file(tmp_path, monkeypatch):
    regions = pl.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [1200]})
    counter = BamRegionCounter(regions, chrom_sizes={"chr1": 10_000})

    calls: list[str] = []

    def fake_count_track(self, track):
        calls.append(str(track["track_name"]))
        return TrackCounts(
            track_name=str(track["track_name"]),
            counts=np.zeros(self.n_regions, dtype=np.int64),
            spec=CountSpec("fragments"),
            n_events=0,
            n_paired=0,
            n_single=0,
        )

    monkeypatch.setattr(BamRegionCounter, "count_track", fake_count_track)
    tracks = pl.DataFrame({"track_name": ["a"], "bam": ["x.bam"], "assay_class": ["ChIP"]})

    counter.count_tracks(tracks, cache_dir=tmp_path)
    assert calls == ["a"]

    # Second run finds the cached parquet and does not recount.
    counter.count_tracks(tracks, cache_dir=tmp_path)
    assert calls == ["a"]


# --------------------------------------------------------------------- #
# Parity with UEF's unique_enhancer_finding.bam_counts on the same inputs
# --------------------------------------------------------------------- #


def test_parity_with_uef_bam_counts_on_the_same_bam_and_region(tmp_path):
    """Values hard-coded from a scratch run of UEF's bam_counts.py.

    The scratch script built the identical BAM (10 dense ATAC events at
    chr1:2000-2009, a duplicate and a low-MAPQ read near chr1:5300-5400, and
    two background events 500 bp apart at chr1:10000/10500) and region
    (chr1:4900-5450), then called
    ``unique_enhancer_finding.bam_counts.BamRegionCounter.count_bam`` with the
    same anchor/background windows and ``scale_bp=300``. It reported a target
    window of chr1:4675-5675, ``counts=[2]``, ``count_scale_high=10.0`` and
    ``count_scale_low=1.0`` -- reproduced here against ReguloNado's port.
    """
    bam_path = tmp_path / "atac.bam"
    reads = [{"pos": 2000 + i} for i in range(10)]
    reads += [{"pos": 10000}, {"pos": 10500}]
    reads += [
        {"pos": 5000},
        {"pos": 5100, "reverse": True},
        {"pos": 5300, "dup": True},
        {"pos": 5400, "mapq": 10},
    ]
    _write_se_bam(bam_path, reads, length=20_000, read_length=36)

    regions = pl.DataFrame({"chrom": ["chr1"], "start": [4900], "end": [5450]})
    counter = BamRegionCounter(
        regions,
        target_width=1000,
        chrom_sizes={"chr1": 20_000},
        anchor_windows=[("chr1", 2000, 2500)],
        background_windows=[("chr1", 10000, 10500)],
    )
    row = counter.regions.row(0, named=True)
    assert (row["target_start"], row["target_end"]) == (4675, 5675)

    spec = CountSpec("insertions", tn5_shift=True)
    result = counter.count_bam(bam_path, spec, track_name="atac", scale_bp=300)

    assert result.counts.tolist() == [2]
    assert result.count_scale_high == 10.0
    assert result.count_scale_low == 1.0


def test_counter_keeps_a_region_sets_own_target_windows():
    """UEF region sets cut targets from the un-resized peak; they must not be recomputed."""
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [816184],
            "end": [818298],
            "target_start": [816742],
            "target_end": [817742],
        }
    )
    counter = BamRegionCounter(regions, target_width=1000)
    assert counter.regions["target_start"].to_list() == [816742]
    with pytest.raises(ValueError, match="target_width is 500"):
        BamRegionCounter(regions, target_width=500)
