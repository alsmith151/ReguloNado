"""Parity and unit contracts for the indexed Rust BAM target modes."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pysam
import pytest
import torch
from bam_reference import reference_count_events
from regulonado import _rs
from regulonado.counts.bam import CountSpec, count_events
from regulonado.training.label_space import resolve_count_label_space
from regulonado.training.losses import (
    mask_missing_bins,
    poisson_multinomial_binwise_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
)
from test_counts_bam import _write_pe_bam, _write_se_bam


def measure(path, spec, windows, mode="events", convention="regulonado"):
    raw, paired, single, high, low = _rs.count_bam_windows(
        str(path),
        json.dumps({**asdict(spec), "insertion_convention": convention}),
        windows,
        [],
        [],
        300,
        2,
        mode,
    )
    return np.frombuffer(raw, dtype="<f8")


@pytest.mark.parametrize("unit", ["insertions", "fragments"])
@pytest.mark.parametrize("proper", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_rust_events_match_reference_with_mixed_layouts(tmp_path, unit, proper, drop):
    path = tmp_path / "mixed.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 20000}]}
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        flags = [0, 16, 1024, 256, 512, 2048, 1, 3 | 128, 3 | 64 | 16, 3 | 128]
        for i, flag in enumerate(flags):
            r = pysam.AlignedSegment()
            r.query_name = str(i)
            r.query_sequence = "A" * 30
            r.cigartuples = [(4, 5), (0, 10), (2, 3), (3, 7), (7, 10), (1, 5)]
            r.reference_id = 0
            r.reference_start = 1 + i * 100
            r.flag = flag
            r.mapping_quality = [255, 30, 29][i % 3]
            r.template_length = [200, -200, 0][i % 3]
            r.next_reference_id = 0
            r.next_reference_start = r.reference_start + 100
            bam.write(r)
    pysam.index(str(path))
    spec = CountSpec(
        unit,
        tn5_shift=unit == "insertions",
        proper_pairs_only=proper,
        drop_duplicates=drop,
        max_fragment_length=250,
        single_end_extension=100 if unit == "fragments" else None,
    )
    expected = reference_count_events(path, "chr1", spec)
    actual = count_events(path, "chr1", spec, threads=2)
    for a, b in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(a, b)
    assert actual[2:] == expected[2:]


def test_insertion_profiles_sum_to_region_counts_and_shift_is_explicit(tmp_path):
    path = tmp_path / "atac.bam"
    _write_se_bam(path, [{"pos": 100}, {"pos": 200, "reverse": True}], read_length=50)
    spec = CountSpec("insertions", tn5_shift=True)
    windows = [("chr1", i, i + 32) for i in range(0, 320, 32)]
    bins = measure(path, spec, windows)
    assert bins.sum() == measure(path, spec, [("chr1", 0, 320)])[0] == 2
    assert measure(path, spec, [("chr1", 244, 245)])[0] == 1
    assert measure(path, spec, [("chr1", 245, 246)], convention="chrombpnet")[0] == 1
    assert measure(path, spec, [("chr1", 249, 250)], convention="unshifted")[0] == 1


def test_fragment_profile_mass_is_distinct_from_fragment_count(tmp_path):
    path = tmp_path / "chip.bam"
    _write_pe_bam(path, [200])
    windows = [("chr1", 990, 1020), ("chr1", 1020, 1050), ("chr1", 1190, 1220)]
    spec = CountSpec("fragments")
    sums = measure(path, spec, windows, "sum")
    means = measure(path, spec, windows, "mean")
    np.testing.assert_array_equal(sums, [20, 30, 10])
    np.testing.assert_allclose(means * 30, sums)
    np.testing.assert_array_equal(measure(path, spec, windows), [1, 1, 1])
    assert measure(path, spec, [("chr1", 990, 1220)])[0] == 1


def test_missing_index_and_contig_fail_clearly(tmp_path):
    path = tmp_path / "reads.bam"
    _write_se_bam(path, [{"pos": 100}])
    with pytest.raises(ValueError, match="absent"):
        measure(path, CountSpec("insertions"), [("chr2", 0, 100)])
    path.with_suffix(".bam.bai").unlink()
    with pytest.raises(ValueError, match="index required"):
        count_events(path, "chr1", CountSpec("insertions"))


@pytest.mark.parametrize(
    "loss", [poisson_multinomial_loss, poisson_multinomial_binwise_loss, poisson_nll_loss]
)
@pytest.mark.parametrize("scale", [1.0, 32.0])
def test_profile_losses_support_coverage_units_and_missing_bins(loss, scale):
    target = torch.tensor([[[0.0, 0.5, 2.0, float("nan")]]]) * scale
    pred = torch.tensor([[[0.2, 0.6, 1.8, 0.1]]], requires_grad=True)
    value = loss(*mask_missing_bins(pred * scale, target))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(pred.grad).all()
    assert pred.grad[0, 0, -1] == 0
    exact = torch.tensor([[[0.0, 0.5, 2.0]]]) * scale
    assert loss(exact + 1e-6, exact) < loss(torch.full_like(exact, 5 * scale), exact)
    zero = torch.zeros(1, 1, 4, requires_grad=True)
    empty = loss(zero, torch.zeros_like(zero))
    empty.backward()
    assert torch.isfinite(empty) and torch.isfinite(zero.grad).all()


def test_stored_bam_units_are_not_converted_again():
    records = [
        {"target_source": "bam", "stored_unit": "insertions"},
        {"target_source": "bam", "stored_unit": "coverage_sum"},
    ]
    space = resolve_count_label_space(records, count_unit="stored", exposure="none", bin_size=32)
    np.testing.assert_array_equal(space.count_factors, [1, 1])
    with pytest.raises(ValueError, match="units differ"):
        resolve_count_label_space(records, count_unit="fragments", exposure="none", bin_size=32)


def test_checkpoint_pooling_mismatch_is_rejected(tmp_path):
    from regulonado.training.runner import _validate_target_checkpoint

    model = SimpleNamespace(config=SimpleNamespace(target_contract={"pooling": "mean"}))
    (tmp_path / "config.json").write_text(json.dumps({"target_contract": {"pooling": "sum"}}))
    with pytest.raises(ValueError, match="units/settings"):
        _validate_target_checkpoint(model, tmp_path)


@pytest.mark.parametrize("pooling", ["mean", "sum"])
def test_bam_profile_build_and_unit_metadata(tmp_path, pooling):
    import pandas as pd
    import pyarrow.parquet as pq
    from regulonado.dataset import build_dataset
    from regulonado.tracks_table import read_track_table, write_track_table

    path = tmp_path / "chip.bam"
    _write_pe_bam(path, [200], length=20000)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "A" * 20000 + "\n")
    pysam.faidx(str(fasta))
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t1000\t1200\ttrain\n")
    tracks = tmp_path / "tracks.parquet"
    write_track_table(
        pd.DataFrame(
            {
                "track_name": ["chip"],
                "track_index": [0],
                "status": ["included"],
                "path": [str(path)],
                "resolved_path": [str(path)],
                "bam": [str(path)],
                "assay_class": ["ChIP"],
                "track_format": ["bam"],
            }
        ),
        tracks,
    )
    output = tmp_path / "dataset"
    build_dataset(
        bed,
        fasta,
        tracks,
        output,
        source="bam",
        bam_pooling=pooling,
        context_length=1024,
        bin_size=10,
        n_pred_bins=20,
        splits={"train": ["train"]},
        n_extract_threads=1,
        write_threads=1,
        stage_to_scratch=False,
    )
    shard = next((output / "data").glob("*.parquet"))
    signal = np.asarray(pq.read_table(shard)["signal"].to_pylist())[0]
    np.testing.assert_array_equal(signal, np.full_like(signal, 1 if pooling == "mean" else 10))
    table = read_track_table(output / "tracks.parquet")
    assert table.stored_unit.iloc[0] == ("mean_coverage" if pooling == "mean" else "coverage_sum")
    assert json.loads((output / "target_contract.json").read_text())["pooling"] == pooling
    with pytest.raises(ValueError, match="units/settings changed"):
        build_dataset(
            bed,
            fasta,
            tracks,
            output,
            source="bam",
            bam_pooling="sum" if pooling == "mean" else "mean",
        )


def test_bigwig_regions_match_crested_summaries_and_roundtrip(tmp_path):
    import pandas as pd
    import polars as pl
    import pybigtools
    from regulonado.counts.bam import BamRegionCounter
    from regulonado.counts.signal import count_signal_regions, gather_signal_regions, signal_tracks
    from regulonado.tracks_table import write_track_table

    path = tmp_path / "signal.bw"
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": 1000}, iter([("chr1", 100, 110, 2.0), ("chr1", 115, 120, 4.0)]))
    tracks = tmp_path / "tracks.parquet"
    write_track_table(
        pd.DataFrame(
            {
                "track_name": ["a"],
                "track_index": [0],
                "status": ["included"],
                "path": [str(path)],
                "resolved_path": [str(path)],
            }
        ),
        tracks,
    )
    windows = [("chr1", 100, 120), ("chr1", 200, 220)]
    for mode, expected in [("mean", 2), ("count", 40), ("max", 4), ("logcount", np.log1p(40))]:
        actual = np.frombuffer(_rs.bigwig_regions(str(path), windows, mode), dtype="<f8")
        np.testing.assert_allclose(actual, [expected, 0])
    regions = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [120]})
    counter = BamRegionCounter(regions, target_width=20)
    table = signal_tracks(tracks)
    count_signal_regions(counter, table, tmp_path / "cache", "mean")
    data = gather_signal_regions(counter, table, tmp_path / "cache", tmp_path / "dataset", "mean")
    assert data.counts.tolist() == [[2]]
    assert data.tracks["target_source"].to_list() == ["bigwig"]


def test_region_cache_invalidates_when_windows_change(tmp_path):
    import polars as pl
    from regulonado.counts.bam import BamRegionCounter

    path = tmp_path / "reads.bam"
    _write_se_bam(path, [{"pos": 1050}])
    tracks = pl.DataFrame({"track_name": ["a"], "bam": [str(path)], "assay_class": ["ATAC"]})

    def counter(start):
        return BamRegionCounter(
            pl.DataFrame({"chrom": ["chr1"], "start": [start], "end": [start + 100]})
        )

    first = counter(1000)
    second = counter(5000)
    assert first.count_tracks(tracks, cache_dir=tmp_path / "cache")[0].counts.tolist() == [1]
    with pytest.raises(ValueError, match="scored regions"):
        second.gather(tmp_path / "cache", tracks)
    assert second.count_tracks(tracks, cache_dir=tmp_path / "cache")[0].counts.tolist() == [0]


def test_region_signal_model_uses_linear_mse():
    from regulonado.training.regions.model import RegionCountConfig, RegionCountModel

    config = RegionCountConfig(
        k=2,
        d=4,
        track_groups=[0],
        log_size_factors=[0.0],
        track_names=["a"],
        group_names=["g"],
        hidden=4,
        count_noise="mse",
    )
    model = RegionCountModel(config).eval()
    features = torch.ones(2, 2, 4)
    labels = torch.tensor([[-2.0], [3.0]])
    result = model(features=features, labels=labels)
    torch.testing.assert_close(result.logits, model.group_log_rates(features))
    assert torch.isfinite(result.loss)
    result.loss.backward()


@pytest.mark.parametrize("unit", ["insertions", "fragments"])
def test_zero_reference_span_matches_pysam(tmp_path, unit):
    path = tmp_path / "soft.bam"
    with pysam.AlignmentFile(
        path, "wb", header={"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    ) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "soft"
        read.query_sequence = "A" * 50
        read.cigartuples = [(4, 50)]
        read.reference_id = 0
        read.reference_start = 10
        read.flag = 16
        read.mapping_quality = 60
        bam.write(read)
    pysam.index(str(path))
    spec = CountSpec(unit)
    actual = count_events(path, "chr1", spec)
    expected = reference_count_events(path, "chr1", spec)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
