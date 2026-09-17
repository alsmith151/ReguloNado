"""Bin-mean denominator and missing-bin value, through both Rust Parquet writers.

One 64 bp contig binned at 8 bp:

    bin 0  [0, 8)    record [0, 4) = 2.0, then no record  -> 1.0 (bin width) / 2.0 (covered bases)
    bin 1  [8, 16)   record NaN                           -> missing
    bin 2  [16, 24)  record 3.0                           -> 3.0
    bins 3-7         no record                            -> 0.0 (zero coverage, not missing)
    bins 8-15        past the contig end (window padding) -> missing
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow.parquet as pq
import pybigtools
import pytest
from test_chrom_pass import _features_json, write_parquet_split_chrom_pass

BIN_SIZE = 8
CHROM_LEN = 64
N_BINS = 16
MISSING = math.nan


@pytest.fixture(scope="module")
def sparse_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("binning_options")
    fasta = root / "ref.fa"
    fasta.write_text(">chrA\n" + "ACGT" * (CHROM_LEN // 4) + "\n")
    import pyfaidx

    pyfaidx.Fasta(str(fasta))

    bw = root / "track.bw"
    writer = pybigtools.open(str(bw), "w")
    writer.write(
        {"chrA": CHROM_LEN},
        iter([("chrA", 0, 4, 2.0), ("chrA", 8, 16, float("nan")), ("chrA", 16, 24, 3.0)]),
    )
    return {"root": root, "fasta": str(fasta), "bw": str(bw)}


def _expected(bin0: float, missing: float) -> np.ndarray:
    return np.array([bin0, missing, 3.0] + [0.0] * 5 + [missing] * 8, dtype=np.float32)


def _write_both(inputs, tmp_path, **binning):
    from regulonado._rs import write_parquet_split_from_bigwigs  # type: ignore[import-not-found]

    signal_intervals = [("chrA", 0, N_BINS * BIN_SIZE)]
    bed_rows = [("chrA", 0, N_BINS * BIN_SIZE, "fold0")]
    context_len = N_BINS * BIN_SIZE

    streaming_path = tmp_path / "streaming.parquet"
    write_parquet_split_from_bigwigs(
        [inputs["bw"]],
        [False],
        signal_intervals,
        str(streaming_path),
        [0],
        bed_rows,
        inputs["fasta"],
        N_BINS,
        context_len,
        _features_json(1, N_BINS, context_len),
        **binning,
    )
    chrom_pass_dir = tmp_path / "chrom_pass"
    chrom_pass_dir.mkdir()
    write_parquet_split_chrom_pass(
        [inputs["bw"]],
        [False],
        signal_intervals,
        chrom_pass_dir,
        [0],
        bed_rows,
        inputs["fasta"],
        N_BINS,
        context_len,
        BIN_SIZE,
        **binning,
    )
    read = lambda path: np.asarray(  # noqa: E731
        pq.read_table(path)["signal"].to_pylist()[0][0], dtype=np.float32
    )
    return read(streaming_path), read(sorted(chrom_pass_dir.glob("split-*.parquet")))


@pytest.mark.parametrize(
    ("binning", "expected"),
    [
        ({}, _expected(1.0, MISSING)),
        ({"mean_over_covered_bases": True}, _expected(2.0, MISSING)),
        ({"missing_as_nan": False}, _expected(1.0, 0.0)),
    ],
    ids=["defaults", "covered_bases", "missing_zero"],
)
def test_binning_options_in_both_writers(sparse_inputs, tmp_path, binning, expected):
    for signal in _write_both(sparse_inputs, tmp_path, **binning):
        np.testing.assert_array_equal(signal, expected)
