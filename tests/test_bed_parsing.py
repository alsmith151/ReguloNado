"""Regression fence for the unified BED / chrom-sizes readers (audit finding F01, R1).

Before this unit, ``dataset.build.load_bed_rows``, ``inference._parse_bed`` and
``normalization.read_regions`` were three independent hand-rolled readers that disagreed on
gzip support, delimiter handling, header/comment skipping, short-row handling, and coordinate
validation -- and none of them validated ``start < end`` or ``start >= 0``. They have been
replaced by a single reader, :func:`regulonado.genomics.read_intervals`, used by every caller
(``dataset.build``, ``inference.iter_windows``, ``normalization.read_regions``,
``qc.intervals_from_bed``, ``design.sequence``, the ``design``/``attribute`` CLI commands).

This file pins down the unified reader's behaviour: gzip (sniffed from magic bytes), tab or
whitespace delimiters, CRLF, blank lines, ``#`` comments and ``track``/``browser`` headers are
all accepted; a 4th column is exposed as ``name`` when present; malformed rows (fewer than 3
fields, non-integer coordinates, negative start, end <= start) raise ``ValueError`` naming the
file path and the offending 1-based line number.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from regulonado.genomics import read_chrom_sizes, read_intervals
from regulonado.qc import intervals_from_bed

# ---------------------------------------------------------------------------
# Case bodies (raw file content), written byte-exact via write_bytes so the
# CRLF case's "\r\n" survives -- text-mode writes would translate it away.
# ---------------------------------------------------------------------------

CASE_TEXT: dict[str, str] = {
    "tab_bed3": "chr1\t100\t200\nchr2\t300\t400\n",
    "tab_bed4_fold": "chr1\t100\t200\tfold0\nchr2\t300\t400\tfold1\n",
    "space_delim": "chr1 100 200\nchr2 300 400\n",
    "header_track_browser": "track name=x\nbrowser position chr1:1-100\nchr1\t100\t200\n",
    "comment_line": "#comment\nchr1\t100\t200\n",
    "two_column": "chr1\t100\n",
    "end_lt_start": "chr1\t200\t100\n",
    "end_eq_start": "chr1\t100\t100\n",
    "negative_start": "chr1\t-10\t100\n",
    "non_integer_coord": "chr1\tabc\t200\n",
    "bad_row_on_second_line": "chr1\t100\t200\nchr2\t300\n",
    "crlf": "chr1\t100\t200\r\nchr2\t300\t400\r\n",
    "trailing_blank_line": "chr1\t100\t200\n\n",
}

GZIPPED_CASE = "gzipped_bed3"  # same content as tab_bed3, gzip-compressed on disk (no .gz suffix)


def _write_case(tmp_path: Path, case: str) -> Path:
    """Materialise one case's file on disk, byte-exact."""
    if case == GZIPPED_CASE:
        # Deliberately no ".gz" suffix: the reader must sniff the gzip magic bytes,
        # not trust the filename.
        path = tmp_path / "regions.bed"
        with gzip.open(path, "wb") as fh:
            fh.write(CASE_TEXT["tab_bed3"].encode())
        return path
    path = tmp_path / "regions.bed"
    path.write_bytes(CASE_TEXT[case].encode())
    return path


def _rows(frame) -> list[tuple]:
    """DataFrame -> list of plain tuples, chrom/start/end (+name if present)."""
    cols = ["chrom", "start", "end"] + (["name"] if "name" in frame.columns else [])
    return list(frame[cols].itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# EXPECTED[case] = ("ok", <rows>) | ("raises", <1-based line number>)
# ---------------------------------------------------------------------------

EXPECTED: dict[str, tuple] = {
    "tab_bed3": ("ok", [("chr1", 100, 200), ("chr2", 300, 400)]),
    "tab_bed4_fold": ("ok", [("chr1", 100, 200, "fold0"), ("chr2", 300, 400, "fold1")]),
    "space_delim": ("ok", [("chr1", 100, 200), ("chr2", 300, 400)]),
    "header_track_browser": ("ok", [("chr1", 100, 200)]),
    "comment_line": ("ok", [("chr1", 100, 200)]),
    "two_column": ("raises", 1),
    "end_lt_start": ("raises", 1),
    "end_eq_start": ("raises", 1),  # zero-width interval: end <= start is now rejected
    "negative_start": ("raises", 1),
    "non_integer_coord": ("raises", 1),
    "bad_row_on_second_line": ("raises", 2),
    "crlf": ("ok", [("chr1", 100, 200), ("chr2", 300, 400)]),
    "trailing_blank_line": ("ok", [("chr1", 100, 200)]),
    GZIPPED_CASE: ("ok", [("chr1", 100, 200), ("chr2", 300, 400)]),
}

_PARAMS = list(EXPECTED.items())
_PARAM_IDS = [case for case, _ in _PARAMS]


@pytest.mark.parametrize("case,spec", _PARAMS, ids=_PARAM_IDS)
def test_read_intervals_behaviour(tmp_path, case, spec):
    path = _write_case(tmp_path, case)
    outcome, expected = spec
    if outcome == "raises":
        line_number = expected
        with pytest.raises(ValueError) as excinfo:
            read_intervals(path)
        message = str(excinfo.value)
        assert str(path) in message
        assert f":{line_number}:" in message
    else:
        assert _rows(read_intervals(path)) == expected


def test_read_intervals_preserves_row_order(tmp_path):
    path = tmp_path / "regions.bed"
    path.write_bytes(b"chr2\t10\t20\nchr1\t30\t40\nchr2\t50\t60\n")
    assert _rows(read_intervals(path)) == [
        ("chr2", 10, 20),
        ("chr1", 30, 40),
        ("chr2", 50, 60),
    ]


# ---------------------------------------------------------------------------
# qc.intervals_from_bed builds on read_intervals + dataset.build.signal_intervals, so it
# inherits read_intervals's parse failures verbatim, and on success returns transformed
# (sig_start, sig_end) windows rather than raw BED coordinates.
# ---------------------------------------------------------------------------

_QC_KWARGS = dict(n_pred_bins=4, bin_size=32, shift_max_bp=0)


def test_intervals_from_bed_matches_read_intervals_plus_signal_intervals(tmp_path):
    from regulonado.dataset.build import signal_intervals

    path = _write_case(tmp_path, "tab_bed3")
    bed_rows = [(c, s, e, "") for c, s, e in _rows(read_intervals(path))]
    expected = signal_intervals(bed_rows, **_QC_KWARGS)
    assert expected == [("chr1", 86, 214), ("chr2", 286, 414)]
    assert intervals_from_bed(path, **_QC_KWARGS) == expected


def test_intervals_from_bed_propagates_read_intervals_error(tmp_path):
    path = _write_case(tmp_path, "two_column")
    with pytest.raises(ValueError) as excinfo:
        intervals_from_bed(path, **_QC_KWARGS)
    assert str(path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# read_chrom_sizes: order preserved, non-primary contigs kept, .fai accepted.
# ---------------------------------------------------------------------------


def test_read_chrom_sizes_accepts_a_fai_file_as_chrom_sizes(tmp_path):
    # A .fai (samtools faidx index) row is
    # name<TAB>length<TAB>offset<TAB>linebases<TAB>linewidth; read_chrom_sizes
    # only looks at the first two tab-separated fields, so a .fai works
    # interchangeably with a real two-column chrom.sizes file.
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t248956422\t0\t60\t61\nchr2\t242193529\t253105344\t60\t61\n")
    assert read_chrom_sizes(fai) == {"chr1": 248956422, "chr2": 242193529}


def test_read_chrom_sizes_keeps_non_primary_contigs(tmp_path):
    # No allow-list / primary-assembly filtering: any line with >= 2 fields
    # is kept as-is, including alt/unplaced contigs like chrUn_*.
    path = tmp_path / "with_alt.chrom.sizes"
    path.write_text("chr1\t1000\nchrUn_KI270742v1\t500\n")
    sizes = read_chrom_sizes(path)
    assert sizes == {"chr1": 1000, "chrUn_KI270742v1": 500}
    assert "chrUn_KI270742v1" in sizes  # kept, not filtered out


def test_read_chrom_sizes_preserves_file_order(tmp_path):
    path = tmp_path / "unsorted.chrom.sizes"
    path.write_text("chr2\t500\nchr10\t100\nchr1\t1000\n")
    sizes = read_chrom_sizes(path)
    assert list(sizes) == ["chr2", "chr10", "chr1"]
