"""Genomic coordinate and sequence primitives shared across the package.

All coordinates are 0-based, half-open (BED convention).

This module is a *home*, not a library: it collects the one-hot/base-LUT sequence helpers, the
prediction-window geometry rule, and the BED/chrom-sizes readers that previously accreted across
``dataset/build.py``, ``inference.py``, ``normalization.py`` and ``qc.py`` as three divergent
hand-rolled parsers reached into via underscore-private cross-module imports (see audit findings
F01, F08, R1, R6, S6).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Base -> one-hot row. Unknown bases (incl. N) map to an all-zero column.
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for _base, _row in {"A": 0, "C": 1, "G": 2, "T": 3}.items():
    _BASE_LUT[ord(_base)] = _row
    _BASE_LUT[ord(_base.lower())] = _row

_DECODE_LUT = np.array(list("ACGT"))


@dataclass(slots=True)
class Window:
    """A single prediction window: predicted region + surrounding model context."""

    chrom: str
    pred_start: int  # start of the predicted (central) region, bp
    pred_end: int  # end of the predicted region, bp (== pred_start + n_pred_bins*bin_size)
    ctx_start: int  # start of the model input context, bp (may be < 0 near chrom start)
    ctx_end: int  # end of the model input context, bp (may be > chrom length near chrom end)


def window_for_interval(
    chrom: str,
    start: int,
    end: int,
    *,
    context_length: int,
    n_pred_bins: int,
    bin_size: int,
) -> Window:
    """Derive the prediction window centred on a BED interval's midpoint.

    This is the rule shared by whole-genome/targeted inference tiling, the dataset window
    index used to seed the design search, and the design module's "no containing window"
    fallback: the predicted region and its surrounding model context are centred on the
    interval midpoint, not on the interval's own span::

        center = (start + end) // 2
        pred_start = center - pred_bp // 2       (pred_bp = n_pred_bins * bin_size)
        ctx_start = center - context_length // 2

    Neither ``pred_start`` nor ``ctx_start`` is clamped to ``>= 0`` and neither is clamped at
    the chromosome end — callers that read sequence or signal from the result are responsible
    for clamping/zero-padding (see :func:`one_hot_context`).
    """
    pred_bp = n_pred_bins * bin_size
    center = (start + end) // 2
    pred_start = center - pred_bp // 2
    ctx_start = center - context_length // 2
    return Window(
        chrom=chrom,
        pred_start=pred_start,
        pred_end=pred_start + pred_bp,
        ctx_start=ctx_start,
        ctx_end=ctx_start + context_length,
    )


# --------------------------------------------------------------------------- #
# One-hot arrays                                                             #
# --------------------------------------------------------------------------- #
def one_hot(seq: str) -> np.ndarray:
    """Encode a DNA string as ``(4, L)`` int8. Unknown bases (incl. N) -> all-zero column."""
    codes = _BASE_LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    out = np.zeros((4, len(seq)), dtype=np.int8)
    valid = codes >= 0
    out[codes[valid], np.nonzero(valid)[0]] = 1
    return out


def decode(one_hot_array: np.ndarray) -> str:
    """Decode a ``(4, L)`` one-hot array back to a DNA string; all-zero columns -> 'N'."""
    has_base = one_hot_array.any(axis=0)
    indices = one_hot_array.argmax(axis=0)
    chars = np.where(has_base, _DECODE_LUT[indices], "N")
    return "".join(chars.tolist())


def reverse_complement(one_hot_array: np.ndarray) -> np.ndarray:
    """Reverse-complement a ``(4, L)`` one-hot array (matches ``dataset/build.py``'s convention)."""
    return np.flip(one_hot_array, axis=(0, 1)).copy()


def one_hot_context(fasta, window: Window, context_length: int, chrom_length: int) -> np.ndarray:
    """One-hot encode the model context for ``window`` as an ``int8`` ``(4, context_length)`` array.

    Positions running off either chromosome end are zero-padded (treated as N).
    """
    out = np.zeros((4, context_length), dtype=np.int8)
    fetch_start = max(0, window.ctx_start)
    fetch_end = min(chrom_length, window.ctx_end)
    if fetch_end <= fetch_start:
        return out

    seq = str(fasta[window.chrom][fetch_start:fetch_end])
    codes = _BASE_LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    valid = codes >= 0
    offset = fetch_start - window.ctx_start
    cols = np.arange(offset, offset + codes.shape[0])[valid]
    out[codes[valid], cols] = 1
    return out


# --------------------------------------------------------------------------- #
# Bin collapsing                                                              #
# --------------------------------------------------------------------------- #
def collapse_bins(
    values: np.ndarray,
    chrom: str,
    pred_start: int,
    bin_size: int,
    rtol: float,
    chrom_length: int,
) -> list[tuple[str, int, int, float]]:
    """Run-length collapse per-bin values into ``(chrom, start, end, value)`` intervals.

    Adjacent bins are merged while the next bin is within a relative tolerance of the current
    run's mean: ``abs(v - mean) <= rtol * max(abs(v), abs(mean))`` (so flat zero regions, where
    both are 0, always merge). Bins are clamped to ``[0, chrom_length)``; empty after clamping
    are skipped.
    """
    intervals: list[tuple[str, int, int, float]] = []
    run_start = run_end = -1
    run_sum = 0.0
    run_count = 0

    def flush() -> None:
        if run_count:
            intervals.append((chrom, run_start, run_end, run_sum / run_count))

    for bin_index, raw in enumerate(values):
        start = pred_start + bin_index * bin_size
        end = start + bin_size
        start = max(start, 0)
        end = min(end, chrom_length)
        if end <= start:
            continue
        v = float(raw)
        if run_count and start == run_end:
            mean = run_sum / run_count
            if abs(v - mean) <= rtol * max(abs(v), abs(mean)):
                run_end = end
                run_sum += v
                run_count += 1
                continue
        flush()
        run_start, run_end, run_sum, run_count = start, end, v, 1
    flush()
    return intervals


def write_bigwigs(
    out_dir: str | Path,
    track_names: Sequence[str],
    selected: Sequence[int],
    accum: dict[int, list[tuple[str, int, int, float]]],
    chrom_sizes: dict[str, int],
) -> list[Path]:
    """Write one BigWig per selected track from accumulated ``(chrom, start, end, value)``."""
    import pybigtools

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_rank = {name: i for i, name in enumerate(chrom_sizes)}
    written: list[Path] = []
    output_names: set[str] = set()
    for track in selected:
        intervals = sorted(accum[track], key=lambda r: (chrom_rank[r[0]], r[1]))
        track_name = safe_track_filename(track_names[track])
        if track_name in output_names:
            raise ValueError(f"Track names produce duplicate output filename: {track_name!r}")
        output_names.add(track_name)
        path = out_dir / f"{track_name}.bw"
        writer = pybigtools.open(str(path), "w")
        writer.write(chrom_sizes, iter(intervals))
        written.append(path)
    return written


def safe_track_filename(name: str) -> str:
    """Validate a track name before using it as a filesystem component."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Track name is not a safe filename: {name!r}")
    if any(character in name for character in "\x00\r\n"):
        raise ValueError(f"Track name contains a control character: {name!r}")
    if len(name.encode("utf-8")) > 200:
        raise ValueError("Track name is too long to use as a filename")
    return name


# --------------------------------------------------------------------------- #
# BED / chrom-sizes reading                                                  #
# --------------------------------------------------------------------------- #
def _is_gzipped(path: Path) -> bool:
    """Sniff the gzip magic bytes rather than trusting the ``.gz`` suffix."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _bed_error(path: str | Path, line_number: int, reason: str, line: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {reason}: {line!r}")


def read_intervals(path: str | Path) -> pd.DataFrame:
    """Read a BED-like file into a ``chrom``/``start``/``end`` (``[name]``) DataFrame.

    The single reader for every BED-shaped input in the package (dataset build intervals,
    prediction/attribution position lists, normalization anchor/background regions). Accepts:

    * gzip compression, detected from the file's magic bytes rather than its suffix
    * tab- or whitespace-delimited fields, and CRLF line endings
    * blank lines, ``#`` comments, and UCSC ``track``/``browser`` header lines (skipped)
    * a 4th column, exposed as ``name`` (e.g. a fold label) when present on any row; rows
      missing it get ``""``. Columns beyond the 4th are ignored.

    Row order is preserved. Raises :class:`ValueError` naming ``path`` and the offending
    1-based line number for a row with fewer than 3 fields, a non-integer ``start``/``end``,
    a negative ``start``, or ``end <= start``.

    bioframe's own ``read_table`` does not meet this spec — it hard-codes a tab delimiter and
    has no comment/header handling, so it raises uninformative ``ValueError``s on exactly the
    malformed input this function is meant to handle cleanly (see audit finding F01).
    """
    path = Path(path)
    open_fn = gzip.open if _is_gzipped(path) else open

    rows: list[tuple[str, int, int, str | None]] = []
    has_name = False
    with open_fn(path, "rt") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 3:
                raise _bed_error(path, line_number, "expected at least 3 fields", line)
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                raise _bed_error(
                    path, line_number, "start/end must be integers", line
                ) from None
            if start < 0:
                raise _bed_error(path, line_number, "start must be >= 0", line)
            if end <= start:
                raise _bed_error(path, line_number, "end must be > start", line)
            name = fields[3] if len(fields) > 3 else None
            if name is not None:
                has_name = True
            rows.append((chrom, start, end, name))

    if not has_name:
        frame = pd.DataFrame(rows, columns=["chrom", "start", "end", "_name"])
        frame = frame.drop(columns=["_name"])
    else:
        frame = pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])
        frame["name"] = frame["name"].fillna("")

    frame["chrom"] = frame["chrom"].astype(str)
    frame["start"] = frame["start"].astype("int64")
    frame["end"] = frame["end"].astype("int64")
    return frame


def read_chrom_sizes(path: str | Path) -> dict[str, int]:
    """Parse a two-column ``chrom<TAB>size`` chrom.sizes file (or a ``.fai``), order preserved.

    Whitespace- or tab-delimited, gzip (sniffed from magic bytes), ``#`` comments and blank
    lines are skipped, and only the first two fields of each line are read — so a ``.fai``
    (three further columns) works interchangeably with a real chrom.sizes file. Deliberately
    not ``bioframe.read_chromsizes``: its defaults (``filter_chroms=True``, UCSC-primary-only
    regexes, natural sort) would both reorder chromosomes and silently drop scaffolds/alt
    contigs, and its hard-coded tab separator rejects whitespace-delimited input — whole-genome
    tiling (:func:`regulonado.inference.iter_windows`) relies on every contig being present, in
    file order, for its chromosome rank.
    """
    path = Path(path)
    open_fn = gzip.open if _is_gzipped(path) else open
    sizes: dict[str, int] = {}
    with open_fn(path, "rt") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 2:
                continue
            sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError(f"No chromosome sizes parsed from {path}")
    return sizes


# --------------------------------------------------------------------------- #
# Overlap removal                                                            #
# --------------------------------------------------------------------------- #
def drop_overlaps(frame: pd.DataFrame, chrom_order) -> tuple[pd.DataFrame, int]:
    """Sort ``frame`` by chromosome order then start, dropping rows that overlap a prior one.

    ``frame`` must have ``chrom``/``start``/``end`` columns (plus any others, e.g. ``value``,
    which are carried through unchanged). ``chrom_order`` is any iterable of chromosome names
    (a ``chrom_sizes`` dict works directly) giving the sort order.

    Coordinates are 0-based, half-open: touching intervals ``[a, b)`` and ``[b, c)`` do **not**
    overlap and both are kept. Within each chromosome, after sorting by ``start``, a row is
    dropped when its ``start`` is strictly less than the running max ``end`` of the rows before
    it (so an interval nested entirely inside an earlier, wider one is also dropped, not just one
    that extends past it). The comparison never crosses a chromosome boundary.

    Returns ``(kept, n_dropped)`` with ``kept`` re-indexed from 0. Does not log; callers decide
    whether/how to report ``n_dropped``.
    """
    rank = {name: i for i, name in enumerate(chrom_order)}
    frame = frame.sort_values(
        by=["chrom", "start"], key=lambda col: col.map(rank) if col.name == "chrom" else col
    ).reset_index(drop=True)
    prior_end = frame.groupby("chrom")["end"].cummax().shift(fill_value=-1)
    same_chrom = frame["chrom"].eq(frame["chrom"].shift())
    overlapping = same_chrom & (frame["start"] < prior_end)
    n_dropped = int(overlapping.sum())
    kept = frame[~overlapping].reset_index(drop=True)
    return kept, n_dropped


__all__ = [
    "Window",
    "collapse_bins",
    "decode",
    "drop_overlaps",
    "one_hot",
    "one_hot_context",
    "read_chrom_sizes",
    "read_intervals",
    "reverse_complement",
    "safe_track_filename",
    "window_for_interval",
    "write_bigwigs",
]
