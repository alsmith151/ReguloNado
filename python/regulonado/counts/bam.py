"""Per-region read counts straight from BAMs, ported from ``unique_enhancer_finding.bam_counts``.

The atlas bigWigs are coverage at a fixed bin size: each bin holds the number
of reads overlapping it, so summing bins over a window counts one read
roughly ``length / bin_bp`` times, with ``length`` differing per track (PE
fragment vs SE read, assay). That is fine for log-MSE targets, but no count
likelihood (Poisson, NB, multinomial across tasks) is valid on it.

So this module reads the BAMs with pysam and counts *events*, one per unit
the assay actually measures (:class:`CountSpec`):

``insertions`` (ATAC)
    the Tn5-shifted 5' end of every read -- both mates of a PE fragment, one
    per SE read. Each end is a separate transposition event.
``fragments`` (ChIP, CUT&RUN)
    each fragment (PE, from the TLEN>0 mate) or read (SE, optionally extended
    along its strand) counted once if it overlaps the window. No cut sites and
    no shift: histone-mark signal sits on flanking nucleosomes, not at ends.

Both units reduce to half-open intervals (an insertion is ``[p, p + 1)``), so a
single overlap count serves every window: ``#(start < b) - #(end <= a)`` over
the sorted starts and ends.

Size factors are measured in count units, from the same events, so offset and
target share a unit: per window, the most events in any ``scale_bp``
sub-window; ``count_scale_high`` is its median over the anchor windows
(``--anchor-regions``), ``count_scale_low`` its median over the background
windows (``--background-regions``) -- the same anchor scheme
``regulonado normalization anchor`` uses for bigWigs, re-measured on raw
events since the anchor's coverage-unit scale factors carry a per-track
``length / bin_bp`` factor the counts do not.

Coordinates are 0-based half-open, as everywhere in ReguloNado.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

__all__ = [
    "BIN_BP",
    "DEFAULT_COUNT_SPECS",
    "DEFAULT_SCALE_BP",
    "SCALE_BP",
    "WINDOW_BP",
    "BamRegionCounter",
    "CountSpec",
    "TrackCounts",
    "count_events",
    "read_count_tracks",
    "recentre_windows",
    "scale_bp_for",
]

CountUnit = Literal["insertions", "fragments"]

#: SAM flags never counted: unmapped, secondary, QC-fail, supplementary.
_EXCLUDED_FLAGS = 0x4 | 0x100 | 0x200 | 0x800
_DUPLICATE_FLAG = 0x400

#: Tn5 binds as a dimer offset by 9 bp; the insertion centre sits +4 from a
#: forward read's 5' end and -5 from a reverse read's. The +/-1 bp convention
#: differences between tools are immaterial at window resolution.
_TN5_FORWARD_SHIFT = 4
_TN5_REVERSE_SHIFT = -5

#: Width every anchor/background QC window is re-cut to before measurement.
WINDOW_BP = 1_000

#: Bin width the rolling-maximum count statistic is stepped at. Fixed,
#: regardless of window width.
BIN_BP = 100

#: Sub-window the rolling maximum is taken over, per assay: narrow for the
#: sharp accessibility peaks, the full window for broad H3K27ac domains.
SCALE_BP: dict[str, int] = {"CUT&RUN": 200, "ATAC": 300, "ChIP": 1_000}

DEFAULT_SCALE_BP = 1_000


def scale_bp_for(assay_class: str | None) -> int:
    """Sub-window width for *assay_class*, falling back to :data:`DEFAULT_SCALE_BP`."""
    return SCALE_BP.get(str(assay_class), DEFAULT_SCALE_BP)


def recentre_windows(
    windows: Sequence[tuple[str, int, int]], width: int = WINDOW_BP
) -> list[tuple[str, int, int]]:
    """Re-cut every window to *width* bp around its midpoint.

    Windows running off the start of a contig (``start <= 0``) are dropped.
    Coordinates are 0-based half-open.
    """
    recut = []
    for chrom, start, end in windows:
        midpoint = (int(start) + int(end)) // 2
        new_start = midpoint - width // 2
        if new_start > 0:
            recut.append((str(chrom), new_start, new_start + width))
    return recut


@dataclass(frozen=True)
class CountSpec:
    """How to turn one BAM's reads into countable events.

    Attributes:
        unit: ``"insertions"`` (one event per read 5' end) or ``"fragments"``
            (one event per molecule, as an interval).
        tn5_shift: shift 5' ends +4/-5 (``insertions`` only).
        min_mapq: minimum mapping quality.
        drop_duplicates: skip reads flagged as duplicates (0x400). Count
            likelihoods assume independent molecules; this only helps if the
            BAMs were duplicate-marked.
        proper_pairs_only: for paired reads, keep only proper pairs. Single-end
            reads are unaffected, so mixed-layout BAMs work.
        max_fragment_length: drop paired reads with ``|TLEN|`` above this.
        single_end_extension: ``fragments`` only -- extend SE reads to this
            length along their strand; ``None`` keeps the aligned span.
    """

    unit: CountUnit
    tn5_shift: bool = False
    min_mapq: int = 30
    drop_duplicates: bool = True
    proper_pairs_only: bool = True
    max_fragment_length: int | None = None
    single_end_extension: int | None = None

    def __post_init__(self) -> None:
        if self.unit not in ("insertions", "fragments"):
            raise ValueError(f"unit must be 'insertions' or 'fragments'; got {self.unit!r}")
        if self.tn5_shift and self.unit != "insertions":
            raise ValueError("tn5_shift only applies to unit='insertions'")
        if self.single_end_extension is not None and self.unit != "fragments":
            raise ValueError("single_end_extension only applies to unit='fragments'")


#: Per-``assay_class`` defaults.
DEFAULT_COUNT_SPECS: dict[str, CountSpec] = {
    "ATAC": CountSpec("insertions", tn5_shift=True),
    "ChIP": CountSpec("fragments"),
    "CUT&RUN": CountSpec("fragments"),
}


def count_events(
    bam_path: str | Path, chrom: str, spec: CountSpec, *, threads: int = 1
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Stream one chromosome of a BAM into sorted event starts and ends.

    Returns:
        ``(starts, ends, n_paired, n_single)``: the sorted int64 start and
        (separately sorted) end coordinates of every event, and how many of
        them came from paired vs single-end reads.
    """
    import pysam

    # array("q") rather than lists: ~8 bytes per event instead of ~36, which
    # matters at tens of millions of events on the large chromosomes.
    starts = array("q")
    ends = array("q")
    n_paired = 0
    n_single = 0
    excluded = _EXCLUDED_FLAGS | (_DUPLICATE_FLAG if spec.drop_duplicates else 0)

    with pysam.AlignmentFile(str(bam_path), "rb", threads=threads) as bam:
        if chrom not in bam.references:
            return np.empty(0, np.int64), np.empty(0, np.int64), 0, 0
        for read in bam.fetch(chrom):
            if read.flag & excluded or read.mapping_quality < spec.min_mapq:
                continue
            paired = read.is_paired
            if paired:
                if spec.proper_pairs_only and not read.is_proper_pair:
                    continue
                if (
                    spec.max_fragment_length is not None
                    and abs(read.template_length) > spec.max_fragment_length
                ):
                    continue

            if spec.unit == "insertions":
                if read.is_reverse:
                    position = read.reference_end - 1
                    position += _TN5_REVERSE_SHIFT if spec.tn5_shift else 0
                else:
                    position = read.reference_start
                    position += _TN5_FORWARD_SHIFT if spec.tn5_shift else 0
                starts.append(position)
                ends.append(position + 1)
            elif paired:
                # One event per fragment: only the leftmost mate (TLEN > 0).
                if read.template_length <= 0:
                    continue
                starts.append(read.reference_start)
                ends.append(read.reference_start + read.template_length)
            else:
                start, end = read.reference_start, read.reference_end
                if spec.single_end_extension is not None:
                    if read.is_reverse:
                        start = end - spec.single_end_extension
                    else:
                        end = start + spec.single_end_extension
                starts.append(start)
                ends.append(end)

            if paired:
                n_paired += 1
            else:
                n_single += 1

    return (
        np.sort(np.frombuffer(starts, dtype=np.int64)),
        np.sort(np.frombuffer(ends, dtype=np.int64)),
        n_paired,
        n_single,
    )


def _count_overlaps(
    starts: np.ndarray, ends: np.ndarray, window_start: np.ndarray, window_end: np.ndarray
) -> np.ndarray:
    """Events overlapping each ``[window_start, window_end)``: ``#(s < b) - #(e <= a)``."""
    return np.searchsorted(starts, window_end, side="left") - np.searchsorted(
        ends, window_start, side="right"
    )


def _best_subwindow_counts(
    starts: np.ndarray,
    ends: np.ndarray,
    window_start: np.ndarray,
    window_bp: int,
    scale_bp: int,
    bin_bp: int = BIN_BP,
) -> np.ndarray:
    """Most events in any *scale_bp* sub-window, stepped at *bin_bp*, per window."""
    width = min(max(bin_bp, scale_bp // bin_bp * bin_bp), window_bp)
    offsets = np.arange(0, window_bp - width + 1, bin_bp, dtype=np.int64)
    sub_start = window_start[:, None] + offsets[None, :]
    counts = _count_overlaps(starts, ends, sub_start.ravel(), (sub_start + width).ravel())
    return counts.reshape(sub_start.shape).max(axis=1)


_COUNT_COLUMN = "count"
_METADATA_KEY = b"regulonado_track_counts"


@dataclass(frozen=True)
class TrackCounts:
    """One BAM's counts over the region set, plus its count-unit anchor levels.

    Attributes:
        track_name: the track this came from.
        counts: ``[n_regions]`` int64 events per target window, row-aligned to
            :attr:`BamRegionCounter.regions`.
        spec: how the events were defined.
        n_events: events over all counted chromosomes (library size, in the
            same unit as *counts*).
        n_paired, n_single: events from paired vs single-end reads.
        count_scale_high: median best-sub-window count over the anchor
            windows, or ``nan`` if none were given.
        count_scale_low: the same over the background windows, or ``nan``.
    """

    track_name: str
    counts: np.ndarray
    spec: CountSpec
    n_events: int
    n_paired: int
    n_single: int
    count_scale_high: float = float("nan")
    count_scale_low: float = float("nan")

    def save(self, path: str | Path) -> Path:
        """Write to a single parquet file: one ``count`` column plus key/value metadata.

        Resumable counting relies on this file existing at *path*: a track
        whose parquet is already present is loaded instead of recounted (see
        :meth:`BamRegionCounter.count_tracks`).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        metadata = {
            "track_name": self.track_name,
            "spec": asdict(self.spec),
            "n_events": self.n_events,
            "n_paired": self.n_paired,
            "n_single": self.n_single,
            "count_scale_high": self.count_scale_high,
            "count_scale_low": self.count_scale_low,
        }
        table = pa.table({_COUNT_COLUMN: pa.array(self.counts, type=pa.int64())})
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(metadata).encode()}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        import os

        os.replace(tmp_path, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrackCounts:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path} has no {_METADATA_KEY!r} metadata; not a TrackCounts file")
        metadata = json.loads(raw)
        counts = table.column(_COUNT_COLUMN).to_numpy(zero_copy_only=False).astype(np.int64)
        return cls(
            track_name=metadata["track_name"],
            counts=counts,
            spec=CountSpec(**metadata["spec"]),
            n_events=int(metadata["n_events"]),
            n_paired=int(metadata["n_paired"]),
            n_single=int(metadata["n_single"]),
            count_scale_high=float(metadata["count_scale_high"]),
            count_scale_low=float(metadata["count_scale_low"]),
        )


#: Free-text ``assay`` spellings (track sheets, SeqNado project assays) mapped onto the
#: ``assay_class`` keys of :data:`DEFAULT_COUNT_SPECS`; matched case-insensitively.
ASSAY_CLASS_ALIASES: dict[str, str] = {
    "atac": "ATAC",
    "atac-seq": "ATAC",
    "atacseq": "ATAC",
    "chip": "ChIP",
    "chip-seq": "ChIP",
    "chipseq": "ChIP",
    "cut&run": "CUT&RUN",
    "cutandrun": "CUT&RUN",
    "cut_and_run": "CUT&RUN",
    "cutrun": "CUT&RUN",
    "cut-run": "CUT&RUN",
    "cut_run": "CUT&RUN",
}


def _assay_class_for(assay: object) -> str | None:
    if assay is None:
        return None
    text = str(assay).strip()
    if text in DEFAULT_COUNT_SPECS:
        return text
    return ASSAY_CLASS_ALIASES.get(text.lower())


def read_count_tracks(path: str | Path, *, require_bam: bool = True) -> pl.DataFrame:
    """Read ``tracks.parquet``'s included tracks as ``track_name``/``bam``/``assay_class``/
    ``group``.

    Either discovery format works: ``tracks discover --format bam`` (the count model's own
    tracks) or a bigWig table whose rows carry a ``bam`` (from a track sheet, a SeqNado
    project or ``--bam-dir``), so a count model can reuse a coverage model's track set.
    ``assay_class`` is taken as-is when present, otherwise derived from ``assay``
    (:data:`ASSAY_CLASS_ALIASES`, e.g. ``atac`` -> ``ATAC``).

    Raises:
        ValueError: on an unresolvable ``assay_class``, or (with *require_bam*) an
            included track with no ``bam``.
    """
    import pandas as pd

    from regulonado.tracks_table import read_track_table

    table = read_track_table(path)
    included = table[table["status"] == "included"].sort_values("track_index")
    rows = included.reset_index(drop=True).to_dict("records")

    def _value(row: dict, column: str) -> object:
        value = row.get(column)
        return None if value is None or pd.isna(value) else value

    assay_classes = [
        _assay_class_for(_value(row, "assay_class")) or _assay_class_for(_value(row, "assay"))
        for row in rows
    ]
    unresolved = [
        f"{row['track_name']} ({_value(row, 'assay_class') or _value(row, 'assay')!r})"
        for row, assay_class in zip(rows, assay_classes, strict=True)
        if assay_class is None
    ]
    if unresolved:
        raise ValueError(
            f"{path}: no assay_class for track(s) {unresolved}; give the track sheet an "
            f"'assay' column using one of {sorted(ASSAY_CLASS_ALIASES)} (or an "
            f"'assay_class' of {', '.join(DEFAULT_COUNT_SPECS)} via track annotations)"
        )

    bams = [_value(row, "bam") for row in rows]
    if require_bam:
        no_bam = [row["track_name"] for row, bam in zip(rows, bams, strict=True) if bam is None]
        if no_bam:
            raise ValueError(
                f"{path}: track(s) with no 'bam': {no_bam}. Discover them with "
                "'tracks discover --format bam', or give the bigWig track sheet a 'bam' "
                "column / --bam-dir"
            )

    groups = [_value(row, "group") for row in rows]
    return pl.DataFrame(
        {
            "track_name": pl.Series([str(row["track_name"]) for row in rows], dtype=pl.Utf8),
            "bam": pl.Series([None if b is None else str(b) for b in bams], dtype=pl.Utf8),
            "assay_class": pl.Series(assay_classes, dtype=pl.Utf8),
            "group": pl.Series([None if g is None else str(g) for g in groups], dtype=pl.Utf8),
        }
    )


@dataclass
class BamRegionCounter:
    """Count BAM events over a fixed region set.

    The target window events are counted in is 1 kb (:data:`WINDOW_BP`'s
    project-wide width), centred on each input region's midpoint; the cache
    the count head trains against supplies sequence context, so no wider
    model-input window is carried here.

    Typical use, locally::

        counter = BamRegionCounter(
            regions, chrom_sizes=sizes, anchor_windows=fit, background_windows=bg
        )
        results = counter.count_tracks(tracks, max_workers=4, cache_dir=out_dir)

    or on the cluster, one BAM per array task, then a gather::

        counter.count_track(tracks.row(i, named=True)).save(out_dir / f"{track_name}.parquet")
        results = counter.gather(out_dir, tracks)

    Attributes:
        regions: input regions, ``chrom``/``start``/``end`` plus any extra
            columns (``split``, provenance flags).
        target_width: width of the window events are counted in, centred on each
            region -- unless *regions* already has ``target_start``/``target_end``
            (of this width), which are kept as given.
        chrom_sizes: contig lengths for the bounds check, or ``None`` to skip.
        anchor_windows: anchor windows (``count_scale_high``).
        background_windows: background/null windows (``count_scale_low``).
        specs: per-``assay_class`` :class:`CountSpec`, overriding
            :data:`DEFAULT_COUNT_SPECS`.
        threads: htslib decompression threads per BAM.
    """

    regions: pl.DataFrame
    target_width: int = WINDOW_BP
    chrom_sizes: Mapping[str, int] | None = None
    anchor_windows: pl.DataFrame | Sequence[tuple[str, int, int]] | None = None
    background_windows: pl.DataFrame | Sequence[tuple[str, int, int]] | None = None
    specs: Mapping[str, CountSpec] = field(default_factory=dict)
    threads: int = 1
    n_dropped_out_of_bounds: int = field(init=False, default=0)
    n_dropped_duplicates: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.regions = self._build_windows(self.regions)
        self.anchor_windows = _recentred_frame(self.anchor_windows)
        self.background_windows = _recentred_frame(self.background_windows)
        self.specs = {**DEFAULT_COUNT_SPECS, **dict(self.specs)}

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def _build_windows(self, regions: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in ("chrom", "start", "end") if c not in regions.columns]
        if missing:
            raise ValueError(f"regions is missing columns: {missing}")
        frame = regions.with_columns(
            pl.col("chrom").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
        )

        target = self.target_width
        if {"target_start", "target_end"} <= set(frame.columns):
            # A region set that already names its scored windows (e.g. UEF's, cut from the
            # un-resized peaks) keeps them, so counts line up with that set's signal targets.
            frame = frame.with_columns(
                pl.col("target_start").cast(pl.Int64), pl.col("target_end").cast(pl.Int64)
            )
            widths = (frame["target_end"] - frame["target_start"]).unique().to_list()
            if widths != [target]:
                raise ValueError(
                    f"regions carry target_start/target_end of width(s) {sorted(widths)[:5]}, "
                    f"but target_width is {target}"
                )
        else:
            centre = (pl.col("start") + pl.col("end")) // 2
            frame = frame.with_columns(
                pl.max_horizontal(pl.lit(0, pl.Int64), centre - target // 2).alias("target_start")
            ).with_columns((pl.col("target_start") + target).alias("target_end"))

        n_before = frame.height
        if self.chrom_sizes is not None:
            sizes = pl.DataFrame(
                {"chrom": list(self.chrom_sizes), "_size": list(self.chrom_sizes.values())},
                schema={"chrom": pl.Utf8, "_size": pl.Int64},
            )
            frame = (
                frame.join(sizes, on="chrom", how="left", maintain_order="left")
                .filter(
                    (pl.col("target_start") >= 0)
                    & (pl.col("target_end") <= pl.col("_size"))
                    & pl.col("_size").is_not_null()
                )
                .drop("_size")
            )
        self.n_dropped_out_of_bounds = n_before - frame.height

        n_before = frame.height
        frame = frame.unique(subset=["chrom", "start", "end"], keep="first", maintain_order=True)
        self.n_dropped_duplicates = n_before - frame.height

        leading = ["chrom", "start", "end", "target_start", "target_end"]
        return frame.select(leading + [c for c in frame.columns if c not in leading])

    @property
    def n_regions(self) -> int:
        return self.regions.height

    def spec_for(self, assay_class: str) -> CountSpec:
        try:
            return self.specs[str(assay_class)]
        except KeyError:
            raise KeyError(
                f"no CountSpec for assay_class {assay_class!r}; known: {sorted(self.specs)}"
            ) from None

    # ------------------------------------------------------------------ #
    # counting
    # ------------------------------------------------------------------ #

    def count_track(self, track: Mapping[str, object]) -> TrackCounts:
        """Count one row of a tracks table (``track_name``, ``bam``, ``assay_class``)."""
        assay_class = str(track["assay_class"])
        return self.count_bam(
            str(track["bam"]),
            self.spec_for(assay_class),
            track_name=str(track["track_name"]),
            scale_bp=scale_bp_for(assay_class),
        )

    def count_bam(
        self,
        bam_path: str | Path,
        spec: CountSpec,
        *,
        track_name: str = "",
        scale_bp: int = DEFAULT_SCALE_BP,
    ) -> TrackCounts:
        """Count one BAM over every region, anchor and background window.

        *scale_bp* is the anchor statistic's sub-window (see
        :func:`scale_bp_for`).
        """
        counts = np.zeros(self.n_regions, dtype=np.int64)
        anchor_scores: list[np.ndarray] = []
        background_scores: list[np.ndarray] = []
        n_events = n_paired = n_single = 0

        chroms = sorted(
            set(self.regions["chrom"].unique())
            | set(self.anchor_windows["chrom"].unique())
            | set(self.background_windows["chrom"].unique())
        )
        import pysam

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            absent = sorted(set(self.regions["chrom"].unique()) - set(bam.references))
        if absent:
            # Otherwise a "1" vs "chr1" naming mismatch counts every region as zero.
            raise ValueError(f"{track_name or bam_path}: region contigs absent from BAM: {absent}")

        region_chrom = self.regions["chrom"].to_numpy()
        target_start = self.regions["target_start"].to_numpy()
        target_end = self.regions["target_end"].to_numpy()

        for chrom in chroms:
            starts, ends, paired, single = count_events(bam_path, chrom, spec, threads=self.threads)
            n_events += starts.size
            n_paired += paired
            n_single += single

            mask = region_chrom == chrom
            if mask.any():
                counts[mask] = _count_overlaps(starts, ends, target_start[mask], target_end[mask])
            for windows, scores in (
                (self.anchor_windows, anchor_scores),
                (self.background_windows, background_scores),
            ):
                on_chrom = windows.filter(pl.col("chrom") == chrom)
                if on_chrom.height:
                    scores.append(
                        _best_subwindow_counts(
                            starts, ends, on_chrom["start"].to_numpy(), WINDOW_BP, scale_bp
                        )
                    )

        def _median(scores: list[np.ndarray]) -> float:
            return float(np.median(np.concatenate(scores))) if scores else float("nan")

        return TrackCounts(
            track_name=track_name,
            counts=counts,
            spec=spec,
            n_events=n_events,
            n_paired=n_paired,
            n_single=n_single,
            count_scale_high=_median(anchor_scores),
            count_scale_low=_median(background_scores),
        )

    def count_tracks(
        self,
        tracks: pl.DataFrame,
        *,
        max_workers: int = 1,
        cache_dir: str | Path | None = None,
    ) -> list[TrackCounts]:
        """Count every track, optionally in parallel and resumably.

        With *cache_dir*, each result is saved as ``<track_name>.parquet`` and
        a track whose file already exists is loaded instead of recounted.
        """
        _require_columns(tracks, ("track_name", "bam", "assay_class"))
        rows = tracks.to_dicts()
        cache = Path(cache_dir) if cache_dir is not None else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)

        def cached(row: dict) -> Path | None:
            return None if cache is None else cache / f"{row['track_name']}.parquet"

        todo = [row for row in rows if (p := cached(row)) is None or not p.exists()]
        if max_workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                fresh = dict(
                    zip(
                        [r["track_name"] for r in todo],
                        pool.map(self.count_track, todo),
                        strict=True,
                    )
                )
        else:
            fresh = {row["track_name"]: self.count_track(row) for row in todo}

        results = []
        for row in rows:
            path = cached(row)
            if row["track_name"] in fresh:
                result = fresh[row["track_name"]]
                if path is not None:
                    result.save(path)
            else:
                result = TrackCounts.load(path)
            results.append(result)
        return results

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #

    def gather(
        self, cache_dir: str | Path, tracks: pl.DataFrame
    ) -> tuple[np.ndarray, list[str], list[TrackCounts]]:
        """Load ``<track_name>.parquet`` for every track in *tracks*.

        Returns:
            ``(counts, track_names, results)``: ``counts`` is
            ``[n_regions, n_tracks]`` float32, row-aligned to :attr:`regions`;
            ``track_names`` is in *tracks*' order; ``results`` is the loaded
            :class:`TrackCounts`, same order, for :meth:`anchor_stats`.
        """
        cache = Path(cache_dir)
        order = [str(t) for t in tracks["track_name"]]
        missing = [t for t in order if not (cache / f"{t}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"no counts in {cache} for tracks: {missing}")
        results = [TrackCounts.load(cache / f"{t}.parquet") for t in order]
        for result in results:
            if result.counts.shape[0] != self.n_regions:
                raise ValueError(
                    f"{result.track_name}: {result.counts.shape[0]} counts for {self.n_regions} "
                    "regions -- counted against a different region set"
                )
        matrix = np.vstack([r.counts for r in results]).T.astype(np.float32)
        return matrix, order, results

    def anchor_stats(self, results: Sequence[TrackCounts]) -> pl.DataFrame:
        """Per-track anchor stats and count-unit ``log_size_factor``, row-aligned to *results*.

        ``log_size_factor`` is ``log(count_scale_high - count_scale_low)``,
        centred to mean zero across tracks; ``nan`` where the anchor does not
        clear background.
        """
        stats = pl.DataFrame(
            {
                "track_name": [r.track_name for r in results],
                "count_unit": [r.spec.unit for r in results],
                "n_events": [r.n_events for r in results],
                "n_paired": [r.n_paired for r in results],
                "n_single": [r.n_single for r in results],
                "count_scale_low": [r.count_scale_low for r in results],
                "count_scale_high": [r.count_scale_high for r in results],
            }
        )
        anchor_range = (stats["count_scale_high"] - stats["count_scale_low"]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            log_range = np.where(anchor_range > 0, np.log(anchor_range), np.nan)
        log_size_factor = (
            log_range - np.nanmean(log_range) if np.isfinite(log_range).any() else log_range
        )
        return stats.with_columns(pl.Series("log_size_factor", log_size_factor))


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"tracks is missing columns: {missing}")


def _recentred_frame(
    windows: pl.DataFrame | Sequence[tuple[str, int, int]] | None,
) -> pl.DataFrame:
    """Normalise *windows* to a ``chrom``/``start``/``end`` frame, re-cut to :data:`WINDOW_BP`."""
    if windows is None:
        rows: list[tuple[str, int, int]] = []
    elif isinstance(windows, pl.DataFrame):
        rows = list(
            windows.select(
                pl.col("chrom").cast(pl.Utf8),
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            ).iter_rows()
        )
    else:
        rows = [(str(c), int(s), int(e)) for c, s, e in windows]
    recut = recentre_windows(rows)
    if not recut:
        return pl.DataFrame(schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64})
    return pl.DataFrame(
        {
            "chrom": [r[0] for r in recut],
            "start": [r[1] for r in recut],
            "end": [r[2] for r in recut],
        },
        schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    )
