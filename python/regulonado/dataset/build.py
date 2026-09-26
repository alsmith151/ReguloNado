"""HuggingFace-native Parquet dataset builder for sequence-to-function model training.

Builds datasets directly from BED + FASTA + BigWig with no intermediate format.
Sequences are stored as uint8 tokens (A0 C1 G2 T3, 4=N/pad), one-hot encoded on
the GPU at train time; signals are stored as float32 raw coverage (T, B).

Stochastic shift augmentation is supported by storing slightly wider arrays
(``shift_max_bp`` extra context on each side) and cropping at read time.
Transforms (scale / squash / clip / RC augmentation / shift crop) are all
applied via a transform function returned by ``make_transform``, used by
``WindowParquetDataset`` (see ``regulonado.training.data``).

The output is a Hugging Face Hub-layout Parquet dataset: a ``README.md``
dataset card (written last, as the completion sentinel), ``tracks.parquet``,
and one ``data/`` directory holding every split's Parquet shards
(``data/{split}-NNNNN-of-MMMMM.parquet``). Anyone can ``load_dataset(path)``
or ``load_dataset(path, streaming=True)`` against it, or upload the directory
to the Hub as is; training reads Parquet row groups directly with pyarrow
instead.

Example::

    from regulonado.dataset import build_dataset

    build_dataset(
        "intervals.bed", "genome.fa", "tracks.parquet",
        output_dir="dataset/v1",
        splits={"train": ["train"], "validation": ["valid"]},
        shift_max_bp=128,
        n_extract_threads=16,
    )

    from datasets import load_dataset
    ds = load_dataset("dataset/v1", streaming=True)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd

from regulonado.genomics import read_intervals

if TYPE_CHECKING:
    from datasets import Features

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT = 524_288
_DEFAULT_PRED_BINS = 6_144
_DEFAULT_BIN_SIZE = 32
_DEFAULT_ZSTD_LEVEL = 3
_DEFAULT_ROWS_PER_ROW_GROUP = 1

# Target on-disk size per Parquet shard file. Chosen to keep shard counts in
# the low hundreds while staying comfortably under the ~500 MB the
# HuggingFace/Arrow ecosystem recommends per shard.
_DEFAULT_SHARD_TARGET_MB = 256
# Rough on-disk compression ratio (compressed / uncompressed) for zstd level 3
# with dictionary encoding on, used only to size shards. Measured on a real
# train shard:
# ~0.41 MB compressed vs ~3.5 MB decoded per example, i.e. ratio ~0.12. 0.15 is
# used here to stay conservative (over-estimate the compressed size, i.e.
# under-fill shards rather than overshoot the target).
_PARQUET_COMPRESSION_RATIO = 0.15


def _recommend_shard_size(
    *,
    n_tracks: int,
    stored_n_bins: int,
    stored_context: int,
    target_mb: int,
) -> int:
    """Samples per shard file to hit ~``target_mb`` on disk.

    Estimates the per-sample on-disk footprint from the schema (float32 labels
    + uint8 sequence tokens + small per-row overhead) and divides the target
    byte budget by it. Clamped to at least one sample per shard.
    """
    label_bytes = n_tracks * stored_n_bins * 4
    seq_bytes = stored_context  # uint8 tokens, 1 byte/base
    row_overhead = 128  # interval string + index/local_index + Parquet framing
    per_sample_uncompressed = label_bytes + seq_bytes + row_overhead
    per_sample_on_disk = max(1, int(per_sample_uncompressed * _PARQUET_COMPRESSION_RATIO))

    target_bytes = max(1, target_mb) * 1_000_000
    return max(1, target_bytes // per_sample_on_disk)


def _regulonado_version() -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("regulonado")
    except PackageNotFoundError:
        return "unknown"


DEFAULT_SPLITS: dict[str, list[str]] = {
    "train": ["fold0", "fold1", "fold2", "fold5", "fold6", "fold7"],
    "validation": ["fold4"],
    "test": ["fold3"],
}


# ---------------------------------------------------------------------------
# Scratch staging
# ---------------------------------------------------------------------------

_FASTA_COMPANIONS = (".fai",)  # pyfaidx index; bgzf FASTA is rejected, see _reject_bgzip_fasta

_GZIP_MAGIC = b"\x1f\x8b"


def _reject_bgzip_fasta(fasta_file: str | Path) -> None:
    """Raise if ``fasta_file`` is gzip/bgzf-compressed.

    The Rust FASTA reader (src/fasta.rs) indexes the FASTA using raw,
    uncompressed byte offsets from the ``.fai`` file. Against a bgzipped
    FASTA those offsets address compressed blocks, so every position reads
    back as an unrecognised byte and is written as an all-zero one-hot
    column — the build succeeds and silently produces blank sequence.
    Checking the two-byte gzip magic number is cheap and catches this before
    any staging or Rust extraction happens.
    """
    path = Path(fasta_file)
    try:
        with path.open("rb") as fh:
            magic = fh.read(2)
    except OSError:
        return  # let the normal file-not-found path surface downstream
    if magic == _GZIP_MAGIC:
        raise ValueError(
            f"bgzf-compressed FASTA is not supported: {path}. Decompress it "
            "(e.g. `bgzip -d`) and index the plain FASTA with `samtools faidx`."
        )


def _stage_relative_path(src: Path) -> Path:
    """Return a collision-safe staged relative path for an absolute source."""
    src_abs = src.expanduser().resolve()
    digest = hashlib.blake2b(str(src_abs).encode(), digest_size=8).hexdigest()
    return Path(digest) / src_abs.name


def _rsync_one(src: Path, scratch: Path, companion_suffixes: tuple[str, ...]) -> str:
    """Copy one file (and any companions) to scratch via rsync. Returns staged path."""
    rel = _stage_relative_path(src)
    dest = scratch / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-au", str(src), str(dest)], check=True)
    for suf in companion_suffixes:
        companion = Path(str(src) + suf)
        if companion.exists():
            companion_dest = Path(str(dest) + suf)
            subprocess.run(["rsync", "-au", str(companion), str(companion_dest)], check=True)
    return str(dest)


def _same_filesystem(a: Path, b: Path) -> bool:
    """True if paths ``a`` and ``b`` resolve to the same filesystem (``st_dev``).

    Neither path needs to exist yet — each walks up to its nearest existing
    ancestor before comparing devices, so a not-yet-created scratch or output
    directory can still be classified correctly.
    """

    def _existing_dev(p: Path) -> int:
        p = Path(p).resolve()
        while not p.exists():
            parent = p.parent
            if parent == p:
                raise OSError(f"no existing ancestor found for {p}")
            p = parent
        return p.stat().st_dev

    try:
        return _existing_dev(a) == _existing_dev(b)
    except OSError:
        return False


def _stage_files(
    paths: Sequence[str | Path],
    scratch_dir: str | Path,
    companion_suffixes: tuple[str, ...] = (),
    max_workers: int = 16,
) -> list[str]:
    """Copy files to scratch in parallel, skipping up-to-date files. Returns new paths.

    Uses mtime comparison so re-running on the same node skips copying.
    Companion files (e.g. .fai index) are copied alongside their parent.
    """
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    srcs = [Path(p) for p in paths]
    unique_srcs = list(dict.fromkeys(srcs))
    n = len(unique_srcs)
    workers = min(max_workers, n) if n else 1
    if len(unique_srcs) != len(srcs):
        logger.info(
            f"Staging {n} unique file(s) for {len(srcs)} requested path(s) → {scratch} "
            f"(workers={workers})"
        )
    else:
        logger.info(f"Staging {n} file(s) → {scratch} (workers={workers})")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_rsync_one, src, scratch, companion_suffixes): src for src in unique_srcs
        }
        staged_map: dict[str, str] = {}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            src = futs[fut]
            staged_map[str(src)] = fut.result()
            if done % 50 == 0 or done == n:
                logger.info(f"  staged {done}/{n}")
    # preserve input order
    return [staged_map[str(Path(p))] for p in paths]


# ---------------------------------------------------------------------------
# BigWig helpers
# ---------------------------------------------------------------------------


def _is_minus_strand(path: str) -> bool:
    stem = Path(path).stem.lower()
    return stem.endswith("_minus") or stem.endswith("-minus") or ".minus" in stem


# ---------------------------------------------------------------------------
# Fast-path helpers
# ---------------------------------------------------------------------------


def signal_intervals(
    bed_rows: list[tuple[str, int, int, str]],
    n_pred_bins: int,
    bin_size: int,
    shift_max_bp: int,
) -> list[tuple[str, int, int]]:
    """Compute signal regions (chrom, sig_start, sig_end) for all BED rows."""
    pred_bp = n_pred_bins * bin_size
    out = []
    for chrom, start, end, _ in bed_rows:
        center = (start + end) // 2
        sig_start = max(0, center - pred_bp // 2 - shift_max_bp)
        sig_end = sig_start + pred_bp + 2 * shift_max_bp
        out.append((chrom, sig_start, sig_end))
    return out


def _edge_unsafe_row_indices(
    bed_rows: list[tuple[str, int, int, str]],
    n_pred_bins: int,
    bin_size: int,
    shift_max_bp: int,
) -> list[int]:
    """BED row positions whose signal window would start before contig position 0.

    ``signal_intervals`` clamps such a row's ``sig_start`` to 0 so the Rust
    writers never see a negative coordinate. But the clamped value is what
    both Rust writers then treat as the literal genome coordinate of *label*
    array position 0 (see ``bin_start`` in src/chromosome_scan_writer.rs and
    the equivalent BigWig read in src/sample_batch_writer.rs), while the
    *sequence* side (src/fasta.rs::read_one_hot_sequence) keeps the true,
    unclamped window start and zero-pads instead of clamping. For a row this
    close to position 0 the two arrays end up offset relative to each other
    by up to ``pred_bp // 2 + shift_max_bp`` bins — the fix is to exclude the
    row from every split rather than write it misaligned.

    The tail of the window needs no equivalent treatment: neither side clamps
    the *reference* coordinate there, only how much is copied — a window that
    runs past the contig end is simply zero-padded identically on both sides
    (``bin_end`` is capped in chromosome_scan_writer.rs and the pre-zeroed
    ``labels``/``out`` buffers supply the padding on both the label and
    sequence side).
    """
    pred_bp = n_pred_bins * bin_size
    half_window = pred_bp // 2 + shift_max_bp
    return [
        i for i, (_, start, end, _) in enumerate(bed_rows) if (start + end) // 2 - half_window < 0
    ]


# Filesystem types that are network-backed, and therefore slow enough for Arrow I/O
# that it is worth warning about. Used only for diagnostics.
_NETWORK_FS_TYPES = frozenset(
    {"ceph", "nfs", "nfs4", "lustre", "gpfs", "beegfs", "glusterfs", "cifs", "smb3", "fuse.sshfs"}
)


def _is_remote_fs(path: Path) -> bool:
    """Return True when ``path`` lives on a network filesystem.

    Used only to decide whether to warn that Arrow I/O will cross the network.

    This used to test for the literal prefixes ``/ceph`` and ``/project``, which
    made it a no-op on any machine that did not happen to use those mount points.
    Reading the actual filesystem type from ``/proc/mounts`` works anywhere Linux
    does; on platforms without ``/proc/mounts`` it returns False, which only costs
    a diagnostic message.

    Parameters
    ----------
    path : pathlib.Path
        Path to test. Resolved before matching, so symlinks are followed.

    Returns
    -------
    bool
        True if the longest matching mount point is a known network filesystem.
    """
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False

    resolved = path.resolve()
    best_len = -1
    best_is_network = False
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point, fs_type = parts[1], parts[2]
        try:
            mount_path = Path(mount_point)
            if resolved != mount_path and mount_path not in resolved.parents:
                continue
        except (OSError, ValueError):
            continue
        # Longest matching mount point wins, so a network mount nested under "/"
        # is not masked by the root filesystem entry.
        if len(mount_point) > best_len:
            best_len = len(mount_point)
            best_is_network = fs_type in _NETWORK_FS_TYPES
    return best_is_network


def _write_dataset_card(
    output_dir: Path,
    *,
    features: Features,
    split_counts: dict[str, int],
) -> None:
    """Write ``README.md``, the dataset card that makes ``output_dir`` HF-loadable.

    Uses the same ``DatasetInfosDict``/``MetadataConfigs`` card-building helpers
    ``datasets`` itself uses in ``push_to_hub``, so the emitted YAML front
    matter is exactly what ``load_dataset_builder(output_dir).info.splits`` and
    streaming's ``info.splits`` expect.

    Written *last*, after every split's Parquet shards are published: its
    presence is the build's completion sentinel, so a directory with
    ``data/*.parquet`` but no ``README.md`` means an interrupted build.
    """
    from datasets.info import DatasetInfo, DatasetInfosDict  # noqa: PLC0415
    from datasets.splits import SplitDict, SplitInfo  # noqa: PLC0415
    from datasets.utils.metadata import MetadataConfigs  # noqa: PLC0415
    from huggingface_hub import DatasetCard, DatasetCardData  # noqa: PLC0415

    splits = SplitDict()
    for split, num_examples in split_counts.items():
        splits.add(SplitInfo(name=split, num_examples=num_examples))
    info = DatasetInfo(features=features, splits=splits)

    card_data = DatasetCardData()
    DatasetInfosDict({"default": info}).to_dataset_card_data(card_data)
    MetadataConfigs(
        {
            "default": {
                "data_files": [
                    {"split": split, "path": f"data/{split}-*"} for split in split_counts
                ]
            }
        }
    ).to_dataset_card_data(card_data)
    card = DatasetCard(f"---\n{card_data}\n---\n")
    card.save(output_dir / "README.md")


_STRATEGIES = frozenset({"in_memory", "streaming"})
_BIN_DENOMINATORS = frozenset({"bin_width", "covered_bases"})
_MISSING_BINS = frozenset({"nan", "zero"})


# ---------------------------------------------------------------------------
# Build planning — pure(ish) functions that produce a frozen BuildPlan.
#
# ``build_dataset`` itself (below the plan) is a linear sequence of calls into
# this section plus the writer/publish steps that follow it. No I/O happens
# here beyond reading small metadata (tracks.parquet, the BED file, the FASTA
# .fai, /proc/mounts) and, if requested, staging inputs to scratch — no
# BigWig/FASTA extraction happens until a strategy runner is invoked.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    """Derived (possibly shift-padded) sequence/label array shapes."""

    stored_context: int
    shift_bins: int
    stored_n_bins: int


@dataclass(frozen=True)
class _WriterSettings:
    """Parquet writer knobs after shard sizing."""

    effective_shard_size: int
    effective_write_threads: int


@dataclass(frozen=True)
class BuildPlan:
    """Everything the writer and publish steps need, computed once up front."""

    output_dir: Path
    splits: dict[str, list[str]]
    track_table: pd.DataFrame
    bed_rows: list[tuple[str, int, int, str]]
    signal_regions: list[tuple[str, int, int]]
    active_fasta: str
    active_bw_paths: list[str]
    minus_flags: list[bool]
    n_tracks: int
    bin_size: int
    geometry: _Geometry
    features: Features
    scratch_out: Path
    split_indices: dict[str, list[int]]
    splits_to_build: list[str]
    effective_shard_size: int
    effective_write_threads: int
    zstd_level: int
    rows_per_row_group: int
    edge_dropped_indices: tuple[int, ...]
    edge_dropped_examples: tuple[str, ...]
    bin_denominator: str
    missing_bins: str

    @property
    def binning_kwargs(self) -> dict[str, bool]:
        """Keyword arguments selecting the Rust writers' bin mean and missing-bin value."""
        return {
            "mean_over_covered_bases": self.bin_denominator == "covered_bases",
            "missing_as_nan": self.missing_bins == "nan",
        }


def _validate_build_args(
    *, shift_max_bp: int, bin_size: int, strategy: str, bin_denominator: str, missing_bins: str
) -> None:
    if bin_denominator not in _BIN_DENOMINATORS:
        raise ValueError(
            f"bin_denominator must be one of {sorted(_BIN_DENOMINATORS)}, got {bin_denominator!r}"
        )
    if missing_bins not in _MISSING_BINS:
        raise ValueError(
            f"missing_bins must be one of {sorted(_MISSING_BINS)}, got {missing_bins!r}"
        )
    if shift_max_bp % bin_size != 0:
        raise ValueError(
            f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})"
        )
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be 'in_memory' or 'streaming', got {strategy!r}")


def _compute_geometry(
    *, context_length: int, bin_size: int, n_pred_bins: int, shift_max_bp: int
) -> _Geometry:
    """Pure derivation of the stored (shift-padded) sequence/label shapes."""
    shift_bins = shift_max_bp // bin_size
    return _Geometry(
        stored_context=context_length + 2 * shift_max_bp,
        shift_bins=shift_bins,
        stored_n_bins=n_pred_bins + 2 * shift_bins,
    )


def _load_verified_track_table(track_table: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Read tracks.parquet and verify each included track's on-disk fingerprint.

    Returns the full table (every status, for round-tripping to the output
    tracks.parquet) and the resolved BigWig paths for included tracks in
    ``track_index`` order.
    """
    from regulonado.tracks_table import (  # noqa: PLC0415
        read_track_table,
        require_track_format,
        verify_fingerprint,
    )

    table = read_track_table(track_table)
    require_track_format(table, "bigwig", "dataset build")
    included = table[table["status"] == "included"].sort_values("track_index")
    for _, row in included.iterrows():
        expected = {k: row[k] for k in row.index if k.startswith("fp_") and pd.notna(row[k])}
        problems = verify_fingerprint(row["resolved_path"], expected)
        if problems:
            raise ValueError(
                f"Track {row['track_name']!r} no longer matches its recorded fingerprint "
                f"({row['resolved_path']}): {'; '.join(problems)}. Re-run discovery/assembly."
            )
    return table, included["resolved_path"].tolist()


def _make_scratch_dir() -> Path:
    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    return Path(tempfile.mkdtemp(prefix="regulonado-build-", dir=scratch_root))


def _stage_inputs_if_requested(
    stage_to_scratch: bool,
    fasta_file: str | Path,
    bed_file: str | Path,
    bw_paths: list[str],
    scratch_out: Path,
) -> tuple[str, list[str]]:
    """Optionally copy FASTA/BED/BigWig inputs to scratch; returns active paths."""
    if not stage_to_scratch:
        return str(fasta_file), bw_paths
    stage_dir = scratch_out / "stage"
    logger.info(f"Staging source files to {stage_dir}")
    active_fasta = _stage_files([fasta_file], stage_dir, _FASTA_COMPANIONS)[0]
    _stage_files([bed_file], stage_dir)
    active_bw_paths = _stage_files(bw_paths, stage_dir)
    return active_fasta, active_bw_paths


def _load_bed_rows(bed_file: str | Path) -> list[tuple[str, int, int, str]]:
    bed_frame = read_intervals(bed_file)
    if "name" in bed_frame.columns:
        return [
            (str(chrom), int(start), int(end), str(name))
            for chrom, start, end, name in bed_frame[["chrom", "start", "end", "name"]].itertuples(
                index=False, name=None
            )
        ]
    return [
        (str(chrom), int(start), int(end), "")
        for chrom, start, end in bed_frame[["chrom", "start", "end"]].itertuples(
            index=False, name=None
        )
    ]


def _log_remote_fs_warnings(scratch_out: Path, output_dir: Path) -> None:
    if _is_remote_fs(scratch_out):
        logger.warning(
            f"scratch_out resolves to {str(scratch_out.resolve())!r} — "
            f"Arrow I/O will hit remote storage"
        )
    if _is_remote_fs(output_dir):
        logger.info(
            f"output_dir resolves to {str(output_dir.resolve())!r}; "
            f"only the final publish step should hit remote storage"
        )


def _log_ram_estimate(
    *,
    n_tracks: int,
    bin_size: int,
    geometry: _Geometry,
    effective_write_threads: int,
    n_extract_threads: int,
    active_fasta: str,
    bed_rows: list[tuple[str, int, int, str]],
) -> None:
    """Log an approximate peak-RAM estimate for the chosen thread settings.

    The Parquet writer holds at most one example per writer thread at a time
    (rows are appended one at a time; see chromosome_scan_writer.rs), so
    writer RAM scales with ``effective_write_threads``, not a batch size.
    """
    stored_context, stored_n_bins = geometry.stored_context, geometry.stored_n_bins
    label_example_gb = n_tracks * stored_n_bins * 4 / 1e9
    seq_example_gb = stored_context / 1e9  # uint8 tokens, 1 byte/base
    writer_peak_gb = effective_write_threads * (label_example_gb + seq_example_gb)

    chrom_lengths: dict[str, int] = {}
    fai_path = Path(str(active_fasta) + ".fai")
    if fai_path.exists():
        for line in fai_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                chrom_lengths[parts[0]] = int(parts[1])
    if chrom_lengths:
        used_chroms = {row[0] for row in bed_rows}
        largest_chrom_bins = max(
            (chrom_lengths.get(chrom, 0) // bin_size for chrom in used_chroms), default=0
        )
        chrom_matrix_gb = n_tracks * largest_chrom_bins * 4 / 1e9
        binning_scratch_gb = n_extract_threads * largest_chrom_bins * (8 + 8) / 1e9
    else:
        chrom_matrix_gb = 0.0
        binning_scratch_gb = 0.0
    in_memory_peak_gb = chrom_matrix_gb + binning_scratch_gb + writer_peak_gb
    logger.info(
        f"Approx per-example RAM: labels={label_example_gb * 1000:.2f} MB, "
        f"in-memory matrix≈{chrom_matrix_gb:.1f} GB, "
        f"binning scratch≈{binning_scratch_gb:.1f} GB, "
        f"writer peak≈{writer_peak_gb * 1000:.1f} MB, "
        f"combined in-memory peak≈{in_memory_peak_gb:.1f} GB "
        f"(n_extract_threads={n_extract_threads}, write_threads={effective_write_threads})"
    )


def _plan_writer_settings(
    *,
    n_tracks: int,
    bin_size: int,
    geometry: _Geometry,
    shard_size: int | None,
    shard_target_mb: int,
    write_threads: int | None,
    n_extract_threads: int,
    active_fasta: str,
    bed_rows: list[tuple[str, int, int, str]],
) -> _WriterSettings:
    """Size shards and writer thread count, and log a RAM estimate."""
    stored_context, stored_n_bins = geometry.stored_context, geometry.stored_n_bins

    # An explicit shard_size wins; otherwise derive it from shard_target_mb.
    if shard_size is not None:
        effective_shard_size = max(1, shard_size)
    else:
        effective_shard_size = _recommend_shard_size(
            n_tracks=n_tracks,
            stored_n_bins=stored_n_bins,
            stored_context=stored_context,
            target_mb=shard_target_mb,
        )
    logger.info(
        f"Shard sizing: {effective_shard_size} samples/shard, "
        f"target≈{shard_target_mb} MB on disk (zstd)"
    )

    effective_write_threads = 4 if write_threads is None else max(1, write_threads)
    _log_ram_estimate(
        n_tracks=n_tracks,
        bin_size=bin_size,
        geometry=geometry,
        effective_write_threads=effective_write_threads,
        n_extract_threads=n_extract_threads,
        active_fasta=active_fasta,
        bed_rows=bed_rows,
    )
    return _WriterSettings(effective_shard_size, effective_write_threads)


def _build_features(n_tracks: int, geometry: _Geometry) -> Features:
    from datasets import Features, List, Value  # noqa: PLC0415

    return Features(
        {
            "sequence_tokens": List(Value("uint8"), length=geometry.stored_context),
            "signal": List(List(Value("float32"), length=geometry.stored_n_bins), length=n_tracks),
            "interval": Value(dtype="string"),
            "index": Value(dtype="int64"),
            "local_index": Value(dtype="int64"),
        }
    )


def _compute_split_indices(
    bed_rows: list[tuple[str, int, int, str]],
    splits: dict[str, list[str]],
    chrom_filter: list[str] | None,
    output_dir: Path,
    overwrite: bool,
    unsafe_indices: set[int],
) -> tuple[dict[str, list[int]], list[str]]:
    """Row indices per split, skipping already-built splits and edge-unsafe rows.

    Rows in ``unsafe_indices`` (see ``_edge_unsafe_row_indices``) are dropped
    from every split's index list — the drop is logged (with a count and
    examples) by the caller; the build always proceeds with what remains.
    """
    n_all_samples = len(bed_rows)
    chrom_filter_set = set(chrom_filter) if chrom_filter else None
    data_dir = output_dir / "data"

    split_indices: dict[str, list[int]] = {}
    splits_to_build: list[str] = []
    for split, folds in splits.items():
        if not overwrite and any(data_dir.glob(f"{split}-*.parquet")):
            continue
        if folds:
            folds_set = set(folds)
            idx = [i for i, r in enumerate(bed_rows) if r[3] in folds_set]
        else:
            idx = list(range(n_all_samples))
        if chrom_filter_set is not None:
            idx = [i for i in idx if bed_rows[i][0] in chrom_filter_set]

        requested = len(idx)
        idx = [i for i in idx if i not in unsafe_indices]
        dropped = requested - len(idx)

        split_indices[split] = idx
        splits_to_build.append(split)
        chrom_note = f", filtered to chroms {sorted(chrom_filter_set)}" if chrom_filter_set else ""
        logger.info(f"Split '{split}': {len(idx)} samples ({dropped} edge-dropped{chrom_note})")
    return split_indices, splits_to_build


def _plan_build(
    *,
    bed_file: Path,
    fasta_file: str | Path,
    track_table: str | Path,
    output_dir: Path,
    splits: dict[str, list[str]],
    context_length: int,
    bin_size: int,
    n_pred_bins: int,
    shift_max_bp: int,
    stage_to_scratch: bool,
    overwrite: bool,
    chrom_filter: list[str] | None,
    shard_size: int | None,
    shard_target_mb: int,
    zstd_level: int,
    rows_per_row_group: int,
    write_threads: int | None,
    n_extract_threads: int,
    bin_denominator: str,
    missing_bins: str,
) -> BuildPlan:
    """Resolve every path, shape and row index the writer/publish steps need."""
    table, bw_paths = _load_verified_track_table(track_table)
    n_tracks = len(bw_paths)
    geometry = _compute_geometry(
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
    )

    scratch_out = _make_scratch_dir()
    active_fasta, active_bw_paths = _stage_inputs_if_requested(
        stage_to_scratch, fasta_file, bed_file, bw_paths, scratch_out
    )

    bed_rows = _load_bed_rows(bed_file)
    edge_dropped = _edge_unsafe_row_indices(bed_rows, n_pred_bins, bin_size, shift_max_bp)
    edge_dropped_examples = tuple(
        f"{bed_rows[i][0]}:{bed_rows[i][1]}-{bed_rows[i][2]}" for i in edge_dropped[:5]
    )
    if edge_dropped:
        logger.warning(
            f"Dropping {len(edge_dropped)} BED row(s) whose signal window starts before "
            f"contig position 0 (labels would misalign with sequence there); examples: "
            f"{', '.join(edge_dropped_examples)}"
        )
    signal_regions = signal_intervals(bed_rows, n_pred_bins, bin_size, shift_max_bp)

    logger.info(
        f"Resolved inputs: {n_tracks} tracks, {len(bed_rows)} BED rows "
        f"({len(edge_dropped)} edge-dropped), stored_context={geometry.stored_context}, "
        f"stored_n_bins={geometry.stored_n_bins}, scratch_out={scratch_out.resolve()}"
    )
    _log_remote_fs_warnings(scratch_out, output_dir)

    writer = _plan_writer_settings(
        n_tracks=n_tracks,
        bin_size=bin_size,
        geometry=geometry,
        shard_size=shard_size,
        shard_target_mb=shard_target_mb,
        write_threads=write_threads,
        n_extract_threads=n_extract_threads,
        active_fasta=active_fasta,
        bed_rows=bed_rows,
    )
    features = _build_features(n_tracks, geometry)

    split_indices, splits_to_build = _compute_split_indices(
        bed_rows, splits, chrom_filter, output_dir, overwrite, set(edge_dropped)
    )

    return BuildPlan(
        output_dir=output_dir,
        splits=splits,
        track_table=table,
        bed_rows=bed_rows,
        signal_regions=signal_regions,
        active_fasta=active_fasta,
        active_bw_paths=active_bw_paths,
        minus_flags=[_is_minus_strand(p) for p in active_bw_paths],
        n_tracks=n_tracks,
        bin_size=bin_size,
        geometry=geometry,
        features=features,
        scratch_out=scratch_out,
        split_indices=split_indices,
        splits_to_build=splits_to_build,
        effective_shard_size=writer.effective_shard_size,
        effective_write_threads=writer.effective_write_threads,
        zstd_level=zstd_level,
        rows_per_row_group=rows_per_row_group,
        edge_dropped_indices=tuple(edge_dropped),
        edge_dropped_examples=edge_dropped_examples,
        bin_denominator=bin_denominator,
        missing_bins=missing_bins,
    )


# ---------------------------------------------------------------------------
# Strategy runners — extract from BigWig/FASTA and write Parquet shards to
# scratch. Both accept a BuildPlan and iterate only ``splits_to_build``,
# writing into one shared ``scratch_out / "data"`` directory (all splits'
# shards side by side, matching the published HF-Hub layout).
# ---------------------------------------------------------------------------


def _run_in_memory_strategy(
    plan: BuildPlan,
    *,
    n_extract_threads: int,
    profile: bool,
) -> dict[str, int]:
    """Shared chromosome-pass strategy: one scan per chromosome, shared by all splits."""
    from regulonado._rs import (
        write_parquet_splits_chrom_pass,  # type: ignore[import]  # noqa: PLC0415
    )

    if not plan.splits_to_build:
        return {}

    scratch_data = plan.scratch_out / "data"
    scratch_data.mkdir(parents=True, exist_ok=True)

    split_names = list(plan.splits_to_build)
    split_sample_indices = [plan.split_indices[split] for split in split_names]
    logger.info(
        f"Queueing {split_names} shard(s) [in_memory shared-scan]: "
        f"{plan.n_tracks} tracks, stored_context={plan.geometry.stored_context} bp, "
        f"stored_n_bins={plan.geometry.stored_n_bins}, "
        f"shard_size={plan.effective_shard_size}, zstd_level={plan.zstd_level}, "
        f"rows_per_row_group={plan.rows_per_row_group}, "
        f"write_threads={plan.effective_write_threads}"
    )

    row_counts = write_parquet_splits_chrom_pass(
        plan.active_bw_paths,
        plan.minus_flags,
        plan.signal_regions,
        split_names,
        str(scratch_data),
        split_sample_indices,
        plan.bed_rows,
        plan.active_fasta,
        plan.geometry.stored_n_bins,
        plan.geometry.stored_context,
        plan.bin_size,
        json.dumps(plan.features.to_dict()),
        shard_size=plan.effective_shard_size,
        rows_per_row_group=plan.rows_per_row_group,
        zstd_level=plan.zstd_level,
        n_threads=n_extract_threads,
        write_threads=plan.effective_write_threads,
        profile=profile,
        **plan.binning_kwargs,
    )

    for split in split_names:
        n_files = len(list(scratch_data.glob(f"{split}-*-of-*.parquet")))
        if n_files == 0:
            raise RuntimeError(
                f"in_memory produced no shards for split '{split}' in {scratch_data}; "
                f"check that the FASTA contains the BED chromosomes"
            )
        logger.info(f"Parquet shard(s) for '{split}' written ({n_files} file(s))")
    return row_counts


def _run_streaming_strategy(
    plan: BuildPlan,
    *,
    n_extract_threads: int,
) -> dict[str, int]:
    """Per-split direct-BigWig strategy: bounded memory, random seeks."""
    from regulonado._rs import (
        write_parquet_split_from_bigwigs,  # type: ignore[import]  # noqa: PLC0415
    )

    scratch_data = plan.scratch_out / "data"
    scratch_data.mkdir(parents=True, exist_ok=True)
    hf_features_json = json.dumps(plan.features.to_dict())

    row_counts: dict[str, int] = {}
    for split in plan.splits_to_build:
        sample_indices = plan.split_indices[split]
        logger.info(
            f"Writing '{split}' shard [streaming]: {len(sample_indices)} samples, "
            f"{plan.n_tracks} tracks, stored_context={plan.geometry.stored_context} bp, "
            f"stored_n_bins={plan.geometry.stored_n_bins}, zstd_level={plan.zstd_level}"
        )
        t_split = time.perf_counter()
        shard_path = scratch_data / f"{split}-00000-of-00001.parquet"
        n_rows = write_parquet_split_from_bigwigs(
            plan.active_bw_paths,
            plan.minus_flags,
            plan.signal_regions,
            str(shard_path),
            sample_indices,
            plan.bed_rows,
            plan.active_fasta,
            plan.geometry.stored_n_bins,
            plan.geometry.stored_context,
            hf_features_json,
            rows_per_row_group=plan.rows_per_row_group,
            zstd_level=plan.zstd_level,
            n_threads=n_extract_threads,
            **plan.binning_kwargs,
        )
        logger.info(
            f"Parquet shard for '{split}' written in "
            f"{time.perf_counter() - t_split:.1f}s ({n_rows} rows)"
        )
        row_counts[split] = n_rows
    return row_counts


def _run_strategy(
    plan: BuildPlan,
    *,
    strategy: str,
    n_extract_threads: int,
    profile: bool,
) -> dict[str, int]:
    t_total = time.perf_counter()
    if strategy == "in_memory":
        row_counts = _run_in_memory_strategy(
            plan, n_extract_threads=n_extract_threads, profile=profile
        )
    else:
        row_counts = _run_streaming_strategy(plan, n_extract_threads=n_extract_threads)
    logger.info(f"Parquet writing completed in {time.perf_counter() - t_total:.1f}s")
    return row_counts


def _publish_split_shards(scratch_data: Path, data_dir: Path, split: str) -> None:
    """Publish one split's freshly written shard files into the dataset's data dir.

    One ``data/`` directory now holds every split's shards side by side, so
    publishing can no longer swap a whole per-split directory into place (that
    would delete other splits' files). Instead, only ``split``'s
    ``{split}-*.parquet`` files are touched: any of the split's stale shards
    already in ``data_dir`` (e.g. left over from a previous build with a
    different shard count) are removed, and the freshly written ones are moved
    in — same-filesystem via `os.replace` (atomic, no bytes copied),
    cross-filesystem via copy-then-delete.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob(f"{split}-*.parquet"):
        stale.unlink()
    new_shards = sorted(scratch_data.glob(f"{split}-*.parquet"))
    if not new_shards:
        raise RuntimeError(f"no Parquet shards produced for split {split!r} in {scratch_data}")
    same_fs = _same_filesystem(scratch_data, data_dir)
    for shard in new_shards:
        dest = data_dir / shard.name
        if same_fs:
            os.replace(shard, dest)
        else:
            shutil.copyfile(shard, dest)
            shard.unlink()


def _publish_splits(plan: BuildPlan) -> None:
    logger.info(f"Publishing rebuilt splits to {plan.output_dir}")
    t_publish = time.perf_counter()
    scratch_data = plan.scratch_out / "data"
    data_dir = plan.output_dir / "data"
    for split in plan.splits_to_build:
        _publish_split_shards(scratch_data, data_dir, split)
    logger.info(f"Publication completed in {time.perf_counter() - t_publish:.1f}s")


def _split_row_count(data_dir: Path, split: str) -> int:
    """Sum ``num_rows`` across an existing split's Parquet shard footers."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    return sum(
        pq.ParquetFile(shard).metadata.num_rows
        for shard in sorted(data_dir.glob(f"{split}-*.parquet"))
    )


def _write_output_track_table(
    plan: BuildPlan,
    *,
    bed_file: str | Path,
    fasta_file: str | Path,
    context_length: int,
    n_pred_bins: int,
    shift_max_bp: int,
    strategy: str,
) -> None:
    from datetime import datetime, timezone  # noqa: PLC0415

    from regulonado.tracks_table import write_track_table  # noqa: PLC0415

    write_track_table(
        plan.track_table,
        plan.output_dir / "tracks.parquet",
        bed_file=str(bed_file),
        fasta_file=str(fasta_file),
        context_length=context_length,
        bin_size=plan.bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        splits=plan.splits,
        build_strategy=strategy,
        write_threads=plan.effective_write_threads,
        zstd_level=plan.zstd_level,
        rows_per_row_group=plan.rows_per_row_group,
        created_at=datetime.now(timezone.utc).isoformat(),
        regulonado_version=_regulonado_version(),
        command=" ".join(sys.argv),
        edge_dropped_rows=len(plan.edge_dropped_indices),
        edge_dropped_examples=list(plan.edge_dropped_examples),
        bin_denominator=plan.bin_denominator,
        missing_bins=plan.missing_bins,
    )


def build_dataset(
    bed_file: str | Path,
    fasta_file: str | Path,
    track_table: str | Path,
    output_dir: str | Path,
    *,
    splits: dict[str, list[str]] | None = None,
    context_length: int = _DEFAULT_CONTEXT,
    bin_size: int = _DEFAULT_BIN_SIZE,
    n_pred_bins: int = _DEFAULT_PRED_BINS,
    shift_max_bp: int = 0,
    n_extract_threads: int = 32,
    shard_size: int | None = None,
    shard_target_mb: int = _DEFAULT_SHARD_TARGET_MB,
    zstd_level: int = _DEFAULT_ZSTD_LEVEL,
    write_threads: int | None = None,
    rows_per_row_group: int = _DEFAULT_ROWS_PER_ROW_GROUP,
    stage_to_scratch: bool = False,
    overwrite: bool = False,
    profile: bool = False,
    strategy: str = "in_memory",
    chrom_filter: list[str] | None = None,
    bin_denominator: str = "bin_width",
    missing_bins: str = "nan",
) -> None:
    """Fast low-scratch dataset build using the Rust extension.

    Writes a Hugging Face Hub-layout Parquet dataset: ``output_dir/README.md``
    (the dataset card, written last as the completion sentinel),
    ``output_dir/tracks.parquet``, and ``output_dir/data/`` holding every
    split's Parquet shards (``data/{split}-NNNNN-of-MMMMM.parquet``). Load it
    with ``datasets.load_dataset(output_dir)`` or
    ``load_dataset(output_dir, streaming=True)``. Peak scratch is the current
    Parquet output plus one in-memory example per writer thread, not a full
    dense ``(tracks, samples, bins)`` signal file.

    Track identity, dedupe and QC are already settled by the time this runs —
    ``track_table`` is ``tracks.parquet`` from ``regulonado tracks assemble``,
    the *only* place a track list is named. This function reads it, verifies
    each included track's fingerprint against disk, and writes the same table
    (with build-time scalars merged in) to ``output_dir/tracks.parquet``.

    Parameters
    ----------
    strategy : {"in_memory", "streaming"}
        - "in_memory" (default): decodes each chromosome's binned signal for
          every track into RAM once, then slices every sample window from
          that shared scan — one scan shared by all splits. Shards are
          ordered chrom-major (largest chromosome first); each chromosome
          yields ``ceil(samples / shard_size)`` shard files sized to
          ~``shard_target_mb`` on disk. Rows within each shard are in
          original BED order. Sequential reads, higher memory (the
          per-example RAM estimate logged below applies to this strategy);
          ~10× fewer BigWig seeks than "streaming".
        - "streaming": reads each sample window's interval from every
          BigWig, split by split. Bounded memory, random seeks; kept mainly
          as the parity reference for "in_memory".
    chrom_filter : list[str] | None
        If given, restrict each split to BED rows on these chromosomes.
        ``bed_rows`` is *not* renumbered — the ``index`` column on every
        output row remains the absolute row position in the input BED
        file. Useful for smoke tests on a single chromosome.
    bin_denominator : {"bin_width", "covered_bases"}
        - "bin_width" (default): a bin's value is its summed signal over its
          in-contig width, so bases with no BigWig record count as zero. Right
          for coverage BigWigs, which often omit zero-coverage stretches.
        - "covered_bases": divide by the bases that have a record instead
          (pyBigWig ``stats(exact=True)`` semantics). Inflates sparse bins when
          zero records are omitted; identical to "bin_width" when they are not.
    missing_bins : {"nan", "zero"}
        Value stored for bins with no data — past the contig end (window
        padding) or wholly NaN-valued in the BigWig. "nan" (default) lets
        training mask them (``data.mask_missing``); "zero" stores 0.0, which
        trains them as observed zero signal.

    Rows whose signal window would start before contig position 0 are
    dropped from every split (see ``_edge_unsafe_row_indices``) rather than
    written misaligned — never fails the build. The count and a few examples
    are logged as a warning and recorded in the output ``tracks.parquet`` as
    ``edge_dropped_rows`` / ``edge_dropped_examples``.
    """
    _validate_build_args(
        shift_max_bp=shift_max_bp,
        bin_size=bin_size,
        strategy=strategy,
        bin_denominator=bin_denominator,
        missing_bins=missing_bins,
    )
    _reject_bgzip_fasta(fasta_file)

    bed_file = Path(bed_file)
    output_dir = Path(output_dir)
    plan = _plan_build(
        bed_file=bed_file,
        fasta_file=fasta_file,
        track_table=track_table,
        output_dir=output_dir,
        splits=splits or DEFAULT_SPLITS,
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        stage_to_scratch=stage_to_scratch,
        overwrite=overwrite,
        chrom_filter=chrom_filter,
        shard_size=shard_size,
        shard_target_mb=shard_target_mb,
        zstd_level=zstd_level,
        rows_per_row_group=rows_per_row_group,
        write_threads=write_threads,
        n_extract_threads=n_extract_threads,
        bin_denominator=bin_denominator,
        missing_bins=missing_bins,
    )

    if plan.splits_to_build:
        row_counts = _run_strategy(
            plan, strategy=strategy, n_extract_threads=n_extract_threads, profile=profile
        )
        _publish_splits(plan)
    else:
        logger.info("All splits already exist; skipping rebuild")
        row_counts = {}
    shutil.rmtree(plan.scratch_out, ignore_errors=True)

    _write_output_track_table(
        plan,
        bed_file=bed_file,
        fasta_file=fasta_file,
        context_length=context_length,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        strategy=strategy,
    )

    data_dir = output_dir / "data"
    all_row_counts = {
        split: row_counts.get(split, _split_row_count(data_dir, split)) for split in plan.splits
    }
    _write_dataset_card(output_dir, features=plan.features, split_counts=all_row_counts)

    logger.info(f"Dataset saved to {output_dir}")
    return None


# ---------------------------------------------------------------------------
# Read-time transform
# ---------------------------------------------------------------------------


def build_rc_permutation(
    track_records: list[dict],
    pairing_fields: tuple[str, ...] = (
        "condition_id",
        "source_id",
        "cell_line_id",
        "assay_type_id",
        "target_id",
    ),
) -> np.ndarray | None:
    """Return per-track permutation that swaps paired +/- strand channels for RC aug."""
    if not track_records:
        return None
    records = sorted(track_records, key=lambda r: int(r.get("track_index", len(track_records))))
    n = len(records)
    perm = np.arange(n, dtype=np.int64)

    available = [f for f in pairing_fields if any(r.get(f) is not None for r in records)]
    groups: dict[tuple, dict[str, list[int]]] = {}
    for i, rec in enumerate(records):
        strand = str(rec.get("strand", "")).strip().lower()
        if strand not in {"+", "plus", "forward", "1", "-", "minus", "reverse", "-1"}:
            continue
        sign = "+" if strand in {"+", "plus", "forward", "1"} else "-"
        key = tuple(rec.get(f) for f in available)
        groups.setdefault(key, {"+": [], "-": []})[sign].append(i)

    for sg in groups.values():
        for p, m in zip(sorted(sg["+"]), sorted(sg["-"])):
            perm[p], perm[m] = m, p

    return perm if not np.all(perm == np.arange(n)) else None


def transform_signal(
    signal: np.ndarray,
    scale_factors: np.ndarray,
    clip_soft: np.ndarray | float,
    clip_hard: np.ndarray | float,
    background: np.ndarray | float | None = None,
    *,
    apply_scale: bool = True,
    apply_squash: bool = True,
    apply_clip: bool = True,
    keep_missing: bool = False,
) -> np.ndarray:
    """Apply scale → clip → squash to a (T, L) signal array.

    The three transforms are applied in order:

    1. **Scale** (``apply_scale``): multiply by per-track scale factors that convert the
       stored normalised BigWig signal (RPKM / coverage) to *raw read counts*.  Scale
       factors are inferred at dataset-build time as ``library_size * bin_size_kb / 1e6``.
    2. **Clip** (``apply_clip``): hard-ceiling at per-track ``clip_hard`` thresholds (in
       raw-count space before squash).
    3. **Squash** (``apply_squash``): Borzoi-style ``(x+1)^0.75 - 1`` that compresses the
       raw-count dynamic range while preserving monotonicity.  Above ``clip_soft`` a softer
       sqrt compression is applied.

    Missing bins (NaN) become 0.0, or stay NaN when ``keep_missing`` so training can
    mask them.

    Mirrors the per-sample signal transform inside ``make_transform``.
    Returns float32.
    """
    sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1, 1)
    cs = np.broadcast_to(np.asarray(clip_soft, dtype=np.float32), (sf.shape[0],)).reshape(-1, 1)
    ch = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (sf.shape[0],)).reshape(-1, 1)

    out = np.asarray(signal, dtype=np.float32).copy()
    missing = np.isnan(out) if keep_missing else None
    np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.maximum(out, 0.0, out=out)
    if apply_scale:
        if background is not None:
            bg = np.asarray(background, dtype=np.float32).reshape(-1, 1)
            out -= bg
            np.maximum(out, 0.0, out=out)
        out *= sf
    if apply_clip:
        np.minimum(out, ch, out=out)
    if apply_squash:
        out = (np.power(out + 1.0, 0.75) - 1.0).astype(np.float32, copy=False)
        if apply_clip:
            cs_sq = (np.power(cs + 1.0, 0.75) - 1.0).astype(np.float32)
            mask = out > cs_sq
            if mask.any():
                out = np.where(
                    mask,
                    cs_sq - 1.0 + np.sqrt(np.maximum(out - cs_sq + 1.0, 0.0)),
                    out,
                ).astype(np.float32)
    if missing is not None and missing.any():
        out[missing] = np.nan
    return out


def count_labels(
    signal: np.ndarray,
    count_factors: np.ndarray,
    *,
    clip_hard: np.ndarray | float | None = None,
    keep_missing: bool = True,
) -> np.ndarray:
    """Convert stored mean-coverage signal ``(T, L)`` to count units, per track.

    No background subtraction or squashing: the ``data.label_space: counts`` path
    (see ``regulonado.training.label_space``). Order: zero negatives -> multiply by
    ``count_factors`` -> apply ``clip_hard`` (if given) as an absolute ceiling.
    Negative and infinite values are zeroed; missing bins (NaN) stay NaN when
    ``keep_missing``, else become 0.0.

    ``clip_hard``: per-track hard ceiling in STORED MEAN-COVERAGE units (i.e.
    ``tracks.parquet``'s ``clip_hard_counts`` field — the BigWig's own unit, same as
    ``signal`` before the ``count_factors`` multiply). Converted into count-label
    units by multiplying by ``count_factors`` (exact: a quantile-derived threshold
    commutes with the same positive linear scaling applied to the signal it was
    computed from), then applied as a ceiling. ``None`` disables clipping.

    No soft clip: ``transform_signal``'s soft clip is a sqrt compression defined only
    in squashed space, which this counts path never enters, so there is nothing for
    a "soft clip" to mean here. ``clip_soft_counts`` is stored in ``tracks.parquet``
    for completeness (and possible future use) but this function does not apply it —
    do not invent a compression for it.
    """
    out = np.asarray(signal, dtype=np.float32).copy()
    missing = np.isnan(out)
    np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.maximum(out, 0.0, out=out)
    cf = np.asarray(count_factors, dtype=np.float32).reshape(-1, 1)
    out *= cf
    if clip_hard is not None:
        ceiling = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (out.shape[0],))
        np.minimum(out, ceiling.reshape(-1, 1) * cf, out=out)
    if keep_missing and missing.any():
        out[missing] = np.nan
    return out


def inverse_transform_signal(
    signal: np.ndarray,
    scale_factors: np.ndarray | None = None,
    background: np.ndarray | float | None = None,
    *,
    apply_squash: bool = True,
    apply_scale: bool = True,
) -> np.ndarray:
    """Reverse the squash and/or scale applied by ``transform_signal`` / ``make_transform``.

    Transforms are reversed in the opposite order to ``transform_signal``:

    1. **Unsquash** (``apply_squash``): ``(x + 1)^(4/3) - 1`` (inverse of ``(x+1)^0.75 - 1``).
    2. **Unscale** (``apply_scale``): divide by per-track scale factors to convert raw read
       counts back to the original normalised BigWig signal (RPKM / coverage).

    After inversion the signal is in the same units as the original BigWig (typically RPKM).
    To obtain *raw counts* stop after the unsquash step (pass ``apply_scale=False``).

    Args:
        signal: (T, L) or (B, T, L) array in transformed (squashed/scaled) space.
        scale_factors: Per-track scale factors, shape (T,). Required when apply_scale=True.
    """
    out = np.maximum(np.asarray(signal, dtype=np.float32), 0.0)
    if apply_squash:
        out = np.power(out + 1.0, 4.0 / 3.0) - 1.0
        np.maximum(out, 0.0, out=out)
    if apply_scale and scale_factors is not None:
        sf = np.asarray(scale_factors, dtype=np.float32)
        # Reshape to broadcast over last two dims regardless of batch dim.
        sf = sf.reshape(*([1] * (out.ndim - 2)), -1, 1)
        out = out / np.maximum(sf, 1e-8)
        if background is not None:
            bg = np.asarray(background, dtype=np.float32)
            bg = bg.reshape(*([1] * (out.ndim - 2)), -1, 1)
            out = out + bg
    return out


# Reverse-complement lookup for token bases: A0 C1 G2 T3, 4 = N/pad (maps to itself).
# Applied to an already-reversed token array, so index i holds the complement of base i.
_COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def make_transform(
    scale_factors: np.ndarray,
    clip_soft: np.ndarray | float,
    clip_hard: np.ndarray | float,
    background: np.ndarray | float | None = None,
    *,
    apply_scale: bool = True,
    apply_squash: bool = True,
    apply_clip: bool = True,
    enable_rc_aug: bool = False,
    rc_permutation: np.ndarray | None = None,
    shift_max_bins: int = 0,
    context_length: int = _DEFAULT_CONTEXT,
    n_pred_bins: int = _DEFAULT_PRED_BINS,
    bin_size: int = _DEFAULT_BIN_SIZE,
    center_crop: bool = False,
    label_space: str = "transformed",
    count_factors: np.ndarray | None = None,
    mask_missing: bool = True,
) -> Callable[[dict], dict]:
    """Return a per-example transform for ``WindowParquetDataset``.

    Applied to one example, keyed by the stored Parquet columns
    ``sequence_tokens`` (uint8, shape ``(stored_context,)``, A0 C1 G2 T3, 4=N/pad) and
    ``signal`` (float32, shape ``(n_tracks, stored_n_bins)``):
        1. Shift crop: random offset when center_crop=False, center offset when center_crop=True.
           Always applied when shift_max_bins > 0.
        2. RC augmentation    (if enable_rc_aug)
        3. Signal transform   (scale → squash → clip) for ``label_space="transformed"``;
           conversion to count units (``count_labels``) for ``label_space="counts"``,
           which ignores the scale/squash/background arguments and applies only
           ``clip_hard`` (converted to count-label units), gated on ``apply_clip`` —
           ``clip_soft`` has no meaning off the squash path and is never applied here

    Writes ``input_ids`` (uint8 tokens; one-hot encoding happens on the GPU in
    ``RegulonadoModel.forward``) and ``labels`` (float32) into the example.

    Args:
        scale_factors: Per-track scale factors, shape (T,).
        clip_soft: Per-track soft clip thresholds (in scaled space), shape (T,) or scalar.
        clip_hard: Per-track hard clip thresholds (in scaled space), shape (T,) or scalar.
        shift_max_bins: Maximum shift in bins; must match what the dataset was built with.
        context_length: Target sequence length after cropping.
        n_pred_bins: Target number of signal bins after cropping.
        bin_size: Sequence positions per signal bin.
        enable_rc_aug: Apply random reverse-complement augmentation.
        rc_permutation: Per-track permutation for swapping strand pairs on RC aug.
        center_crop: If True, always crop from the center (s=shift_max_bins). Use for eval/test.
        label_space: "transformed" or "counts" (see ``regulonado.training.label_space``).
        count_factors: Per-track mean-coverage → count-unit factors; required for "counts".
        mask_missing: Keep missing bins as NaN for the loss to mask, instead of 0.0.
    """
    if label_space not in {"transformed", "counts"}:
        raise ValueError(f"label_space must be 'transformed' or 'counts', got {label_space!r}")
    if label_space == "counts" and count_factors is None:
        raise ValueError("label_space='counts' requires count_factors")
    sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1)
    n_tracks = sf.size
    cs = np.broadcast_to(np.asarray(clip_soft, dtype=np.float32), (n_tracks,)).copy()
    ch = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (n_tracks,)).copy()
    bg = (
        None
        if background is None
        else np.broadcast_to(np.asarray(background, dtype=np.float32), (n_tracks,)).copy()
    )
    cf = None if count_factors is None else np.asarray(count_factors, dtype=np.float32).reshape(-1)

    def transform_example(example: dict) -> dict:
        example = dict(example)
        seq = np.asarray(example.pop("sequence_tokens"), dtype=np.uint8)
        sig = np.asarray(example.pop("signal"), dtype=np.float32)
        _sf, _cs, _ch, _bg, _cf = sf, cs, ch, bg, cf

        # --- shift crop (always applied when shift buffer was stored)
        if shift_max_bins > 0:
            s = shift_max_bins if center_crop else int(np.random.randint(0, 2 * shift_max_bins + 1))
            seq = seq[s * bin_size : s * bin_size + context_length]
            sig = sig[:, s : s + n_pred_bins]

        # --- RC augmentation
        if enable_rc_aug and np.random.rand() < 0.5:
            seq = _COMPLEMENT[seq[::-1]]
            sig = np.flip(sig, axis=-1)
            if rc_permutation is not None:
                sig = np.take(sig, rc_permutation, axis=-2)
                _sf = sf[rc_permutation]
                _cs = cs[rc_permutation]
                _ch = ch[rc_permutation]
                _bg = None if bg is None else bg[rc_permutation]
                _cf = None if cf is None else cf[rc_permutation]
            sig = sig.copy()

        example["input_ids"] = seq.astype(np.uint8, copy=False)
        if label_space == "counts":
            example["labels"] = count_labels(
                sig, _cf, clip_hard=_ch if apply_clip else None, keep_missing=mask_missing
            )
        else:
            example["labels"] = transform_signal(
                sig,
                _sf,
                _cs,
                _ch,
                background=_bg,
                apply_scale=apply_scale,
                apply_squash=apply_squash,
                apply_clip=apply_clip,
                keep_missing=mask_missing,
            )
        return example

    return transform_example
