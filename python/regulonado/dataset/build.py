"""HuggingFace Arrow dataset builder for sequence-to-function model training.

Builds datasets directly from BED + FASTA + BigWig with no intermediate format.
Sequences stored as int8 one-hot (4, L), signals as float32 raw coverage (T, B).

Stochastic shift augmentation is supported by storing slightly wider arrays
(``shift_max_bp`` extra context on each side) and cropping at read time.
Transforms (scale / squash / clip / RC augmentation / shift crop) are all
applied via a transform function returned by ``make_transform`` and attached
to the HF dataset with ``dataset.set_transform(fn)``.

Example::

    from regulonado.dataset import build_dataset_fast, make_transform

    build_dataset_fast(
        "intervals.bed", "genome.fa", ["plus.bw", "minus.bw"],
        output_dir="dataset/hf-v1",
        splits={"train": ["train"], "validation": ["valid"]},
        shift_max_bp=128,
        n_extract_threads=16,
    )

    ds = datasets.load_from_disk("dataset/hf-v1")
    ds["train"].set_transform(make_transform(scale_factors, clip_soft, clip_hard))
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datasets import Features
from loguru import logger

_DEFAULT_CONTEXT = 524_288
_DEFAULT_PRED_BINS = 6_144
_DEFAULT_BIN_SIZE = 32
_DEFAULT_SIGNAL_SAMPLE_CHUNK = 8
_DEFAULT_SIGNAL_TRACK_CHUNK = 128
_DEFAULT_ARROW_BATCH_SIZE = 8
_DEFAULT_ARROW_COMPRESSION = "lz4"

# Target on-disk size per Arrow shard file. The chrom_pass writer groups whole
# record batches into shard files to hit roughly this size, decoupling the
# shard count from the (RAM-bounded) record-batch size. ~256 MB keeps shard
# counts in the low hundreds while staying comfortably under the ~500 MB the
# HuggingFace/Arrow ecosystem recommends per shard.
_DEFAULT_SHARD_TARGET_MB = 256
# Rough on-disk compression ratio (compressed / uncompressed) per IPC codec,
# used only to size shards. One-hot sequence compresses very well; float32
# labels much less, so these are deliberately conservative (over-estimate the
# compressed size, i.e. under-fill shards rather than overshoot the target).
_SHARD_COMPRESSION_RATIO = {"zstd": 0.5, "lz4": 0.65, "none": 1.0}


def _recommend_shard_size(
    *,
    n_tracks: int,
    stored_n_bins: int,
    stored_context: int,
    compression: str,
    target_mb: int,
    batch_size: int,
) -> int:
    """Samples per shard file to hit ~``target_mb`` on disk.

    Estimates the per-sample on-disk footprint from the schema (float32 labels +
    int8 one-hot sequence + small per-row overhead) and divides the target byte
    budget by it. The result is rounded up to a whole number of record batches
    so every shard is a clean sequence of ``batch_size`` batches, and clamped to
    at least one batch.
    """
    label_bytes = n_tracks * stored_n_bins * 4
    seq_bytes = 4 * stored_context  # int8 one-hot, 4 channels
    row_overhead = 128  # interval string + index/local_index + Arrow framing
    per_sample_uncompressed = label_bytes + seq_bytes + row_overhead
    ratio = _SHARD_COMPRESSION_RATIO.get(compression.lower(), 0.65)
    per_sample_on_disk = max(1, int(per_sample_uncompressed * ratio))

    target_bytes = max(1, target_mb) * 1_000_000
    est_samples = max(1, target_bytes // per_sample_on_disk)
    # Round up to a whole number of record batches, but never below one batch.
    n_batches = max(1, -(-est_samples // batch_size))
    return n_batches * batch_size


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

_FASTA_COMPANIONS = (".fai", ".gzi")  # pyfaidx index, bgzf index


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


def _rsync_tree(src: Path, dest: Path, *, delete: bool) -> None:
    """Copy one directory tree into another via rsync."""
    args = ["rsync", "-a"]
    if delete:
        args.append("--delete")
    args.extend([f"{src}/", f"{dest}/"])
    subprocess.run(args, check=True)


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


def _read_track(reader, chrom: str, start: int, end: int, n_bins: int) -> np.ndarray:
    return np.asarray(reader.values(chrom, start, end, bins=n_bins, exact=True), dtype=np.float32)


def _read_all_tracks(
    readers: list,
    paths: list[str],
    chrom: str,
    start: int,
    end: int,
    n_bins: int,
    executor: ThreadPoolExecutor,
) -> np.ndarray:
    """Read all BigWig tracks in parallel; pybigtools releases the GIL for I/O."""
    futs = [executor.submit(_read_track, r, chrom, start, end, n_bins) for r in readers]
    stacked = np.stack([f.result() for f in futs], axis=0)  # (T, n_bins)
    np.nan_to_num(stacked, nan=0.0, copy=False)
    for i, path in enumerate(paths):
        if _is_minus_strand(path):
            row = stacked[i]
            nz = row[row != 0.0]
            if nz.size > 0 and float(np.mean(nz < 0)) >= 0.8 and float(np.median(nz)) < 0:
                stacked[i] *= -1.0
    return stacked


# ---------------------------------------------------------------------------
# Fast-path helpers
# ---------------------------------------------------------------------------


def _load_bed_rows(bed_file: str | Path) -> list[tuple[str, int, int, str]]:
    """Return all (chrom, start, end, fold) tuples from a BED file (gzipped ok)."""
    rows: list[tuple[str, int, int, str]] = []
    open_fn = gzip.open if str(bed_file).endswith(".gz") else open
    with open_fn(str(bed_file), "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            fold = parts[3] if len(parts) > 3 else ""
            rows.append((chrom, start, end, fold))
    return rows


def _signal_intervals(
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


def _is_contiguous(indices: Sequence[int]) -> bool:
    return bool(indices) and indices[-1] - indices[0] + 1 == len(indices)


def _sample_major_signal_generator(
    signal_path: str | Path,
    n_tracks: int,
    n_split_samples: int,
    stored_n_bins: int,
    sample_indices: list[int],
    bed_file: Path,
    bed_rows: list[tuple[str, int, int, str]],
    fasta_file: str | Path,
    stored_context: int,
) -> Iterator[dict]:
    """Phase 2 generator: read sequences from FASTA and sample-major signals.

    ``sample_indices`` are global row indices into ``bed_rows``. The signal
    memmap is local to this split and stores rows in the same order.
    """
    signal_mm = np.memmap(
        signal_path,
        dtype="<f4",
        mode="r",
        shape=(n_split_samples, n_tracks, stored_n_bins),
    )

    # Build a GenomeIntervalDataset over the FULL BED (no fold filter) so we can
    # index by global row number via sample_indices.
    from enformer_pytorch.data import GenomeIntervalDataset  # noqa: PLC0415

    gid = GenomeIntervalDataset(
        str(bed_file),
        fasta_file=str(fasta_file),
        context_length=stored_context,
        rc_aug=False,
        return_augs=False,
    )

    chroms = [r[0] for r in bed_rows]
    starts = [r[1] for r in bed_rows]
    ends = [r[2] for r in bed_rows]

    for local_idx, global_idx in enumerate(sample_indices):
        try:
            seq = gid[global_idx].permute(1, 0).numpy().astype(np.int8)
            signal = np.array(signal_mm[local_idx], dtype=np.float32, copy=True)
            yield {
                "input_ids": seq,
                "labels": signal,
                "interval": f"{chroms[global_idx]}:{starts[global_idx]}-{ends[global_idx]}",
                "index": np.int64(global_idx),
                "local_index": np.int64(local_idx),
            }
        except Exception:
            logger.exception(f"Skipping global index {global_idx}")


def _write_hf_split_metadata(
    split_dir: Path,
    *,
    features: Features,
    arrow_filenames: list[str],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "citation": "",
        "description": "",
        "features": features.to_dict(),
        "homepage": "",
        "license": "",
    }
    state = {
        "_data_files": [{"filename": f} for f in arrow_filenames],
        "_fingerprint": "regulonado-rust-arrow",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (split_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    (split_dir / "state.json").write_text(json.dumps(state, indent=2))


def build_dataset_fast(
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
    signal_sample_chunk: int = _DEFAULT_SIGNAL_SAMPLE_CHUNK,
    signal_track_chunk: int = _DEFAULT_SIGNAL_TRACK_CHUNK,
    arrow_batch_size: int = _DEFAULT_ARROW_BATCH_SIZE,
    shard_size: int | None = None,
    shard_target_mb: int = _DEFAULT_SHARD_TARGET_MB,
    arrow_compression: str = _DEFAULT_ARROW_COMPRESSION,
    arrow_write_threads: int | None = None,
    num_proc: int = 1,
    cache_dir: str | None = None,
    writer_batch_size: int = 500,
    stage_to_scratch: bool = False,
    overwrite: bool = False,
    profile: bool = False,
    strategy: str = "chrom_pass",
    chrom_filter: list[str] | None = None,
    return_dataset: bool = True,
) -> object | None:
    """Fast low-scratch dataset build using the Rust extension.

    Each split is written directly from BigWig + FASTA sources into compressed
    Arrow shards. Peak scratch is the current Arrow output plus one in-memory
    record batch, not a full dense `(tracks, samples, bins)` signal file.

    Track identity, dedupe and QC are already settled by the time this runs —
    ``track_table`` is ``tracks.parquet`` from ``regulonado tracks assemble``,
    the *only* place a track list is named. This function reads it, verifies
    each included track's fingerprint against disk, and writes the same table
    (with build-time scalars merged in) to ``output_dir/tracks.parquet``.

    Parameters
    ----------
    strategy : {"chrom_pass", "fast"}
        - "chrom_pass" (default): one chromosome at a time, decoding all
          tracks' binned signal once per chromosome and slicing per-sample
          rows from RAM. Shards are ordered chrom-major (largest chromosome
          first); each chromosome yields ``ceil(samples / shard_size)`` shard
          files sized to ~``shard_target_mb`` on disk. Rows within each shard
          are in original BED order. ~10× fewer BigWig seeks than "fast".
        - "fast": sample-batched writer; reads each sample's interval from
          every BigWig per batch. Retained for parity testing.
    chrom_filter : list[str] | None
        If given, restrict each split to BED rows on these chromosomes.
        ``bed_rows`` is *not* renumbered — the ``index`` column on every
        output row remains the absolute row position in the input BED
        file. Useful for smoke tests on a single chromosome.
    return_dataset : bool
        If False, skip reopening the saved DatasetDict from ``output_dir``.
        This avoids an expensive post-build reload when the caller only
        needs the on-disk dataset.
    """
    from regulonado.tracks_table import read_track_table, verify_fingerprint

    if splits is None:
        splits = DEFAULT_SPLITS
    if shift_max_bp % bin_size != 0:
        raise ValueError(
            f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})"
        )

    bed_file = Path(bed_file)
    output_dir = Path(output_dir)
    table = read_track_table(track_table)
    included = table[table["status"] == "included"].sort_values("track_index")
    bw_paths = included["resolved_path"].tolist()
    for _, row in included.iterrows():
        expected = {k: row[k] for k in row.index if k.startswith("fp_") and pd.notna(row[k])}
        problems = verify_fingerprint(row["resolved_path"], expected)
        if problems:
            raise ValueError(
                f"Track {row['track_name']!r} no longer matches its recorded fingerprint "
                f"({row['resolved_path']}): {'; '.join(problems)}. Re-run discovery/assembly."
            )
    n_tracks = len(bw_paths)
    stored_context = context_length + 2 * shift_max_bp
    shift_bins = shift_max_bp // bin_size
    stored_n_bins = n_pred_bins + 2 * shift_bins

    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    if cache_dir is None:
        cache_dir = str(scratch_root / "hf_cache")
    scratch_out = Path(tempfile.mkdtemp(prefix="regulonado-build-", dir=scratch_root))

    active_fasta = str(fasta_file)
    active_bw_paths = bw_paths
    if stage_to_scratch:
        stage_dir = scratch_out / "stage"
        logger.info(f"Staging source files to {stage_dir}")
        active_fasta = _stage_files([fasta_file], stage_dir, _FASTA_COMPANIONS)[0]
        _stage_files([bed_file], stage_dir)
        active_bw_paths = _stage_files(bw_paths, stage_dir)

    bed_rows = _load_bed_rows(bed_file)
    n_all_samples = len(bed_rows)
    signal_intervals = _signal_intervals(bed_rows, n_pred_bins, bin_size, shift_max_bp)
    logger.info(
        f"Resolved inputs: {n_tracks} tracks, {n_all_samples} BED rows, "
        f"stored_context={stored_context}, stored_n_bins={stored_n_bins}, "
        f"scratch_root={scratch_root.resolve()}, scratch_out={scratch_out.resolve()}"
    )
    if _is_remote_fs(scratch_out):
        logger.warning(
            f"scratch_out resolves to {str(scratch_out.resolve())!r} — "
            f"Arrow I/O will hit remote storage"
        )
    if _is_remote_fs(output_dir):
        logger.info(
            f"output_dir resolves to {str(output_dir.resolve())!r}; "
            f"only the final rsync should hit remote storage"
        )
    # Arrow ListArray uses i32 offsets; batch * n_tracks * n_bins must fit.
    _i32_max = 2_147_483_647
    _max_safe_batch = max(1, _i32_max // max(1, n_tracks * stored_n_bins))
    effective_arrow_batch = max(1, min(arrow_batch_size, _max_safe_batch))
    if effective_arrow_batch < arrow_batch_size:
        logger.warning(
            f"Capping arrow_batch_size from {arrow_batch_size} to {effective_arrow_batch} "
            f"to avoid Arrow i32 offset overflow ({n_tracks} tracks × {stored_n_bins} bins)"
        )
    # Shard files group whole record batches up to a target on-disk size. An
    # explicit shard_size wins; otherwise derive it from shard_target_mb.
    if shard_size is not None:
        effective_shard_size = max(effective_arrow_batch, shard_size)
    else:
        effective_shard_size = _recommend_shard_size(
            n_tracks=n_tracks,
            stored_n_bins=stored_n_bins,
            stored_context=stored_context,
            compression=arrow_compression,
            target_mb=shard_target_mb,
            batch_size=effective_arrow_batch,
        )
    logger.info(
        f"Shard sizing: {effective_shard_size} samples/shard "
        f"({effective_shard_size // effective_arrow_batch} record batch(es) of "
        f"{effective_arrow_batch}), target≈{shard_target_mb} MB on disk, "
        f"compression={arrow_compression}"
    )
    label_batch_gb = effective_arrow_batch * n_tracks * stored_n_bins * 4 / 1e9
    seq_batch_gb = effective_arrow_batch * 4 * stored_context / 1e9
    effective_arrow_write_threads = (
        4 if arrow_write_threads is None else max(1, arrow_write_threads)
    )
    chrom_pass_writer_peak_gb = effective_arrow_write_threads * (label_batch_gb + seq_batch_gb)
    chrom_lengths: dict[str, int] = {}
    fai_path = Path(str(active_fasta) + ".fai")
    if fai_path.exists():
        for line in fai_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                chrom_lengths[parts[0]] = int(parts[1])
    if chrom_lengths:
        used_chroms = {bed_rows[i][0] for i in range(n_all_samples)}
        largest_chrom_bins = max(
            (chrom_lengths.get(chrom, 0) // bin_size for chrom in used_chroms),
            default=0,
        )
        chrom_matrix_gb = n_tracks * largest_chrom_bins * 4 / 1e9
        binning_scratch_gb = n_extract_threads * largest_chrom_bins * (8 + 8) / 1e9
    else:
        chrom_matrix_gb = 0.0
        binning_scratch_gb = 0.0
    chrom_pass_peak_gb = chrom_matrix_gb + binning_scratch_gb + chrom_pass_writer_peak_gb
    logger.info(
        f"Approx per-batch RAM: labels={label_batch_gb:.1f} GB, "
        f"direct-extract peak≈{2 * label_batch_gb + seq_batch_gb:.1f} GB, "
        f"chrom-pass matrix≈{chrom_matrix_gb:.1f} GB, "
        f"binning scratch≈{binning_scratch_gb:.1f} GB, "
        f"writer peak≈{chrom_pass_writer_peak_gb:.1f} GB, "
        f"combined chrom-pass peak≈{chrom_pass_peak_gb:.1f} GB "
        f"(batch_size={effective_arrow_batch}, n_extract_threads={n_extract_threads}, "
        f"arrow_write_threads={effective_arrow_write_threads})"
    )

    from datasets import Array2D, Dataset, DatasetDict, Features, Value  # noqa: PLC0415

    features = Features(
        {
            "input_ids": Array2D(dtype="int8", shape=(4, stored_context)),
            "labels": Array2D(dtype="float32", shape=(n_tracks, stored_n_bins)),
            "interval": Value(dtype="string"),
            "index": Value(dtype="int64"),
            "local_index": Value(dtype="int64"),
        }
    )

    chrom_filter_set = set(chrom_filter) if chrom_filter else None

    split_indices: dict[str, list[int]] = {}
    splits_to_build: list[str] = []
    for split, folds in splits.items():
        split_out = output_dir / split
        if not overwrite and (split_out / "dataset_info.json").exists():
            continue
        if folds:
            folds_set = set(folds)
            idx = [i for i, r in enumerate(bed_rows) if r[3] in folds_set]
        else:
            idx = list(range(n_all_samples))
        if chrom_filter_set is not None:
            idx = [i for i in idx if bed_rows[i][0] in chrom_filter_set]
        split_indices[split] = idx
        splits_to_build.append(split)
        if chrom_filter_set is not None:
            logger.info(
                f"Split '{split}': {len(idx)} samples "
                f"(filtered to chroms {sorted(chrom_filter_set)})"
            )
        else:
            logger.info(f"Split '{split}': {len(idx)} samples")

    split_datasets: dict[str, Dataset] = {}
    if not splits_to_build:
        if not return_dataset:
            logger.info("All splits already exist; skipping load because return_dataset=False")
            return None
        logger.info("All splits already exist; loading from disk")
        for split in splits:
            split_datasets[split] = Dataset.load_from_disk(str(output_dir / split))
        return DatasetDict(split_datasets)

    if strategy not in {"chrom_pass", "fast"}:
        raise ValueError(f"strategy must be 'chrom_pass' or 'fast', got {strategy!r}")
    from regulonado._rs import (  # type: ignore[import]
        write_arrow_split_from_bigwigs,
        write_arrow_splits_chrom_pass,
    )

    scratch_out.mkdir(parents=True, exist_ok=True)

    minus_flags = [_is_minus_strand(p) for p in active_bw_paths]
    t_arrow_total = time.perf_counter()
    if strategy == "chrom_pass":
        chrom_split_names: list[str] = []
        chrom_split_out_dirs: list[str] = []
        chrom_split_indices: list[list[int]] = []
        for split in splits:
            split_out = output_dir / split
            if not overwrite and (split_out / "dataset_info.json").exists():
                logger.info(f"Split '{split}' exists; skipping rebuild")
                if return_dataset:
                    split_datasets[split] = Dataset.load_from_disk(str(split_out))
                continue

            sample_indices = split_indices[split]
            logger.info(
                f"Queueing '{split}' shard(s) [chrom_pass shared-scan]: "
                f"{len(sample_indices)} samples, {n_tracks} tracks, "
                f"stored_context={stored_context} bp, stored_n_bins={stored_n_bins}, "
                f"batch_size={effective_arrow_batch}, compression={arrow_compression}, "
                f"arrow_write_threads={effective_arrow_write_threads}"
            )

            split_scratch = scratch_out / split
            if split_scratch.exists():
                shutil.rmtree(split_scratch)
            split_scratch.mkdir(parents=True, exist_ok=True)
            chrom_split_names.append(split)
            chrom_split_out_dirs.append(str(split_scratch))
            chrom_split_indices.append(sample_indices)

        if chrom_split_names:
            write_arrow_splits_chrom_pass(
                active_bw_paths,
                minus_flags,
                signal_intervals,
                chrom_split_names,
                chrom_split_out_dirs,
                chrom_split_indices,
                bed_rows,
                active_fasta,
                stored_n_bins,
                stored_context,
                bin_size,
                batch_size=effective_arrow_batch,
                shard_size=effective_shard_size,
                n_threads=n_extract_threads,
                arrow_write_threads=effective_arrow_write_threads,
                compression=arrow_compression,
                profile=profile,
            )

        for split, split_scratch_str in zip(chrom_split_names, chrom_split_out_dirs, strict=True):
            split_scratch = Path(split_scratch_str)
            arrow_filenames = sorted(p.name for p in split_scratch.glob("data-*-of-*.arrow"))
            if not arrow_filenames:
                raise RuntimeError(
                    f"chrom_pass produced no shards in {split_scratch}; "
                    f"check that the FASTA contains the BED chromosomes"
                )
            logger.info(f"Arrow shard(s) for '{split}' written ({len(arrow_filenames)} file(s))")
            _write_hf_split_metadata(
                split_scratch, features=features, arrow_filenames=arrow_filenames
            )
            if return_dataset:
                split_datasets[split] = Dataset.load_from_disk(str(split_scratch))
    else:
        for split, folds in splits.items():
            split_out = output_dir / split
            if not overwrite and (split_out / "dataset_info.json").exists():
                logger.info(f"Split '{split}' exists; skipping rebuild")
                if return_dataset:
                    split_datasets[split] = Dataset.load_from_disk(str(split_out))
                continue

            sample_indices = split_indices[split]
            logger.info(
                f"Writing '{split}' shard(s) [{strategy}]: {len(sample_indices)} samples, "
                f"{n_tracks} tracks, stored_context={stored_context} bp, "
                f"stored_n_bins={stored_n_bins}, batch_size={effective_arrow_batch}, "
                f"compression={arrow_compression}"
            )

            split_scratch = scratch_out / split
            if split_scratch.exists():
                shutil.rmtree(split_scratch)
            split_scratch.mkdir(parents=True, exist_ok=True)

            t_split_arrow = time.perf_counter()
            arrow_filenames = ["data-00000-of-00001.arrow"]
            write_arrow_split_from_bigwigs(
                active_bw_paths,
                minus_flags,
                signal_intervals,
                str(split_scratch / arrow_filenames[0]),
                sample_indices,
                bed_rows,
                active_fasta,
                stored_n_bins,
                stored_context,
                batch_size=effective_arrow_batch,
                n_threads=n_extract_threads,
                compression=arrow_compression,
            )
            logger.info(
                f"Arrow shard(s) for '{split}' written in "
                f"{time.perf_counter() - t_split_arrow:.1f}s ({len(arrow_filenames)} file(s))"
            )
            _write_hf_split_metadata(
                split_scratch, features=features, arrow_filenames=arrow_filenames
            )
            if return_dataset:
                split_datasets[split] = Dataset.load_from_disk(str(split_scratch))
    logger.info(f"Arrow writing completed in {time.perf_counter() - t_arrow_total:.1f}s")

    logger.info(f"Publishing rebuilt splits to {output_dir}")
    t_rsync = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits_to_build:
        _rsync_tree(scratch_out / split, output_dir / split, delete=True)
    (output_dir / "dataset_dict.json").write_text(
        json.dumps({"splits": list(splits)}, indent=2)
    )
    logger.info(f"Publication completed in {time.perf_counter() - t_rsync:.1f}s")
    shutil.rmtree(scratch_out)

    from datetime import datetime, timezone  # noqa: PLC0415

    from regulonado.tracks_table import write_track_table  # noqa: PLC0415

    write_track_table(
        table,
        output_dir / "tracks.parquet",
        bed_file=str(bed_file),
        fasta_file=str(fasta_file),
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        splits=splits,
        build_strategy=strategy,
        arrow_write_threads=effective_arrow_write_threads,
        created_at=datetime.now(timezone.utc).isoformat(),
        regulonado_version=_regulonado_version(),
        command=" ".join(sys.argv),
    )

    logger.info(f"Dataset saved to {output_dir}")
    if not return_dataset:
        logger.info(
            "Skipping final DatasetDict.load_from_disk(); caller can reopen output_dir if needed"
        )
        return None
    return DatasetDict.load_from_disk(str(output_dir))


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

    Mirrors the per-sample signal transform inside ``make_transform``.
    Returns float32.
    """
    sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1, 1)
    cs = np.broadcast_to(np.asarray(clip_soft, dtype=np.float32), (sf.shape[0],)).reshape(-1, 1)
    ch = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (sf.shape[0],)).reshape(-1, 1)

    out = np.asarray(signal, dtype=np.float32).copy()
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
) -> Callable[[dict], dict]:
    """Return a transform compatible with ``dataset.set_transform()``.

    Applied per batch:
        1. Shift crop: random offset when center_crop=False, center offset when center_crop=True.
           Always applied when shift_max_bins > 0.
        2. RC augmentation    (if enable_rc_aug)
        3. Signal transform   (scale → squash → clip)
        4. Cast input_ids to float32

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
    """
    sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1)
    n_tracks = sf.size
    cs = np.broadcast_to(np.asarray(clip_soft, dtype=np.float32), (n_tracks,)).copy()
    ch = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (n_tracks,)).copy()
    bg = (
        None
        if background is None
        else np.broadcast_to(np.asarray(background, dtype=np.float32), (n_tracks,)).copy()
    )

    def _transform_signal(
        labels: np.ndarray,
        _sf: np.ndarray,
        _cs: np.ndarray,
        _ch: np.ndarray,
        _bg: np.ndarray | None,
    ) -> np.ndarray:
        return transform_signal(
            labels,
            _sf,
            _cs,
            _ch,
            background=_bg,
            apply_scale=apply_scale,
            apply_squash=apply_squash,
            apply_clip=apply_clip,
        )

    def transform_batch(batch: dict) -> dict:
        batch = dict(batch)
        raw_ids = batch.get("input_ids")
        raw_labels = batch.get("labels")

        ids_arr = np.asarray(raw_ids, dtype=np.float32) if raw_ids is not None else None
        lbl_arr = np.asarray(raw_labels, dtype=np.float32) if raw_labels is not None else None

        batched = ids_arr is not None and ids_arr.ndim == 3
        bs = ids_arr.shape[0] if batched else 1

        def _unpack(arr: np.ndarray) -> list[np.ndarray]:
            return [arr[i] for i in range(bs)] if batched else [arr]

        ids_list = _unpack(ids_arr) if ids_arr is not None else [None] * bs
        lbl_list = _unpack(lbl_arr) if lbl_arr is not None else [None] * bs

        out_ids: list[np.ndarray] = []
        out_lbl: list[np.ndarray] = []
        for seq, sig in zip(ids_list, lbl_list):
            _sf, _cs, _ch, _bg = sf, cs, ch, bg

            # --- shift crop (always applied when shift buffer was stored)
            if shift_max_bins > 0:
                s = (
                    shift_max_bins
                    if center_crop
                    else int(np.random.randint(0, 2 * shift_max_bins + 1))
                )
                if seq is not None:
                    seq = seq[:, s * bin_size : s * bin_size + context_length]
                if sig is not None:
                    sig = sig[:, s : s + n_pred_bins]

            # --- RC augmentation
            if enable_rc_aug and np.random.rand() < 0.5:
                if seq is not None:
                    seq = np.flip(seq, axis=(0, 1)).copy()
                if sig is not None:
                    sig = np.flip(sig, axis=-1)
                    if rc_permutation is not None:
                        sig = np.take(sig, rc_permutation, axis=-2)
                        _sf = sf[rc_permutation]
                        _cs = cs[rc_permutation]
                        _ch = ch[rc_permutation]
                        _bg = None if bg is None else bg[rc_permutation]
                    sig = sig.copy()

            if seq is not None:
                out_ids.append(seq)
            if sig is not None:
                out_lbl.append(_transform_signal(sig, _sf, _cs, _ch, _bg))

        if out_ids:
            batch["input_ids"] = np.stack(out_ids) if batched else out_ids[0]
        if out_lbl:
            batch["labels"] = np.stack(out_lbl) if batched else out_lbl[0]

        return batch

    return transform_batch
