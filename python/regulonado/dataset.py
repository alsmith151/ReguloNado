"""HuggingFace Arrow dataset builder for sequence-to-function model training.

Builds datasets directly from BED + FASTA + BigWig with no intermediate format.
Sequences stored as int8 one-hot (4, L), signals as float32 raw coverage (T, B).

Two build strategies are available:

* **Fast path** (default): direct Rust BigWig → Arrow writing. Each Arrow
  record batch reads all tracks for that batch, writes compressed Arrow, and
  avoids the previous multi-TB dense scratch signal file.

* **Legacy path**: ``sample_generator`` with pybigtools + ThreadPoolExecutor.
  Pass ``use_fast_path=False`` to use it.

Stochastic shift augmentation is supported by storing slightly wider arrays
(``shift_max_bp`` extra context on each side) and cropping at read time.
Transforms (scale / squash / clip / RC augmentation / shift crop) are all
applied via a transform function returned by ``make_transform`` and attached
to the HF dataset with ``dataset.set_transform(fn)``.

Example::

    from regulonado.dataset import build_dataset, make_transform

    build_dataset(
        "intervals.bed", "genome.fa", ["plus.bw", "minus.bw"],
        output_dir="dataset/hf-v1",
        splits={"train": ["train"], "validation": ["valid"]},
        shift_max_bp=128,
        num_proc=16,
    )

    ds = datasets.load_from_disk("dataset/hf-v1")
    ds["train"].set_transform(make_transform(scale_factors, clip_soft, clip_hard))
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from loguru import logger

_DEFAULT_CONTEXT = 524_288
_DEFAULT_PRED_BINS = 6_144
_DEFAULT_BIN_SIZE = 32
_DEFAULT_SIGNAL_SAMPLE_CHUNK = 8
_DEFAULT_SIGNAL_TRACK_CHUNK = 128
_DEFAULT_ARROW_BATCH_SIZE = 8
_DEFAULT_ARROW_COMPRESSION = "zstd"

DEFAULT_SPLITS: dict[str, list[str]] = {
    "train": ["fold0", "fold1", "fold2", "fold5", "fold6", "fold7"],
    "validation": ["fold4"],
    "test": ["fold3"],
}


# ---------------------------------------------------------------------------
# Scratch staging
# ---------------------------------------------------------------------------

_FASTA_COMPANIONS = (".fai", ".gzi")  # pyfaidx index, bgzf index


def _rsync_one(src: Path, scratch: Path, companion_suffixes: tuple[str, ...]) -> str:
    """Copy one file (and any companions) to scratch via rsync. Returns staged path."""
    subprocess.run(["rsync", "-au", str(src), f"{scratch}/"], check=True)
    for suf in companion_suffixes:
        companion = Path(str(src) + suf)
        if companion.exists():
            subprocess.run(["rsync", "-au", str(companion), f"{scratch}/"], check=True)
    return str(scratch / src.name)


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
    n = len(srcs)
    workers = min(max_workers, n) if n else 1
    logger.info(f"Staging {n} file(s) → {scratch} (workers={workers})")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_rsync_one, src, scratch, companion_suffixes): src for src in srcs}
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
    return np.asarray(
        reader.values(chrom, start, end, bins=n_bins, exact=True), dtype=np.float32
    )


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


def _phase1_extract_signals(
    bw_paths: list[str],
    signal_intervals: list[tuple[str, int, int]],
    signal_path: Path,
    stored_n_bins: int,
    n_threads: int,
) -> None:
    """Phase 1: extract each BigWig for all intervals → one track-major float32 file.

    Output file: ``signal_path``
    Shape: ``(n_tracks, n_intervals, stored_n_bins)`` float32 row-major LE.
    Python reads back with::

        np.memmap(path, dtype="<f4", mode="r", shape=(n_tracks, n_intervals, stored_n_bins))
    """
    from regulonado._rs import extract_all_tracks_to_file  # type: ignore[import]

    signal_path.parent.mkdir(parents=True, exist_ok=True)
    minus_flags = [_is_minus_strand(p) for p in bw_paths]
    n = len(signal_intervals)
    total_gb = len(bw_paths) * n * stored_n_bins * 4 / 1e9
    logger.info(
        f"Phase 1: extracting {len(bw_paths)} tracks × {n} intervals "
        f"→ {signal_path}  (~{total_gb:.1f} GB on disk)"
    )
    extract_all_tracks_to_file(
        bw_paths,
        signal_intervals,
        stored_n_bins,
        str(signal_path),
        minus_flags=minus_flags,
        n_threads=n_threads,
    )


def _is_contiguous(indices: Sequence[int]) -> bool:
    return bool(indices) and indices[-1] - indices[0] + 1 == len(indices)


def _materialize_sample_major_split(
    track_signal_path: str | Path,
    split_signal_path: str | Path,
    sample_indices: list[int],
    n_tracks: int,
    n_all_samples: int,
    stored_n_bins: int,
    *,
    sample_chunk_size: int = _DEFAULT_SIGNAL_SAMPLE_CHUNK,
    track_chunk_size: int = _DEFAULT_SIGNAL_TRACK_CHUNK,
) -> None:
    """Transpose selected rows from track-major extraction into sample-major split data.

    Phase 1 writes tracks sequentially because that is the fastest BigWig access
    pattern. Arrow generation wants one complete sample at a time. This step
    pays one streaming-ish transpose per split so the generator can read each
    labels array as a single contiguous block.
    """
    split_signal_path = Path(split_signal_path)
    split_signal_path.parent.mkdir(parents=True, exist_ok=True)

    expected_bytes = len(sample_indices) * n_tracks * stored_n_bins * 4
    if split_signal_path.exists() and split_signal_path.stat().st_size == expected_bytes:
        logger.info(f"Using existing sample-major signal file: {split_signal_path}")
        return

    logger.info(
        f"Materializing sample-major signals: {len(sample_indices)} samples × "
        f"{n_tracks} tracks → {split_signal_path} (~{expected_bytes / 1e9:.1f} GB)"
    )

    track_mm = np.memmap(
        track_signal_path,
        dtype="<f4",
        mode="r",
        shape=(n_tracks, n_all_samples, stored_n_bins),
    )
    split_mm = np.memmap(
        split_signal_path,
        dtype="<f4",
        mode="w+",
        shape=(len(sample_indices), n_tracks, stored_n_bins),
    )

    for sample_start in range(0, len(sample_indices), sample_chunk_size):
        sample_end = min(sample_start + sample_chunk_size, len(sample_indices))
        idxs = sample_indices[sample_start:sample_end]
        if _is_contiguous(idxs):
            idx_sel: slice | list[int] = slice(idxs[0], idxs[-1] + 1)
        else:
            idx_sel = idxs

        for track_start in range(0, n_tracks, track_chunk_size):
            track_end = min(track_start + track_chunk_size, n_tracks)
            block = np.asarray(track_mm[track_start:track_end, idx_sel, :], dtype=np.float32)
            split_mm[sample_start:sample_end, track_start:track_end, :] = np.moveaxis(block, 0, 1)

    split_mm.flush()


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
            }
        except Exception:
            logger.exception(f"Skipping global index {global_idx}")


def _write_hf_split_metadata(
    split_dir: Path,
    *,
    features: Features,
    arrow_filename: str,
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
        "_data_files": [{"filename": arrow_filename}],
        "_fingerprint": "regulonado-rust-arrow",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (split_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    (split_dir / "state.json").write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Generator  (HF index-shard compatible)
# ---------------------------------------------------------------------------

def sample_generator(
    bed_file: str | Path,
    fasta_file: str | Path,
    bigwig_paths: Sequence[str | Path],
    indices: list[int],
    *,
    dataset_folds: list[str] | None = None,
    context_length: int = _DEFAULT_CONTEXT,
    bin_size: int = _DEFAULT_BIN_SIZE,
    n_pred_bins: int = _DEFAULT_PRED_BINS,
    shift_max_bp: int = 0,
    n_io_threads: int = 8,
) -> Iterator[dict]:
    """Yield raw samples from BED/FASTA/BigWig sources.

    Pass a list of indices so HF can shard across ``num_proc`` workers::

        Dataset.from_generator(
            sample_generator,
            gen_kwargs={"bed_file": ..., ..., "indices": list(range(n))},
            num_proc=16,
        )

    Stored shapes:
        input_ids : int8   (4,  context_length + 2*shift_max_bp)
        labels    : float32  (T,  n_pred_bins   + 2*(shift_max_bp // bin_size))
    """
    import polars as pl  # noqa: PLC0415
    import pybigtools  # noqa: PLC0415
    from enformer_pytorch.data import GenomeIntervalDataset  # noqa: PLC0415

    stored_context = context_length + 2 * shift_max_bp
    shift_bins = shift_max_bp // bin_size
    stored_n_bins = n_pred_bins + 2 * shift_bins
    pred_bp = n_pred_bins * bin_size

    filter_fn = None
    if dataset_folds:
        folds = set(dataset_folds)
        filter_fn = lambda df: df.filter(pl.col("column_4").is_in(folds))

    gid = GenomeIntervalDataset(
        str(bed_file),
        fasta_file=str(fasta_file),
        context_length=stored_context,
        filter_df_fn=filter_fn,
        rc_aug=False,
        return_augs=False,
    )

    # Pre-extract intervals once to avoid repeated polars row() lookups
    df = gid.df
    chroms = df.get_column(df.columns[0]).to_list()
    bed_starts = df.get_column(df.columns[1]).cast(pl.Int64).to_list()
    bed_ends = df.get_column(df.columns[2]).cast(pl.Int64).to_list()

    bw_paths = [str(p) for p in bigwig_paths]
    readers = [pybigtools.open(p) for p in bw_paths]
    n_threads = min(n_io_threads, len(readers)) if readers else 1

    try:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for index in indices:
                try:
                    # Sequence: (stored_context, 4) → permute → (4, stored_context)
                    seq = gid[index].permute(1, 0).numpy().astype(np.int8)

                    chrom = chroms[index]
                    center = (bed_starts[index] + bed_ends[index]) // 2

                    # Signal window: pred window + shift buffer, centered on interval
                    sig_start = center - pred_bp // 2 - shift_max_bp
                    sig_end = sig_start + pred_bp + 2 * shift_max_bp

                    signal = _read_all_tracks(
                        readers, bw_paths, chrom, sig_start, sig_end, stored_n_bins, executor
                    )

                    yield {
                        "input_ids": seq,
                        "labels": signal,
                        "interval": f"{chrom}:{bed_starts[index]}-{bed_ends[index]}",
                        "index": np.int64(index),
                    }
                except Exception:
                    logger.exception(f"Skipping index {index}")
    finally:
        for r in readers:
            try:
                r.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    bed_file: str | Path,
    fasta_file: str | Path,
    bigwig_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    splits: dict[str, list[str]] | None = None,
    context_length: int = _DEFAULT_CONTEXT,
    bin_size: int = _DEFAULT_BIN_SIZE,
    n_pred_bins: int = _DEFAULT_PRED_BINS,
    shift_max_bp: int = 0,
    n_io_threads: int = 8,
    num_proc: int = 1,
    cache_dir: str | None = None,
    writer_batch_size: int = 500,
    stage_to_scratch: bool = False,
    overwrite: bool = False,
    drop_missing: bool = False,
) -> DatasetDict:
    """Build and save an HF Arrow DatasetDict from genomic sources.

    Args:
        bed_file: BED file with genomic intervals (column 4 = fold label).
        fasta_file: Reference genome FASTA (must be indexed).
        bigwig_paths: Ordered list of BigWig files (one per track/condition).
        output_dir: Where to save the Arrow dataset.
        splits: Mapping of split name → list of fold labels to include,
            e.g. ``{"train": ["train"], "validation": ["valid"]}``.
        context_length: Nominal input sequence length fed to the model.
        bin_size: Signal bin size in bp (typically 32 for borzoi).
        n_pred_bins: Number of output bins the model predicts.
        shift_max_bp: Extra context on each side (must be a multiple of
            bin_size). Enables stochastic shift in make_transform.
        n_io_threads: Threads for parallel BigWig reads within a sample.
        num_proc: Parallel processes for dataset generation (HF shards index list).
        cache_dir: Arrow cache dir; defaults to $SLURM_TMPDIR/hf_cache or /tmp.
        writer_batch_size: Samples buffered before flushing an Arrow shard.
            Lower → more frequent Ceph writes. Default 500 balances memory and I/O.
        stage_to_scratch: Copy FASTA and all BigWig source files to the scratch
            dir before reading. Strongly recommended on Ceph — eliminates repeated
            random seeks over a distributed filesystem during generation.
        overwrite: Regenerate splits that already exist.
        drop_missing: If True, silently remove bigwig paths that don't exist and
            continue. If False (default), raise FileNotFoundError instead.

    Returns:
        The loaded DatasetDict (also saved to disk at output_dir).
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    if shift_max_bp % bin_size != 0:
        raise ValueError(f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})")

    bw_paths_raw = [str(p).strip().strip('"').strip("'") for p in bigwig_paths]
    missing = [p for p in bw_paths_raw if not Path(p).exists()]
    if missing:
        if drop_missing:
            logger.warning(
                f"Dropping {len(missing)}/{len(bw_paths_raw)} missing bigwig paths:\n"
                + "\n".join(f"  {p}" for p in missing)
            )
            bw_paths_raw = [p for p in bw_paths_raw if p not in set(missing)]
        else:
            raise FileNotFoundError(
                f"{len(missing)}/{len(bw_paths_raw)} bigwig paths do not exist:\n"
                + "\n".join(f"  {p}" for p in missing)
            )

    output_dir = Path(output_dir)
    bw_paths = bw_paths_raw
    n_tracks = len(bw_paths)
    stored_context = context_length + 2 * shift_max_bp
    stored_n_bins = n_pred_bins + 2 * (shift_max_bp // bin_size)

    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    if cache_dir is None:
        cache_dir = str(scratch_root / "hf_cache")

    import polars as pl  # noqa: PLC0415
    from datasets import Array2D, Dataset, DatasetDict, Features, Value  # noqa: PLC0415
    from enformer_pytorch.data import GenomeIntervalDataset  # noqa: PLC0415

    # Stage source files to local scratch to avoid Ceph random-seek overhead.
    # Done once in the main process so num_proc workers all read from local disk.
    active_fasta = str(fasta_file)
    active_bed = str(bed_file)
    active_bw_paths = bw_paths
    if stage_to_scratch:
        stage_dir = scratch_root / "regulonado_stage"
        logger.info(f"Staging source files to {stage_dir}")
        active_fasta = _stage_files([fasta_file], stage_dir, _FASTA_COMPANIONS)[0]
        active_bed = _stage_files([bed_file], stage_dir)[0]
        active_bw_paths = _stage_files(bw_paths, stage_dir)

    features = Features({
        "input_ids": Array2D(dtype="int8", shape=(4, stored_context)),
        "labels": Array2D(dtype="float32", shape=(n_tracks, stored_n_bins)),
        "interval": Value(dtype="string"),
        "index": Value(dtype="int64"),
    })

    gen_kwargs_base = dict(
        bed_file=active_bed,
        fasta_file=active_fasta,
        bigwig_paths=tuple(active_bw_paths),  # tuple: not sharded by HF
        context_length=context_length,
        bin_size=bin_size,
        n_pred_bins=n_pred_bins,
        shift_max_bp=shift_max_bp,
        n_io_threads=n_io_threads,
    )

    # Write the Arrow dataset to scratch first, then copy to the final Ceph path.
    # A single sequential copy is much faster than many small Arrow flush writes.
    scratch_out = scratch_root / "regulonado_build"

    split_datasets: dict[str, Dataset] = {}
    for split, folds in splits.items():
        split_out = output_dir / split
        if not overwrite and (split_out / "dataset_info.json").exists():
            logger.info(f"Split '{split}' exists; loading from disk")
            split_datasets[split] = Dataset.load_from_disk(str(split_out))
            continue

        # Count samples to build the index list for sharding
        filter_fn = None
        if folds:
            folds_set = set(folds)
            filter_fn = lambda df, f=folds_set: df.filter(pl.col("column_4").is_in(f))
        gid_probe = GenomeIntervalDataset(
            active_bed,
            fasta_file=active_fasta,
            context_length=stored_context,
            filter_df_fn=filter_fn,
        )
        n_samples = len(gid_probe)
        del gid_probe

        logger.info(
            f"Building '{split}': {n_samples} samples, {n_tracks} tracks, "
            f"stored_context={stored_context} bp, stored_n_bins={stored_n_bins}"
        )

        split_datasets[split] = Dataset.from_generator(
            sample_generator,
            gen_kwargs={**gen_kwargs_base, "indices": list(range(n_samples)), "dataset_folds": tuple(folds)},
            features=features,
            cache_dir=cache_dir,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            split=split,
        )

    ds_dict = DatasetDict(split_datasets)

    # Save to scratch then rsync to output_dir so Ceph sees one large sequential write.
    scratch_out.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(scratch_out))
    logger.info(f"Rsyncing dataset from scratch to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Trailing slash on source: copy contents, not the directory itself.
    # --delete keeps output_dir in sync if overwriting a previous build.
    subprocess.run(
        ["rsync", "-a", "--delete", f"{scratch_out}/", f"{output_dir}/"],
        check=True,
    )
    shutil.rmtree(scratch_out)

    metadata = {
        "bigwig_paths": bw_paths,
        "bed_file": str(bed_file),
        "fasta_file": str(fasta_file),
        "context_length": context_length,
        "bin_size": bin_size,
        "n_pred_bins": n_pred_bins,
        "shift_max_bp": shift_max_bp,
        "splits": splits,
    }
    (output_dir / "regulonado_metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"Dataset saved to {output_dir}")
    return DatasetDict.load_from_disk(str(output_dir))


def build_dataset_fast(
    bed_file: str | Path,
    fasta_file: str | Path,
    bigwig_paths: Sequence[str | Path],
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
    arrow_compression: str = _DEFAULT_ARROW_COMPRESSION,
    num_proc: int = 1,
    cache_dir: str | None = None,
    writer_batch_size: int = 500,
    stage_to_scratch: bool = False,
    overwrite: bool = False,
    drop_missing: bool = False,
    profile: bool = False,
) -> DatasetDict:
    """Fast low-scratch dataset build using the Rust extension.

    Each split is written directly from BigWig + FASTA sources into compressed
    Arrow shards. Peak scratch is the current Arrow output plus one in-memory
    record batch, not a full dense `(tracks, samples, bins)` signal file.
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    if shift_max_bp % bin_size != 0:
        raise ValueError(f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})")

    bed_file = Path(bed_file)
    bw_paths_raw = [str(p).strip().strip('"').strip("'") for p in bigwig_paths]
    missing = [p for p in bw_paths_raw if not Path(p).exists()]
    if missing:
        if drop_missing:
            logger.warning(
                f"Dropping {len(missing)}/{len(bw_paths_raw)} missing bigwig paths:\n"
                + "\n".join(f"  {p}" for p in missing)
            )
            bw_paths_raw = [p for p in bw_paths_raw if p not in set(missing)]
        else:
            raise FileNotFoundError(
                f"{len(missing)}/{len(bw_paths_raw)} bigwig paths do not exist:\n"
                + "\n".join(f"  {p}" for p in missing)
            )

    output_dir = Path(output_dir)
    bw_paths = bw_paths_raw
    n_tracks = len(bw_paths)
    stored_context = context_length + 2 * shift_max_bp
    shift_bins = shift_max_bp // bin_size
    stored_n_bins = n_pred_bins + 2 * shift_bins

    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    if cache_dir is None:
        cache_dir = str(scratch_root / "hf_cache")
    scratch_out = scratch_root / "regulonado_build"

    active_fasta = str(fasta_file)
    active_bed = str(bed_file)
    active_bw_paths = bw_paths
    if stage_to_scratch:
        stage_dir = scratch_root / "regulonado_stage"
        logger.info(f"Staging source files to {stage_dir}")
        active_fasta = _stage_files([fasta_file], stage_dir, _FASTA_COMPANIONS)[0]
        active_bed = _stage_files([bed_file], stage_dir)[0]
        active_bw_paths = _stage_files(bw_paths, stage_dir)

    bed_rows = _load_bed_rows(bed_file)
    n_all_samples = len(bed_rows)
    signal_intervals = _signal_intervals(bed_rows, n_pred_bins, bin_size, shift_max_bp)
    logger.info(
        f"Resolved inputs: {n_tracks} tracks, {n_all_samples} BED rows, "
        f"stored_context={stored_context}, stored_n_bins={stored_n_bins}, "
        f"scratch_root={scratch_root.resolve()}, scratch_out={scratch_out.resolve()}"
    )
    for label, path in [("scratch_out", scratch_out), ("output_dir", output_dir)]:
        resolved = str(path.resolve())
        if resolved.startswith("/ceph") or resolved.startswith("/project"):
            logger.warning(f"{label} resolves to {resolved!r} — Arrow I/O will hit Ceph, not local scratch")
    # Arrow ListArray uses i32 offsets; batch * n_tracks * n_bins must fit.
    _i32_max = 2_147_483_647
    _max_safe_batch = max(1, _i32_max // max(1, n_tracks * stored_n_bins))
    effective_arrow_batch = max(1, min(arrow_batch_size, _max_safe_batch))
    if effective_arrow_batch < arrow_batch_size:
        logger.warning(
            f"Capping arrow_batch_size from {arrow_batch_size} to {effective_arrow_batch} "
            f"to avoid Arrow i32 offset overflow ({n_tracks} tracks × {stored_n_bins} bins)"
        )
    label_batch_gb = effective_arrow_batch * n_tracks * stored_n_bins * 4 / 1e9
    seq_batch_gb = effective_arrow_batch * 4 * stored_context / 1e9
    logger.info(
        f"Approx per-batch RAM: labels={label_batch_gb:.1f} GB, "
        f"direct-extract peak≈{2 * label_batch_gb + seq_batch_gb:.1f} GB "
        f"(batch_size={effective_arrow_batch})"
    )

    from datasets import Array2D, Dataset, DatasetDict, Features, Value  # noqa: PLC0415

    features = Features({
        "input_ids": Array2D(dtype="int8", shape=(4, stored_context)),
        "labels": Array2D(dtype="float32", shape=(n_tracks, stored_n_bins)),
        "interval": Value(dtype="string"),
        "index": Value(dtype="int64"),
    })

    split_indices: dict[str, list[int]] = {}
    splits_to_build: list[str] = []
    for split, folds in splits.items():
        split_out = output_dir / split
        if not overwrite and (split_out / "dataset_info.json").exists():
            continue
        if folds:
            folds_set = set(folds)
            split_indices[split] = [i for i, r in enumerate(bed_rows) if r[3] in folds_set]
        else:
            split_indices[split] = list(range(n_all_samples))
        splits_to_build.append(split)
        logger.info(f"Split '{split}': {len(split_indices[split])} samples")

    split_datasets: dict[str, Dataset] = {}
    if not splits_to_build:
        logger.info("All splits already exist; loading from disk")
        for split in splits:
            split_datasets[split] = Dataset.load_from_disk(str(output_dir / split))
        return DatasetDict(split_datasets)

    from regulonado._rs import write_arrow_split_from_bigwigs  # type: ignore[import]

    scratch_out.mkdir(parents=True, exist_ok=True)

    # --- Write Arrow directly from BigWigs, avoiding a multi-TB signal file ----
    minus_flags = [_is_minus_strand(p) for p in active_bw_paths]
    t_arrow_total = time.perf_counter()
    for split, folds in splits.items():
        split_out = output_dir / split
        if not overwrite and (split_out / "dataset_info.json").exists():
            logger.info(f"Split '{split}' exists; loading from disk")
            split_datasets[split] = Dataset.load_from_disk(str(split_out))
            continue

        sample_indices = split_indices[split]
        logger.info(
            f"Writing '{split}' Arrow shard directly from BigWigs: {len(sample_indices)} samples, "
            f"{n_tracks} tracks, stored_context={stored_context} bp, "
            f"stored_n_bins={stored_n_bins}, batch_size={effective_arrow_batch}, "
            f"compression={arrow_compression}"
        )

        split_scratch = scratch_out / split
        if split_scratch.exists():
            shutil.rmtree(split_scratch)
        split_scratch.mkdir(parents=True, exist_ok=True)
        arrow_filename = "data-00000-of-00001.arrow"
        t_split_arrow = time.perf_counter()
        write_arrow_split_from_bigwigs(
            active_bw_paths,
            minus_flags,
            signal_intervals,
            str(split_scratch / arrow_filename),
            sample_indices,
            bed_rows,
            active_fasta,
            stored_n_bins,
            stored_context,
            batch_size=effective_arrow_batch,
            n_threads=n_extract_threads,
            compression=arrow_compression,
            profile=profile,
        )
        logger.info(f"Arrow shard '{split}' written in {time.perf_counter() - t_split_arrow:.1f}s")
        _write_hf_split_metadata(split_scratch, features=features, arrow_filename=arrow_filename)
        split_datasets[split] = Dataset.load_from_disk(str(split_scratch))
    logger.info(f"Arrow writing completed in {time.perf_counter() - t_arrow_total:.1f}s")

    (scratch_out / "dataset_dict.json").write_text(json.dumps({"splits": list(splits)}, indent=2))
    logger.info(f"Rsyncing dataset from scratch to {output_dir}")
    t_rsync = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", f"{scratch_out}/", f"{output_dir}/"],
        check=True,
    )
    logger.info(f"Rsync completed in {time.perf_counter() - t_rsync:.1f}s")
    shutil.rmtree(scratch_out)

    metadata = {
        "bigwig_paths": bw_paths,
        "bed_file": str(bed_file),
        "fasta_file": str(fasta_file),
        "context_length": context_length,
        "bin_size": bin_size,
        "n_pred_bins": n_pred_bins,
        "shift_max_bp": shift_max_bp,
        "splits": splits,
        "build_strategy": "fast",
    }
    (output_dir / "regulonado_metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"Dataset saved to {output_dir}")
    return DatasetDict.load_from_disk(str(output_dir))


# ---------------------------------------------------------------------------
# Read-time transform
# ---------------------------------------------------------------------------

def build_rc_permutation(
    track_records: list[dict],
    pairing_fields: tuple[str, ...] = ("condition_id", "cell_line_id", "assay_type_id", "target_id"),
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


def make_transform(
    scale_factors: np.ndarray,
    clip_soft: np.ndarray | float,
    clip_hard: np.ndarray | float,
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
) -> Callable[[dict], dict]:
    """Return a transform compatible with ``dataset.set_transform()``.

    Applied per batch:
        1. Random shift crop  (if shift_max_bins > 0)
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
    """
    sf = np.asarray(scale_factors, dtype=np.float32).reshape(-1)
    n_tracks = sf.size
    cs = np.broadcast_to(np.asarray(clip_soft, dtype=np.float32), (n_tracks,)).copy()
    ch = np.broadcast_to(np.asarray(clip_hard, dtype=np.float32), (n_tracks,)).copy()

    def _transform_signal(labels: np.ndarray, _sf: np.ndarray, _cs: np.ndarray, _ch: np.ndarray) -> np.ndarray:
        out = np.asarray(labels, dtype=np.float32).copy()
        np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.maximum(out, 0.0, out=out)
        if apply_scale:
            out *= _sf.reshape(-1, 1)
        if apply_clip:
            np.minimum(out, _ch.reshape(-1, 1), out=out)
        if apply_squash:
            out = (np.power(out + 1.0, 0.75) - 1.0).astype(np.float32, copy=False)
            if apply_clip:
                cs_squashed = (np.power(_cs.reshape(-1, 1) + 1.0, 0.75) - 1.0).astype(np.float32)
                mask = out > cs_squashed
                if mask.any():
                    out = np.where(
                        mask,
                        cs_squashed - 1.0 + np.sqrt(np.maximum(out - cs_squashed + 1.0, 0.0)),
                        out,
                    ).astype(np.float32)
        return out

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
            _sf, _cs, _ch = sf, cs, ch

            # --- shift crop (random offset into the wider stored arrays)
            if shift_max_bins > 0:
                s = int(np.random.randint(0, 2 * shift_max_bins + 1))
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
                    sig = sig.copy()

            if seq is not None:
                out_ids.append(seq)
            if sig is not None:
                out_lbl.append(_transform_signal(sig, _sf, _cs, _ch))

        if out_ids:
            batch["input_ids"] = np.stack(out_ids) if batched else out_ids[0]
        if out_lbl:
            batch["labels"] = np.stack(out_lbl) if batched else out_lbl[0]

        return batch

    return transform_batch
