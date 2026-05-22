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
import hashlib
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
_DEFAULT_ARROW_COMPRESSION = "lz4"
_DEDUPE_TRACK_MODES = {"none", "identity", "content"}

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
        futs = {pool.submit(_rsync_one, src, scratch, companion_suffixes): src for src in unique_srcs}
        staged_map: dict[str, str] = {}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            src = futs[fut]
            staged_map[str(src)] = fut.result()
            if done % 50 == 0 or done == n:
                logger.info(f"  staged {done}/{n}")
    # preserve input order
    return [staged_map[str(Path(p))] for p in paths]


# ---------------------------------------------------------------------------
# Track filtering/provenance
# ---------------------------------------------------------------------------

def _hash_file_blake2b(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _track_file_record(source_index: int, path: str) -> dict:
    p = Path(path).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p.absolute()

    try:
        st = resolved.stat()
        size_bytes = int(st.st_size)
        identity_key = f"inode:{st.st_dev}:{st.st_ino}"
    except OSError:
        size_bytes = None
        identity_key = f"path:{resolved}"

    return {
        "source_index": source_index,
        "path": path,
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "identity_key": identity_key,
    }


def _resolve_bigwig_tracks(
    bigwig_paths: Sequence[str | Path],
    *,
    drop_missing: bool,
    dedupe_tracks: str,
) -> tuple[list[str], dict]:
    """Filter requested tracks and return final paths plus provenance metadata."""
    if dedupe_tracks not in _DEDUPE_TRACK_MODES:
        raise ValueError(
            f"dedupe_tracks must be one of {sorted(_DEDUPE_TRACK_MODES)}, got {dedupe_tracks!r}"
        )

    requested = [str(p).strip().strip('"').strip("'") for p in bigwig_paths]
    existing: list[dict] = []
    missing_records: list[dict] = []
    for source_index, path in enumerate(requested):
        if Path(path).expanduser().exists():
            existing.append(_track_file_record(source_index, path))
        else:
            missing_records.append({"source_index": source_index, "path": path})

    if missing_records:
        if drop_missing:
            logger.warning(
                f"Dropping {len(missing_records)}/{len(requested)} missing bigwig paths:\n"
                + "\n".join(f"  {r['path']}" for r in missing_records)
            )
        else:
            raise FileNotFoundError(
                f"{len(missing_records)}/{len(requested)} bigwig paths do not exist:\n"
                + "\n".join(f"  {r['path']}" for r in missing_records)
            )

    identity_canonical: dict[int, int] = {}
    identity_first: dict[str, int] = {}
    survivors: list[dict] = []
    n_identity_dropped = 0
    if dedupe_tracks in {"identity", "content"}:
        for rec in existing:
            source_index = int(rec["source_index"])
            key = str(rec["identity_key"])
            if key in identity_first:
                identity_canonical[source_index] = identity_first[key]
                n_identity_dropped += 1
            else:
                identity_first[key] = source_index
                identity_canonical[source_index] = source_index
                survivors.append(rec)
    else:
        for rec in existing:
            source_index = int(rec["source_index"])
            identity_canonical[source_index] = source_index
            survivors.append(rec)

    content_canonical: dict[int, int] = {int(rec["source_index"]): int(rec["source_index"]) for rec in survivors}
    content_hash_by_source: dict[int, str] = {}
    n_hashed = 0
    n_content_dropped = 0
    hash_algorithm = "blake2b-256"
    if dedupe_tracks == "content":
        by_size: dict[int, list[dict]] = {}
        for rec in survivors:
            size = rec.get("size_bytes")
            if size is not None:
                by_size.setdefault(int(size), []).append(rec)

        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            first_for_hash: dict[str, int] = {}
            for rec in same_size:
                source_index = int(rec["source_index"])
                digest = _hash_file_blake2b(Path(str(rec["resolved_path"])))
                content_hash_by_source[source_index] = digest
                n_hashed += 1
                if digest in first_for_hash:
                    content_canonical[source_index] = first_for_hash[digest]
                    n_content_dropped += 1
                else:
                    first_for_hash[digest] = source_index

    final_canonical: dict[int, int] = {}
    for rec in existing:
        source_index = int(rec["source_index"])
        identity_source = identity_canonical[source_index]
        final_canonical[source_index] = content_canonical.get(identity_source, identity_source)

    final_source_indices = {
        source_index for source_index, canonical in final_canonical.items() if source_index == canonical
    }
    final_records: list[dict] = []
    final_track_index_by_source: dict[int, int] = {}
    rec_by_source = {int(rec["source_index"]): rec for rec in existing}
    for rec in existing:
        source_index = int(rec["source_index"])
        if source_index not in final_source_indices:
            continue
        track_index = len(final_records)
        final_track_index_by_source[source_index] = track_index
        content_hash = content_hash_by_source.get(source_index)
        dedupe_method = "content" if content_hash is not None else (
            "identity" if dedupe_tracks in {"identity", "content"} else "none"
        )
        dedupe_key = f"content:{content_hash}" if content_hash is not None else str(rec["identity_key"])
        out = {
            "track_index": track_index,
            "source_index": source_index,
            "path": rec["path"],
            "resolved_path": rec["resolved_path"],
            "size_bytes": rec["size_bytes"],
            "dedupe_key": dedupe_key,
            "dedupe_method": dedupe_method,
        }
        if content_hash is not None:
            out["content_hash"] = content_hash
        final_records.append(out)

    dropped_records: list[dict] = []
    for rec in existing:
        source_index = int(rec["source_index"])
        if source_index in final_source_indices:
            continue
        duplicate_of_source_index = final_canonical[source_index]
        identity_source = identity_canonical[source_index]
        content_hash = content_hash_by_source.get(identity_source)
        dedupe_method = "identity" if identity_source != source_index else "content"
        dropped = {
            "source_index": source_index,
            "path": rec["path"],
            "resolved_path": rec["resolved_path"],
            "size_bytes": rec["size_bytes"],
            "duplicate_of_track_index": final_track_index_by_source[duplicate_of_source_index],
            "duplicate_of_source_index": duplicate_of_source_index,
            "dedupe_method": dedupe_method,
            "dedupe_key": (
                f"content:{content_hash}" if dedupe_method == "content" and content_hash is not None
                else str(rec["identity_key"])
            ),
        }
        if dedupe_method == "content" and content_hash is not None:
            dropped["content_hash"] = content_hash
        duplicate_of = rec_by_source[duplicate_of_source_index]
        dropped["duplicate_of_path"] = duplicate_of["path"]
        dropped["duplicate_of_resolved_path"] = duplicate_of["resolved_path"]
        dropped_records.append(dropped)

    final_paths = [str(rec["path"]) for rec in final_records]
    if dedupe_tracks != "none":
        logger.info(
            f"Track dedupe mode={dedupe_tracks}: {len(final_paths)} final track(s) from "
            f"{len(requested)} requested; dropped {len(dropped_records)} duplicate(s) "
            f"({n_identity_dropped} identity, {n_content_dropped} content); hashed {n_hashed} file(s)"
        )

    provenance = {
        "bigwig_paths": final_paths,
        "final_bigwig_paths": final_paths,
        "requested_bigwig_paths": requested,
        "final_track_records": final_records,
        "dropped_duplicate_tracks": dropped_records,
        "missing_bigwig_paths": missing_records,
        "n_requested_tracks": len(requested),
        "n_missing_tracks": len(missing_records),
        "n_dropped_duplicate_tracks": len(dropped_records),
        "n_final_tracks": len(final_paths),
        "dedupe_tracks": {
            "mode": dedupe_tracks,
            "keep": "first",
            "identity_method": "stat(st_dev,st_ino) after resolve; fallback resolved_path",
            "hash_algorithm": hash_algorithm if dedupe_tracks == "content" else None,
            "hash_limited_to_same_size_groups": dedupe_tracks == "content",
            "n_hashed_files": n_hashed,
            "n_identity_duplicates": n_identity_dropped,
            "n_content_duplicates": n_content_dropped,
        },
    }
    return final_paths, provenance


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


def _is_remote_fs(path: Path) -> bool:
    resolved = str(path.resolve())
    return resolved.startswith("/ceph") or resolved.startswith("/project")


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
            for local_idx, index in enumerate(indices):
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
                        "local_index": np.int64(local_idx),
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
    dedupe_tracks: str = "none",
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
        dedupe_tracks: Track deduplication mode: "none", "identity", or "content".
            Content mode first deduplicates filesystem identity, then hashes
            same-size candidates and drops byte-identical duplicates.

    Returns:
        The loaded DatasetDict (also saved to disk at output_dir).
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    if shift_max_bp % bin_size != 0:
        raise ValueError(f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})")

    output_dir = Path(output_dir)
    bw_paths, track_metadata = _resolve_bigwig_tracks(
        bigwig_paths,
        drop_missing=drop_missing,
        dedupe_tracks=dedupe_tracks,
    )
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
        "local_index": Value(dtype="int64"),
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
        **track_metadata,
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
    arrow_write_threads: int | None = None,
    num_proc: int = 1,
    cache_dir: str | None = None,
    writer_batch_size: int = 500,
    stage_to_scratch: bool = False,
    overwrite: bool = False,
    drop_missing: bool = False,
    dedupe_tracks: str = "none",
    profile: bool = False,
    strategy: str = "chrom_pass",
    chrom_filter: list[str] | None = None,
) -> DatasetDict:
    """Fast low-scratch dataset build using the Rust extension.

    Each split is written directly from BigWig + FASTA sources into compressed
    Arrow shards. Peak scratch is the current Arrow output plus one in-memory
    record batch, not a full dense `(tracks, samples, bins)` signal file.

    Parameters
    ----------
    strategy : {"chrom_pass", "fast"}
        - "chrom_pass" (default): one chromosome at a time, decoding all
          tracks' binned signal once per chromosome and slicing per-sample
          rows from RAM. One Arrow shard per chromosome, ordered chrom-major
          (largest chromosome first). Rows within each shard are in
          original BED order. ~10× fewer BigWig seeks than "fast".
        - "fast": sample-batched writer; reads each sample's interval from
          every BigWig per batch. Retained for parity testing.
    chrom_filter : list[str] | None
        If given, restrict each split to BED rows on these chromosomes.
        ``bed_rows`` is *not* renumbered — the ``index`` column on every
        output row remains the absolute row position in the input BED
        file. Useful for smoke tests on a single chromosome.
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    if shift_max_bp % bin_size != 0:
        raise ValueError(f"shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})")

    bed_file = Path(bed_file)
    output_dir = Path(output_dir)
    bw_paths, track_metadata = _resolve_bigwig_tracks(
        bigwig_paths,
        drop_missing=drop_missing,
        dedupe_tracks=dedupe_tracks,
    )
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
    if _is_remote_fs(scratch_out):
        logger.warning(
            f"scratch_out resolves to {str(scratch_out.resolve())!r} — Arrow I/O will hit remote storage"
        )
    if _is_remote_fs(output_dir):
        logger.info(
            f"output_dir resolves to {str(output_dir.resolve())!r}; only the final rsync should hit remote storage"
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
    label_batch_gb = effective_arrow_batch * n_tracks * stored_n_bins * 4 / 1e9
    seq_batch_gb = effective_arrow_batch * 4 * stored_context / 1e9
    effective_arrow_write_threads = (
        4
        if arrow_write_threads is None
        else max(1, arrow_write_threads)
    )
    chrom_pass_writer_peak_gb = effective_arrow_write_threads * (
        label_batch_gb + seq_batch_gb
    )
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
        binning_scratch_gb = n_extract_threads * largest_chrom_bins * (8 + 8 + bin_size * 4) / 1e9
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

    features = Features({
        "input_ids": Array2D(dtype="int8", shape=(4, stored_context)),
        "labels": Array2D(dtype="float32", shape=(n_tracks, stored_n_bins)),
        "interval": Value(dtype="string"),
        "index": Value(dtype="int64"),
        "local_index": Value(dtype="int64"),
    })

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
                f"Split '{split}': {len(idx)} samples (filtered to chroms {sorted(chrom_filter_set)})"
            )
        else:
            logger.info(f"Split '{split}': {len(idx)} samples")

    split_datasets: dict[str, Dataset] = {}
    if not splits_to_build:
        logger.info("All splits already exist; loading from disk")
        for split in splits:
            split_datasets[split] = Dataset.load_from_disk(str(output_dir / split))
        return DatasetDict(split_datasets)

    if strategy not in {"chrom_pass", "fast"}:
        raise ValueError(f"strategy must be 'chrom_pass' or 'fast', got {strategy!r}")
    from regulonado._rs import (  # type: ignore[import]
        write_arrow_splits_chrom_pass,
        write_arrow_split_from_bigwigs,
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
                logger.info(f"Split '{split}' exists; loading from disk")
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
                n_threads=n_extract_threads,
                arrow_write_threads=effective_arrow_write_threads,
                compression=arrow_compression,
                profile=profile,
            )

        for split, split_scratch_str in zip(chrom_split_names, chrom_split_out_dirs, strict=True):
            split_scratch = Path(split_scratch_str)
            arrow_filenames = sorted(
                p.name for p in split_scratch.glob("data-*-of-*.arrow")
            )
            if not arrow_filenames:
                raise RuntimeError(
                    f"chrom_pass produced no shards in {split_scratch}; "
                    f"check that the FASTA contains the BED chromosomes"
                )
            logger.info(
                f"Arrow shard(s) for '{split}' written ({len(arrow_filenames)} file(s))"
            )
            _write_hf_split_metadata(
                split_scratch, features=features, arrow_filenames=arrow_filenames
            )
            split_datasets[split] = Dataset.load_from_disk(str(split_scratch))
    else:
        for split, folds in splits.items():
            split_out = output_dir / split
            if not overwrite and (split_out / "dataset_info.json").exists():
                logger.info(f"Split '{split}' exists; loading from disk")
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
        **track_metadata,
        "bed_file": str(bed_file),
        "fasta_file": str(fasta_file),
        "context_length": context_length,
        "bin_size": bin_size,
        "n_pred_bins": n_pred_bins,
        "shift_max_bp": shift_max_bp,
        "splits": splits,
        "build_strategy": strategy,
        "arrow_write_threads": effective_arrow_write_threads,
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
