"""Benchmark bigwig reading strategies to quantify the dataset-build bottleneck.

Run with:
    cd /ceph/project/milne_group/asmith/software/Regulonado
    .venv/bin/python tests/bench_bw_reading.py

Key findings from initial run:
- ThreadPoolExecutor HURTS: more threads = more overhead, not faster reads
- ThreadPoolExecutor(1) beats ThreadPoolExecutor(8) by ~2x at 100 tracks
- Threading overhead from 2299 futures/sample × 41699 samples is the bottleneck
- Sequential single-call loop is nearly as fast as threaded, sometimes faster
- >256 concurrent open file handles on Ceph causes open() bottleneck

Goal: the Rust extension (regulonado_rs) should replace _read_all_tracks with a
rayon-based impl that has zero Python per-track overhead (no futures, no GIL).
"""

from __future__ import annotations

import gzip
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pybigtools

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
BED_FILE = Path("/project/milne_group/asmith/Projects/2025-07-19-myeloid-specific-enhancer-identification/data/external/sequences_human.bed.gz")
BW_LIST = REPO / "notebooks" / "2026-05-20-dataset-paths.txt"

N_PRED_BINS = 6_144
BIN_SIZE = 32
PRED_BP = N_PRED_BINS * BIN_SIZE  # 196,608 bp

# Never open more than this many file handles simultaneously on Ceph
MAX_OPEN_READERS = 256

# Benchmark matrix — keep small enough to finish in <60s total
TRACK_COUNTS = [10, 50, 100, 500, 2_299]
SAMPLE_COUNTS = [10, 50, 100]
THREAD_COUNTS = [1, 8, 32]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_intervals(n: int) -> list[tuple[str, int, int]]:
    """Return first n intervals (chrom, sig_start, sig_end) from the BED file."""
    intervals: list[tuple[str, int, int]] = []
    with gzip.open(BED_FILE, "rt") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            center = (start + end) // 2
            sig_start = center - PRED_BP // 2
            sig_end = sig_start + PRED_BP
            intervals.append((chrom, sig_start, sig_end))
            if len(intervals) >= n:
                break
    return intervals


def load_bw_paths(n: int) -> list[str]:
    """Return first n bigwig paths that exist on disk."""
    paths: list[str] = []
    with open(BW_LIST) as fh:
        for line in fh:
            p = line.strip()
            if p and Path(p).exists():
                paths.append(p)
                if len(paths) >= n:
                    break
    if len(paths) < n:
        print(f"  [warn] only {len(paths)}/{n} bigwig paths found on disk")
    return paths


def open_readers(paths: list[str]) -> list:
    """Open at most MAX_OPEN_READERS file handles at a time to avoid Ceph overload."""
    if len(paths) > MAX_OPEN_READERS:
        raise ValueError(
            f"open_readers: {len(paths)} > MAX_OPEN_READERS={MAX_OPEN_READERS}. "
            "Use a batched strategy instead."
        )
    return [pybigtools.open(p) for p in paths]


def close_readers(readers: list) -> None:
    for r in readers:
        try:
            r.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reading strategies
# ---------------------------------------------------------------------------

def _read_one(reader, chrom: str, start: int, end: int, n_bins: int) -> np.ndarray:
    return np.asarray(
        reader.values(chrom, start, end, bins=n_bins, exact=True),
        dtype=np.float32,
    )


def strategy_sequential(
    readers: list,
    intervals: list[tuple[str, int, int]],
) -> np.ndarray:
    """Pure serial: one read at a time, no threading overhead at all."""
    n_tracks = len(readers)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, N_PRED_BINS), dtype=np.float32)
    for i, (chrom, start, end) in enumerate(intervals):
        for j, r in enumerate(readers):
            out[i, j] = _read_one(r, chrom, start, end, N_PRED_BINS)
    return out


def strategy_thread_pool(
    readers: list,
    intervals: list[tuple[str, int, int]],
    n_threads: int,
) -> np.ndarray:
    """Current approach: submit all tracks per sample to a shared thread pool.

    Overhead: n_tracks future objects created and collected per sample.
    With 2299 tracks × 41699 samples = 95.8M Future objects total.
    """
    n_tracks = len(readers)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, N_PRED_BINS), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for i, (chrom, start, end) in enumerate(intervals):
            futs = [ex.submit(_read_one, r, chrom, start, end, N_PRED_BINS) for r in readers]
            for j, f in enumerate(futs):
                out[i, j] = f.result()
    return out


def strategy_batched(
    paths: list[str],
    intervals: list[tuple[str, int, int]],
    batch_size: int = MAX_OPEN_READERS,
) -> np.ndarray:
    """Process tracks in batches to cap concurrent open file handles.

    Opens ≤ batch_size bigwigs at once, reads all samples for them, closes,
    then moves to the next batch. Avoids Ceph file-handle exhaustion.
    """
    n_tracks = len(paths)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, N_PRED_BINS), dtype=np.float32)
    for batch_start in range(0, n_tracks, batch_size):
        batch_paths = paths[batch_start : batch_start + batch_size]
        readers = open_readers(batch_paths)
        try:
            for i, (chrom, start, end) in enumerate(intervals):
                for j, r in enumerate(readers):
                    out[i, batch_start + j] = _read_one(r, chrom, start, end, N_PRED_BINS)
        finally:
            close_readers(readers)
    return out


def strategy_rust_per_track(
    paths: list[str],
    intervals: list[tuple[str, int, int]],
    n_bins: int,
    tmp_dir: Path,
    n_threads: int = 32,
) -> np.ndarray:
    """Phase 1 only — Rust rayon extraction, one file per track.

    Opens each BigWig once, reads all intervals sequentially, writes
    ``tmp_dir/track_{j:06d}.bin`` (float32 row-major LE).  Reads back
    memmaps to verify and return the same (n_samples, n_tracks, n_bins) array
    as the Python strategies so results are comparable.
    """
    from regulonado._rs import extract_all_tracks_to_dir  # type: ignore[import]

    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_all_tracks_to_dir(paths, intervals, n_bins, str(tmp_dir), n_threads=n_threads)

    n_tracks = len(paths)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, n_bins), dtype=np.float32)
    for j in range(n_tracks):
        mm = np.memmap(
            tmp_dir / f"track_{j:06d}.bin",
            dtype="<f4",
            mode="r",
            shape=(n_samples, n_bins),
        )
        out[:, j, :] = mm[:]
    return out


def strategy_batched_threads(
    paths: list[str],
    intervals: list[tuple[str, int, int]],
    n_threads: int,
    batch_size: int = MAX_OPEN_READERS,
) -> np.ndarray:
    """Batched open + ThreadPoolExecutor within each batch.

    Caps open handles at batch_size while still using threads for concurrent
    reads within a batch.
    """
    n_tracks = len(paths)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, N_PRED_BINS), dtype=np.float32)
    for batch_start in range(0, n_tracks, batch_size):
        batch_paths = paths[batch_start : batch_start + batch_size]
        readers = open_readers(batch_paths)
        try:
            with ThreadPoolExecutor(max_workers=min(n_threads, len(readers))) as ex:
                for i, (chrom, start, end) in enumerate(intervals):
                    futs = [
                        ex.submit(_read_one, r, chrom, start, end, N_PRED_BINS)
                        for r in readers
                    ]
                    for j, f in enumerate(futs):
                        out[i, batch_start + j] = f.result()
        finally:
            close_readers(readers)
    return out


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timeit(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds/60:.1f}min"


def extrapolate(t: float, n_samples_bench: int, n_tracks_bench: int) -> str:
    """Extrapolate to full dataset: 2,299 tracks × 41,699 samples."""
    full_t = t * (41_699 / n_samples_bench) * (2_299 / n_tracks_bench)
    return fmt(full_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(n_tracks: int, n_samples: int):
    print(f"\n{'='*70}")
    print(f"  {n_tracks} tracks × {n_samples} samples")
    print(f"{'='*70}")

    paths = load_bw_paths(n_tracks)
    intervals = load_intervals(n_samples)
    if len(paths) < n_tracks or not intervals:
        print("  [skip] not enough data on disk")
        return

    print(f"\n  {'Strategy':<48} {'Time':>8}  {'→ full dataset':>14}")
    print(f"  {'-'*48} {'-'*8}  {'-'*14}")

    baseline_t: float | None = None

    if n_tracks <= MAX_OPEN_READERS:
        readers = open_readers(paths)

        t, _ = timeit(strategy_sequential, readers, intervals)
        print(f"  {'sequential (no threads)':<48} {fmt(t):>8}  {extrapolate(t, n_samples, n_tracks):>14}")

        for n_threads in THREAD_COUNTS:
            name = f"ThreadPoolExecutor({n_threads})  [current@8]" if n_threads == 8 else f"ThreadPoolExecutor({n_threads})"
            t, _ = timeit(strategy_thread_pool, readers, intervals, n_threads)
            print(f"  {name:<48} {fmt(t):>8}  {extrapolate(t, n_samples, n_tracks):>14}")
            if n_threads == 1:
                baseline_t = t

        close_readers(readers)

    # Batched strategies (work at any track count)
    t, _ = timeit(strategy_batched, paths, intervals)
    print(f"  {'batched serial (≤256 handles)':<48} {fmt(t):>8}  {extrapolate(t, n_samples, n_tracks):>14}")
    if baseline_t is None:
        baseline_t = t

    for n_threads in THREAD_COUNTS:
        name = f"batched + ThreadPoolExecutor({n_threads})"
        t, _ = timeit(strategy_batched_threads, paths, intervals, n_threads)
        print(f"  {name:<48} {fmt(t):>8}  {extrapolate(t, n_samples, n_tracks):>14}")

    # Rust phase-1 strategy
    import tempfile
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)
        try:
            t_rust, _ = timeit(strategy_rust_per_track, paths, intervals, N_PRED_BINS, tmp_dir)
            speedup = baseline_t / t_rust if t_rust > 0 else float("inf")
            name = f"Rust extract_all_tracks_to_dir (n_threads=32)"
            print(
                f"  {name:<48} {fmt(t_rust):>8}  {extrapolate(t_rust, n_samples, n_tracks):>14}"
                f"  [{speedup:.1f}× vs ThreadPoolExecutor(1)]"
            )
        except Exception as exc:
            print(f"  {'Rust strategy FAILED':<48}  {exc}")


if __name__ == "__main__":
    import sys
    print("BigWig reading benchmark — using real data")
    print(f"BED: {BED_FILE}")
    print(f"BWs: {BW_LIST}")
    print(f"Region size: {PRED_BP:,} bp → {N_PRED_BINS} bins at {BIN_SIZE} bp/bin")
    print(f"MAX_OPEN_READERS: {MAX_OPEN_READERS}")

    # Accept optional track/sample counts from CLI for quick targeted runs
    # e.g.:  python bench_bw_reading.py 100 50
    if len(sys.argv) == 3:
        run_benchmark(int(sys.argv[1]), int(sys.argv[2]))
    else:
        for n_tracks in TRACK_COUNTS:
            for n_samples in SAMPLE_COUNTS:
                run_benchmark(n_tracks, n_samples)
