"""Full-dataset benchmark: Rust phase-1 vs best Python baseline.

Measures:
  1. Rust extract_all_tracks_to_dir at multiple (tracks × samples) scales,
     both on Ceph and staged to $SLURM_TMPDIR/NVMe.
  2. Best Python baseline: batched serial (only valid strategy at 2299 tracks
     since that exceeds MAX_OPEN_READERS=256).
  3. Direct staged-vs-unstaged comparison to quantify NVMe seek benefit.

Designed to run as a SLURM job with NVMe scratch available:

    sbatch tests/bench_full_dataset.sh

or interactively on a node with $SLURM_TMPDIR set:

    SLURM_TMPDIR=/tmp .venv/bin/python tests/bench_full_dataset.py
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
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

N_ALL_TRACKS = 2_299
N_ALL_SAMPLES = 55_497

N_PRED_BINS = 6_144
BIN_SIZE = 32
PRED_BP = N_PRED_BINS * BIN_SIZE  # 196,608 bp

MAX_OPEN_READERS = 256

SCRATCH = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")

# Scaling ladder — bracketing up to full dataset
TRACK_COUNTS  = [100, 500, 1_000, 2_299]
SAMPLE_COUNTS = [100, 500, 2_000, 10_000, 41_699]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_intervals(n: int) -> list[tuple[str, int, int]]:
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


def stage_bigwigs(paths: list[str], dest: Path) -> list[str]:
    """Rsync BigWig files to dest (NVMe). Returns staged paths."""
    dest.mkdir(parents=True, exist_ok=True)
    total_gb = sum(Path(p).stat().st_size for p in paths) / 1e9
    print(f"  Staging {len(paths)} BigWigs ({total_gb:.1f} GB) → {dest} ...", flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        ["rsync", "-a", "--files-from=-", "/", str(dest)],
        input="\n".join(p.lstrip("/") for p in paths).encode(),
        check=True,
    )
    dt = time.perf_counter() - t0
    # Rebuild staged paths: dest + original relative structure
    staged = [str(dest / p.lstrip("/")) for p in paths]
    print(f"  Staged in {dt:.1f}s ({total_gb/dt:.1f} GB/s)")
    return staged


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _read_one(reader, chrom, start, end, n_bins):
    return np.asarray(reader.values(chrom, start, end, bins=n_bins, exact=True), dtype=np.float32)


def strategy_batched_serial(
    paths: list[str],
    intervals: list[tuple[str, int, int]],
    batch_size: int = MAX_OPEN_READERS,
) -> None:
    """Best Python baseline at 2299 tracks: open ≤256 handles at a time."""
    n_tracks = len(paths)
    n_samples = len(intervals)
    out = np.empty((n_samples, n_tracks, N_PRED_BINS), dtype=np.float32)
    for batch_start in range(0, n_tracks, batch_size):
        batch = paths[batch_start : batch_start + batch_size]
        readers = [pybigtools.open(p) for p in batch]
        try:
            for i, (chrom, start, end) in enumerate(intervals):
                for j, r in enumerate(readers):
                    out[i, batch_start + j] = _read_one(r, chrom, start, end, N_PRED_BINS)
        finally:
            for r in readers:
                try:
                    r.close()
                except Exception:
                    pass


def strategy_rust(
    paths: list[str],
    intervals: list[tuple[str, int, int]],
    tmp_dir: Path,
    n_threads: int = 32,
) -> None:
    """Rust phase-1: each BigWig opened once, all intervals read sequentially."""
    from regulonado._rs import extract_all_tracks_to_dir  # type: ignore[import]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_all_tracks_to_dir(paths, intervals, N_PRED_BINS, str(tmp_dir), n_threads=n_threads)


# ---------------------------------------------------------------------------
# Timing / reporting
# ---------------------------------------------------------------------------

def fmt(s: float) -> str:
    if s < 1:    return f"{s*1000:.0f}ms"
    if s < 60:   return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def extrapolate(t: float, n_samples: int, n_tracks: int) -> str:
    full_t = t * (N_ALL_SAMPLES / n_samples) * (N_ALL_TRACKS / n_tracks)
    return fmt(full_t)


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

def run_scale_benchmark(rust_tmp: Path, skip_python: bool = False):
    """Scaling ladder: Rust vs batched-serial at multiple track/sample counts."""
    print("\n" + "="*80)
    print("  SCALING BENCHMARK  (Rust vs best Python baseline — Ceph, no staging)")
    print("="*80)
    if skip_python:
        print(f"  {'tracks':>8}  {'samples':>8}  {'Rust (32t)':>12}  {'→ full (Rust)':>14}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*14}")
    else:
        print(f"  {'tracks':>8}  {'samples':>8}  {'Python batched':>16}  {'Rust (32t)':>12}  {'speedup':>8}  {'→ full (Rust)':>14}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*16}  {'-'*12}  {'-'*8}  {'-'*14}")

    for n_tracks in TRACK_COUNTS:
        for n_samples in SAMPLE_COUNTS:
            paths = load_bw_paths(n_tracks)
            intervals = load_intervals(n_samples)
            if len(paths) < n_tracks or not intervals:
                print(f"  {n_tracks:>8}  {n_samples:>8}  [skip — not enough data on disk]")
                continue

            rust_dir = rust_tmp / f"scale_{n_tracks}_{n_samples}"
            shutil.rmtree(rust_dir, ignore_errors=True)
            t_rs = timeit(strategy_rust, paths, intervals, rust_dir)
            shutil.rmtree(rust_dir, ignore_errors=True)

            if skip_python:
                print(f"  {n_tracks:>8}  {n_samples:>8}  {fmt(t_rs):>12}  {extrapolate(t_rs, n_samples, n_tracks):>14}")
            else:
                t_py = timeit(strategy_batched_serial, paths, intervals)
                speedup = t_py / t_rs if t_rs > 0 else float("inf")
                print(
                    f"  {n_tracks:>8}  {n_samples:>8}  {fmt(t_py):>16}  {fmt(t_rs):>12}  "
                    f"{speedup:>7.1f}×  {extrapolate(t_rs, n_samples, n_tracks):>14}"
                )
            sys.stdout.flush()


def run_staging_benchmark(rust_tmp: Path, skip_python: bool = False):
    """Staged vs unstaged: quantify NVMe seek benefit for a fixed track count.

    Uses 500 tracks (manageable staging time, still representative of seek load).
    """
    n_tracks = 500
    n_samples = 500
    print("\n" + "="*80)
    print(f"  STAGING BENCHMARK  ({n_tracks} tracks × {n_samples} samples)")
    print("="*80)

    ceph_paths = load_bw_paths(n_tracks)
    intervals = load_intervals(n_samples)
    if len(ceph_paths) < n_tracks or not intervals:
        print("  [skip] not enough data on disk")
        return

    stage_dir = SCRATCH / "bench_staged_bw"
    staged_paths = stage_bigwigs(ceph_paths, stage_dir)

    results: list[tuple[str, float]] = []

    for label, paths in [("Ceph (no staging)", ceph_paths), ("NVMe staged", staged_paths)]:
        if not skip_python:
            t_py = timeit(strategy_batched_serial, paths, intervals)
            results.append((f"Python batched  [{label}]", t_py))

        rust_dir = rust_tmp / f"stage_{label.replace(' ', '_')}"
        shutil.rmtree(rust_dir, ignore_errors=True)
        t_rs = timeit(strategy_rust, paths, intervals, rust_dir)
        shutil.rmtree(rust_dir, ignore_errors=True)
        results.append((f"Rust (32t)      [{label}]", t_rs))

    # Cleanup staged files
    shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"\n  {'Strategy':<52} {'Time':>8}  {'→ full dataset':>14}")
    print(f"  {'-'*52} {'-'*8}  {'-'*14}")
    for label, t in results:
        print(f"  {label:<52} {fmt(t):>8}  {extrapolate(t, n_samples, n_tracks):>14}")
    sys.stdout.flush()


def run_thread_scaling(rust_tmp: Path):
    """Rust thread-count scaling at 2299 tracks × 500 samples."""
    n_tracks = 2_299
    n_samples = 500
    thread_counts = [1, 4, 8, 16, 32, 64]

    print("\n" + "="*80)
    print(f"  RUST THREAD SCALING  ({n_tracks} tracks × {n_samples} samples, Ceph)")
    print("="*80)
    print(f"  {'n_threads':>10}  {'Time':>8}  {'→ full dataset':>14}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}")

    paths = load_bw_paths(n_tracks)
    intervals = load_intervals(n_samples)
    if len(paths) < n_tracks or not intervals:
        print("  [skip]")
        return

    for n_threads in thread_counts:
        rust_dir = rust_tmp / f"threads_{n_threads}"
        shutil.rmtree(rust_dir, ignore_errors=True)
        t = timeit(strategy_rust, paths, intervals, rust_dir, n_threads=n_threads)
        shutil.rmtree(rust_dir, ignore_errors=True)
        print(f"  {n_threads:>10}  {fmt(t):>8}  {extrapolate(t, n_samples, n_threads):>14}")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["scale", "staging", "threads"], help="Run only one section")
    parser.add_argument("--no-staging", action="store_true", help="Skip the staging benchmark (saves time if not on NVMe node)")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python baseline (use when pybigtools chokes at 2299 tracks)")
    args = parser.parse_args()

    rust_tmp = SCRATCH / "bench_rust_out"
    rust_tmp.mkdir(parents=True, exist_ok=True)

    print(f"BigWig full-dataset benchmark")
    print(f"BED:     {BED_FILE}")
    print(f"BWs:     {BW_LIST}  ({N_ALL_TRACKS} tracks)")
    print(f"Scratch: {SCRATCH}")

    skip_python = args.skip_python

    try:
        if args.only == "scale" or args.only is None:
            run_scale_benchmark(rust_tmp, skip_python=skip_python)
        if args.only == "staging" or (args.only is None and not args.no_staging):
            run_staging_benchmark(rust_tmp, skip_python=skip_python)
        if args.only == "threads" or args.only is None:
            run_thread_scaling(rust_tmp)
    finally:
        shutil.rmtree(rust_tmp, ignore_errors=True)
