"""End-to-end benchmark for the Rust-backed Arrow dataset builder.

This exercises the real fast path:

  BED + FASTA + BigWigs
      -> direct Rust BigWig/FASTA Arrow IPC shard writing
      -> HuggingFace load_from_disk validation

Designed for quick interactive smoke tests and larger SLURM runs:

    .venv/bin/python tests/bench_arrow_build.py --tracks 25 --samples 25
    sbatch tests/bench_arrow_build.sh
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).parent.parent
BED_FILE = Path(
    os.environ.get(
        "BED_FILE",
        "/project/milne_group/asmith/Projects/2025-07-19-myeloid-specific-enhancer-identification/data/external/sequences_human.bed.gz",
    )
)
FASTA_FILE = Path(
    os.environ.get(
        "FASTA_FILE",
        "/ceph/project/milne_group/shared/seqnado_reference/hg38/UCSC/sequence/hg38.fa",
    )
)
BW_LIST = Path(os.environ.get("BIGWIG_LIST", REPO / "notebooks" / "2026-05-20-dataset-paths.txt"))
SCRATCH = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")

N_ALL_TRACKS = 2_299
N_ALL_SAMPLES = 55_497


def fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_bw_paths(n: int) -> list[str]:
    paths: list[str] = []
    with open(BW_LIST) as fh:
        for line in fh:
            path = line.strip().strip('"').strip("'")
            if path and Path(path).exists():
                paths.append(path)
                if len(paths) >= n:
                    break
    if len(paths) < n:
        raise RuntimeError(f"Only found {len(paths)}/{n} existing BigWig paths in {BW_LIST}")
    return paths


def write_subset_bed(src: Path, dest: Path, n: int) -> None:
    opener = gzip.open if str(src).endswith(".gz") else open
    written = 0
    with opener(src, "rt") as inp, open(dest, "wt") as out:
        for line in inp:
            if not line.strip() or line.startswith("#"):
                continue
            out.write(line)
            written += 1
            if written >= n:
                break
    if written < n:
        raise RuntimeError(f"Only wrote {written}/{n} BED rows from {src}")


def build_cmd(args: argparse.Namespace, bed_subset: Path, output_dir: Path, bw_paths: list[str]) -> list[str]:
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        "-m",
        "regulonado",
        "build",
        str(bed_subset),
        str(FASTA_FILE),
        str(output_dir),
        "--split",
        "bench:",
        "--overwrite",
        "--drop-missing",
        "--context-length",
        str(args.context_length),
        "--bin-size",
        str(args.bin_size),
        "--n-pred-bins",
        str(args.n_pred_bins),
        "--shift-max-bp",
        str(args.shift_max_bp),
        "--num-proc",
        str(args.num_proc),
        "--n-extract-threads",
        str(args.extract_threads),
        "--signal-sample-chunk",
        str(args.signal_sample_chunk),
        "--signal-track-chunk",
        str(args.signal_track_chunk),
        "--arrow-batch-size",
        str(args.arrow_batch_size),
        "--arrow-compression",
        args.arrow_compression,
    ]
    if args.stage:
        cmd.append("--stage")
    if args.profile:
        cmd.append("--profile")
    for path in bw_paths:
        cmd.extend(["--bigwig", path])
    return cmd


def run_and_capture(cmd: list[str]) -> tuple[float, str]:
    lines: list[str] = []
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    rc = proc.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return elapsed, "".join(lines)


def parse_phase_seconds(log_text: str) -> dict[str, float]:
    patterns = {
        "extract": r"Phase 1 extraction completed in ([0-9.]+)s",
        "arrow": r"(?:Phase 2 )?Arrow writing completed in ([0-9.]+)s",
        "rsync": r"Rsync completed in ([0-9.]+)s",
    }
    out: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            out[name] = float(match.group(1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=int, default=int(os.environ.get("TRACKS", "50")))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("SAMPLES", "50")))
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("CONTEXT_LENGTH", "524288")))
    parser.add_argument("--bin-size", type=int, default=int(os.environ.get("BIN_SIZE", "32")))
    parser.add_argument("--n-pred-bins", type=int, default=int(os.environ.get("N_PRED_BINS", "6144")))
    parser.add_argument("--shift-max-bp", type=int, default=int(os.environ.get("SHIFT_MAX_BP", "0")))
    parser.add_argument("--extract-threads", type=int, default=int(os.environ.get("N_EXTRACT_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "32"))))
    parser.add_argument("--num-proc", type=int, default=int(os.environ.get("NUM_PROC", "1")))
    parser.add_argument("--signal-sample-chunk", type=int, default=int(os.environ.get("SIGNAL_SAMPLE_CHUNK", "8")))
    parser.add_argument("--signal-track-chunk", type=int, default=int(os.environ.get("SIGNAL_TRACK_CHUNK", "128")))
    parser.add_argument("--arrow-batch-size", type=int, default=int(os.environ.get("ARROW_BATCH_SIZE", "512")))
    parser.add_argument("--arrow-compression", default=os.environ.get("ARROW_COMPRESSION", "zstd"))
    parser.add_argument("--stage", action="store_true", default=os.environ.get("STAGE", "false").lower() == "true")
    parser.add_argument("--profile", action="store_true", default=os.environ.get("PROFILE", "false").lower() == "true")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_root = SCRATCH / f"regulonado_arrow_bench_{os.getpid()}"
    output_dir = args.output_dir or run_root / "dataset"
    bed_subset = run_root / f"subset_{args.samples}.bed"

    print("Rust Arrow dataset benchmark")
    print(f"Repo        : {REPO}")
    print(f"BED         : {BED_FILE}")
    print(f"FASTA       : {FASTA_FILE}")
    print(f"BigWig list : {BW_LIST}")
    print(f"Scratch     : {SCRATCH}")
    print(f"Tracks      : {args.tracks}")
    print(f"Samples     : {args.samples}")
    print(f"Output      : {output_dir}")
    print(f"Stage files : {args.stage}")
    print("", flush=True)

    shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        bw_paths = load_bw_paths(args.tracks)
        write_subset_bed(BED_FILE, bed_subset, args.samples)

        cmd = build_cmd(args, bed_subset, output_dir, bw_paths)
        print("Command:")
        print("  " + " ".join(cmd[:18]) + f" ... --bigwig × {len(bw_paths)}")
        print("", flush=True)

        elapsed, build_log = run_and_capture(cmd)
        phases = parse_phase_seconds(build_log)

        ds = load_from_disk(str(output_dir))
        n_rows = len(ds["bench"])
        first = ds["bench"][0]
        labels_shape = (len(first["labels"]), len(first["labels"][0]) if first["labels"] else 0)
        out_gb = dir_size(output_dir) / 1e9

        units = args.tracks * args.samples
        full_est = elapsed * (N_ALL_TRACKS * N_ALL_SAMPLES / units)
        measured_phase_total = sum(phases.values())
        fixed_overhead = max(0.0, elapsed - measured_phase_total)
        phase_full_est = measured_phase_total * (N_ALL_TRACKS * N_ALL_SAMPLES / units)
        print("\nResult")
        print(f"  elapsed             : {fmt(elapsed)}")
        if phases:
            print(f"  measured phases     : {fmt(measured_phase_total)}")
            for name in ("extract", "arrow", "rsync"):
                if name in phases:
                    print(f"    {name:<7}          : {fmt(phases[name])}")
            print(f"  fixed overhead      : {fmt(fixed_overhead)}")
        print(f"  sample-track units  : {units:,}")
        print(f"  throughput          : {units / elapsed:,.1f} sample-tracks/s")
        if measured_phase_total > 0:
            print(f"  phase throughput    : {units / measured_phase_total:,.1f} sample-tracks/s")
        print(f"  output size         : {out_gb:.2f} GB")
        print(f"  loaded rows         : {n_rows:,}")
        print(f"  labels shape        : {labels_shape}")
        print(f"  rough full estimate : {fmt(full_est)}  (includes fixed startup)")
        if measured_phase_total > 0:
            print(f"  phase full estimate : {fmt(phase_full_est)}  (usually more meaningful)")
    finally:
        if not args.keep_output and args.output_dir is None:
            shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
