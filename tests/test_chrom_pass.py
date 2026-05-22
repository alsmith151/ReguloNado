"""Integration test for the chrom_pass Arrow writer.

Builds a tiny synthetic dataset (2 chromosomes × 3 BigWigs × 8 intervals) and
verifies that the chrom_pass writer produces bit-identical labels to the
existing per-sample direct-bigwig writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import pybigtools
import pytest


# Tiny but valid: bin_size 8, two chromosomes 4 kb each, 8 intervals of 512 bp.
N_TRACKS = 3
CHROM_LEN = 4096
BIN_SIZE = 8
N_PRED_BINS = 64  # interval bp = N_PRED_BINS * BIN_SIZE = 512
INTERVAL_BP = N_PRED_BINS * BIN_SIZE
CONTEXT_LEN = INTERVAL_BP  # no shift augmentation


@pytest.fixture(scope="module")
def synth_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    chroms = [("chrA", CHROM_LEN), ("chrB", CHROM_LEN)]

    # FASTA: deterministic "A" with periodic markers so one-hot encoding is non-trivial.
    fasta_path = root / "ref.fa"
    with fasta_path.open("w") as fh:
        for name, n in chroms:
            fh.write(f">{name}\n")
            seq = ("ACGT" * ((n + 3) // 4))[:n]
            for i in range(0, n, 80):
                fh.write(seq[i : i + 80] + "\n")
    # Build the .fai (pyfaidx writes the index on access).
    import pyfaidx

    pyfaidx.Fasta(str(fasta_path))

    # BigWigs: each track is a different deterministic sinusoid-like pattern.
    bw_paths: list[str] = []
    rng = np.random.default_rng(0)
    for t in range(N_TRACKS):
        path = root / f"track_{t}.bw"
        chromsize_map = {name: n for name, n in chroms}
        entries: list[tuple[str, int, int, float]] = []
        for name, n in chroms:
            xs = np.arange(n)
            vals = (
                np.sin(2 * np.pi * (xs + 17 * t) / 256.0) * 5.0
                + 10.0
                + t * 2.0
                + rng.normal(0, 0.1, size=n)
            ).astype(np.float32)
            entries.extend((name, int(i), int(i + 1), float(v)) for i, v in enumerate(vals))
        w = pybigtools.open(str(path), "w")
        w.write(chromsize_map, iter(entries))
        bw_paths.append(str(path))

    # BED rows: 4 intervals per chrom at varied offsets, interleaved.
    starts = [0, 1024, 2048, 3072]
    bed_path = root / "intervals.bed"
    rows: list[tuple[str, int, int]] = []
    with bed_path.open("w") as fh:
        for i in range(4):
            for cname, _ in chroms:
                s = starts[i]
                e = s + INTERVAL_BP
                fh.write(f"{cname}\t{s}\t{e}\tfold0\n")
                rows.append((cname, s, e))

    return {
        "root": root,
        "fasta": str(fasta_path),
        "bw_paths": bw_paths,
        "bed_path": str(bed_path),
        "bed_rows": rows,
    }


def _load_arrow_files(split_dir: Path) -> dict[str, list[np.ndarray]]:
    """Read every Arrow shard in a split dir and return columns as Python lists."""
    cols: dict[str, list[np.ndarray]] = {
        "labels": [],
        "input_ids": [],
        "index": [],
        "local_index": [],
        "interval": [],
    }
    for shard in sorted(split_dir.glob("data-*-of-*.arrow")):
        with shard.open("rb") as fh:
            reader = ipc.open_stream(fh)
            for batch in reader:
                tbl = batch.to_pydict()
                cols["labels"].extend(np.asarray(r, dtype=np.float32) for r in tbl["labels"])
                cols["input_ids"].extend(np.asarray(r, dtype=np.int8) for r in tbl["input_ids"])
                cols["index"].extend(tbl["index"])
                cols["local_index"].extend(tbl["local_index"])
                cols["interval"].extend(tbl["interval"])
    return cols


def test_chrom_pass_matches_direct_bigwig(synth_dataset):
    """chrom_pass output must be numerically equal to direct-bigwig output."""
    from regulonado._rs import (  # type: ignore[import-not-found]
        write_arrow_split_chrom_pass,
        write_arrow_split_from_bigwigs,
    )

    root = synth_dataset["root"]
    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [
        (c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]
    ]
    sample_indices = list(range(len(bed_rows_tuples)))
    # signal_intervals == bed intervals (no shift augmentation)
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]
    minus_flags = [False] * N_TRACKS

    # --- direct-bigwig (single shard) ---
    fast_dir = root / "fast"
    fast_dir.mkdir()
    write_arrow_split_from_bigwigs(
        bw_paths,
        minus_flags,
        signal_intervals,
        str(fast_dir / "data-00000-of-00001.arrow"),
        sample_indices,
        bed_rows_tuples,
        synth_dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        batch_size=4,
        n_threads=2,
        compression="none",
    )

    # --- chrom_pass (one shard per chromosome) ---
    cp_dir = root / "chrom_pass"
    cp_dir.mkdir()
    write_arrow_split_chrom_pass(
        bw_paths,
        minus_flags,
        signal_intervals,
        str(cp_dir),
        sample_indices,
        bed_rows_tuples,
        synth_dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        BIN_SIZE,
        batch_size=4,
        n_threads=2,
        compression="none",
    )

    fast = _load_arrow_files(fast_dir)
    cp = _load_arrow_files(cp_dir)

    # Same row count.
    assert len(fast["labels"]) == len(cp["labels"]) == len(bed_rows_tuples)

    # local_index column exists and is monotonic within each output.
    assert sorted(fast["local_index"]) == list(range(len(bed_rows_tuples)))
    assert sorted(cp["local_index"]) == list(range(len(bed_rows_tuples)))

    # index column == global BED row index (preserved across both strategies).
    assert sorted(fast["index"]) == sample_indices
    assert sorted(cp["index"]) == sample_indices

    # Numerical parity: pair rows by global `index` and compare.
    fast_by_idx = {int(idx): (lab, inp) for idx, lab, inp in
                   zip(fast["index"], fast["labels"], fast["input_ids"])}
    cp_by_idx = {int(idx): (lab, inp) for idx, lab, inp in
                 zip(cp["index"], cp["labels"], cp["input_ids"])}

    for gid in sample_indices:
        fl, fi = fast_by_idx[gid]
        cl, ci = cp_by_idx[gid]
        np.testing.assert_array_equal(
            fl, cl, err_msg=f"labels mismatch at global index {gid}"
        )
        np.testing.assert_array_equal(
            fi, ci, err_msg=f"input_ids mismatch at global index {gid}"
        )


def test_chrom_pass_loads_via_datasets(synth_dataset, tmp_path):
    """The chrom_pass output dir should load via datasets.load_from_disk."""
    from datasets import Array2D, Features, Value, load_from_disk

    from regulonado._rs import write_arrow_split_chrom_pass  # type: ignore[import-not-found]
    from regulonado.dataset import _write_hf_split_metadata

    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [
        (c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]
    ]
    sample_indices = list(range(len(bed_rows_tuples)))
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    write_arrow_split_chrom_pass(
        bw_paths,
        [False] * N_TRACKS,
        signal_intervals,
        str(split_dir),
        sample_indices,
        bed_rows_tuples,
        synth_dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        BIN_SIZE,
        batch_size=4,
        n_threads=2,
        compression="zstd",
    )

    features = Features({
        "input_ids": Array2D(dtype="int8", shape=(4, CONTEXT_LEN)),
        "labels": Array2D(dtype="float32", shape=(N_TRACKS, N_PRED_BINS)),
        "interval": Value(dtype="string"),
        "index": Value(dtype="int64"),
        "local_index": Value(dtype="int64"),
    })
    shards = sorted(p.name for p in split_dir.glob("data-*-of-*.arrow"))
    _write_hf_split_metadata(split_dir, features=features, arrow_filenames=shards)

    ds = load_from_disk(str(split_dir))
    assert len(ds) == len(bed_rows_tuples)
    assert set(ds.column_names) == {"input_ids", "labels", "interval", "index", "local_index"}

    # The schema declares labels as (N_TRACKS, N_PRED_BINS) — verify a row.
    row = ds[0]
    arr = np.asarray(row["labels"])
    assert arr.shape == (N_TRACKS, N_PRED_BINS)
