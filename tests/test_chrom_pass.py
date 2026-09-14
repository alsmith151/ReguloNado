"""Integration test for the chrom_pass Parquet writer.

Builds a tiny synthetic dataset (2 chromosomes × 3 BigWigs × 8 intervals) and
verifies that the chrom_pass writer produces bit-identical output to the
existing per-sample direct-bigwig writer, and that a built dataset is loadable
via ``datasets.load_dataset`` per the HF-Hub-layout contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pybigtools
import pytest


def _features_json(n_tracks: int, n_bins: int, context_len: int) -> str:
    from datasets import Features, List, Value

    features = Features(
        {
            "sequence_tokens": List(Value("uint8"), length=context_len),
            "signal": List(List(Value("float32"), length=n_bins), length=n_tracks),
            "interval": Value(dtype="string"),
            "index": Value(dtype="int64"),
            "local_index": Value(dtype="int64"),
        }
    )
    return json.dumps(features.to_dict())


def write_parquet_split_chrom_pass(
    bw_paths,
    minus_flags,
    signal_intervals,
    out_dir,
    sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    bin_size,
    **kwargs,
):
    """Write a single split via the multi-split chrom_pass writer.

    The extension only exports a multi-split entry point (all splits share one
    scan and one output directory) — there is no production caller for a
    single-split variant, only these tests. Driving the real writer with a
    one-element split list keeps these tests exercising the production path.
    The split name is a filename prefix (``{split}-NNNNN-of-MMMMM.parquet``),
    so callers select this split's files with a ``split-*.parquet`` glob.
    """
    from regulonado._rs import write_parquet_splits_chrom_pass  # type: ignore[import-not-found]

    hf_features_json = kwargs.pop(
        "hf_features_json", _features_json(len(bw_paths), n_bins, context_len)
    )
    return write_parquet_splits_chrom_pass(
        bw_paths,
        minus_flags,
        signal_intervals,
        ["split"],
        str(out_dir),
        [sample_indices],
        bed_rows,
        fasta_path,
        n_bins,
        context_len,
        bin_size,
        hf_features_json,
        **kwargs,
    )


# Tiny but valid: bin_size 8, two chromosomes 4 kb each, 8 intervals of 512 bp.
N_TRACKS = 3
CHROM_LEN = 4096
BIN_SIZE = 8
N_PRED_BINS = 64  # interval bp = N_PRED_BINS * BIN_SIZE = 512
INTERVAL_BP = N_PRED_BINS * BIN_SIZE
CONTEXT_LEN = INTERVAL_BP  # no shift augmentation


def _build_synth_dataset(tmp_path_factory, *, segmented: bool):
    root = tmp_path_factory.mktemp("synth_segmented" if segmented else "synth")
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
    rng = np.random.default_rng(0 if not segmented else 7)
    for t in range(N_TRACKS):
        path = root / f"track_{t}.bw"
        chromsize_map = {name: n for name, n in chroms}
        entries: list[tuple[str, int, int, float]] = []
        for chrom_idx, (name, n) in enumerate(chroms):
            if segmented:
                pos = 0
                step = 0
                while pos < n:
                    gap = [0, 3, 7, 11][(step + chrom_idx + t) % 4]
                    pos += gap
                    if pos >= n:
                        break
                    seg_len = [5, 13, 29, 61, 97][(step * 2 + chrom_idx + t) % 5]
                    end = min(pos + seg_len, n)
                    value = float(
                        np.sin((pos + 19 * t + 23 * chrom_idx) / 91.0) * 4.5
                        + 8.0
                        + t * 1.75
                        + rng.normal(0, 0.05)
                    )
                    entries.append((name, int(pos), int(end), value))
                    pos = end
                    step += 1
            else:
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


@pytest.fixture(scope="module")
def synth_dataset(tmp_path_factory):
    return _build_synth_dataset(tmp_path_factory, segmented=False)


@pytest.fixture(scope="module")
def segmented_synth_dataset(tmp_path_factory):
    return _build_synth_dataset(tmp_path_factory, segmented=True)


def _load_parquet_files(data_dir: Path, split: str = "*") -> dict[str, list]:
    """Read every Parquet shard matching ``{split}-*.parquet`` and return columns."""
    shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    assert shards, f"no Parquet shards found in {data_dir}"
    table = pq.read_table(shards)
    return {
        "signal": [np.asarray(r, dtype=np.float32) for r in table["signal"].to_pylist()],
        "sequence_tokens": [
            np.asarray(r, dtype=np.uint8) for r in table["sequence_tokens"].to_pylist()
        ],
        "index": table["index"].to_pylist(),
        "local_index": table["local_index"].to_pylist(),
        "interval": table["interval"].to_pylist(),
    }


def _assert_chrom_pass_matches_direct_bigwig(dataset):
    from regulonado._rs import (  # type: ignore[import-not-found]
        write_parquet_split_from_bigwigs,
    )

    root = dataset["root"]
    bw_paths = dataset["bw_paths"]
    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in dataset["bed_rows"]]
    sample_indices = list(range(len(bed_rows_tuples)))
    # signal_intervals == bed intervals (no shift augmentation)
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in dataset["bed_rows"]]
    minus_flags = [False] * N_TRACKS
    hf_features_json = _features_json(N_TRACKS, N_PRED_BINS, CONTEXT_LEN)

    # --- direct-bigwig (single shard) ---
    fast_dir = root / "fast"
    fast_dir.mkdir()
    write_parquet_split_from_bigwigs(
        bw_paths,
        minus_flags,
        signal_intervals,
        str(fast_dir / "data-00000-of-00001.parquet"),
        sample_indices,
        bed_rows_tuples,
        dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        hf_features_json,
        zstd_level=1,
        n_threads=2,
    )

    # --- chrom_pass (one shard per chromosome) ---
    cp_dir = root / "chrom_pass"
    cp_dir.mkdir()
    write_parquet_split_chrom_pass(
        bw_paths,
        minus_flags,
        signal_intervals,
        str(cp_dir),
        sample_indices,
        bed_rows_tuples,
        dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        BIN_SIZE,
        hf_features_json=hf_features_json,
        shard_size=2,
        n_threads=2,
        write_threads=2,
        zstd_level=1,
    )

    fast = _load_parquet_files(fast_dir)
    cp = _load_parquet_files(cp_dir, split="split")
    assert len(sorted(cp_dir.glob("split-*.parquet"))) >= 4

    # Same row count.
    assert len(fast["signal"]) == len(cp["signal"]) == len(bed_rows_tuples)

    # local_index column exists and is monotonic within each output.
    assert sorted(fast["local_index"]) == list(range(len(bed_rows_tuples)))
    assert sorted(cp["local_index"]) == list(range(len(bed_rows_tuples)))

    # index column == global BED row index (preserved across both strategies).
    assert sorted(fast["index"]) == sample_indices
    assert sorted(cp["index"]) == sample_indices

    # Numerical parity: pair rows by global `index` and compare.
    fast_by_idx = {
        int(idx): (sig, seq)
        for idx, sig, seq in zip(fast["index"], fast["signal"], fast["sequence_tokens"])
    }
    cp_by_idx = {
        int(idx): (sig, seq)
        for idx, sig, seq in zip(cp["index"], cp["signal"], cp["sequence_tokens"])
    }

    for gid in sample_indices:
        fs, fi = fast_by_idx[gid]
        cs, ci = cp_by_idx[gid]
        np.testing.assert_array_equal(fs, cs, err_msg=f"signal mismatch at global index {gid}")
        np.testing.assert_array_equal(
            fi, ci, err_msg=f"sequence_tokens mismatch at global index {gid}"
        )


def test_chrom_pass_matches_direct_bigwig(synth_dataset):
    """chrom_pass output must be numerically equal to direct-bigwig output."""
    _assert_chrom_pass_matches_direct_bigwig(synth_dataset)


def test_chrom_pass_matches_direct_bigwig_segmented_intervals(segmented_synth_dataset):
    """Segmented BigWig intervals must match the direct per-sample writer exactly."""
    _assert_chrom_pass_matches_direct_bigwig(segmented_synth_dataset)


def _build_tiny_hf_dataset(synth_dataset, output_dir: Path) -> None:
    """Build a tiny full dataset (train + validation) via build_dataset."""
    import pandas as pd  # noqa: PLC0415
    from regulonado.dataset import build_dataset  # noqa: PLC0415
    from regulonado.tracks_table import write_track_table  # noqa: PLC0415

    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]]
    bed_path = synth_dataset["root"] / "intervals_hf.bed"
    with bed_path.open("w") as fh:
        for i, (chrom, start, end, _name) in enumerate(bed_rows_tuples):
            fold = "fold0" if i % 2 == 0 else "fold1"
            fh.write(f"{chrom}\t{start}\t{end}\t{fold}\n")

    track_names = [f"track_{t}" for t in range(N_TRACKS)]
    tracks_parquet = synth_dataset["root"] / "tracks_hf.parquet"
    write_track_table(
        pd.DataFrame(
            {
                "track_name": track_names,
                "status": ["included"] * N_TRACKS,
                "track_index": list(range(N_TRACKS)),
                "path": synth_dataset["bw_paths"],
                "resolved_path": synth_dataset["bw_paths"],
            }
        ),
        tracks_parquet,
    )

    build_dataset(
        bed_path,
        synth_dataset["fasta"],
        tracks_parquet,
        output_dir,
        splits={"train": ["fold0"], "validation": ["fold1"]},
        context_length=CONTEXT_LEN,
        bin_size=BIN_SIZE,
        n_pred_bins=N_PRED_BINS,
        shift_max_bp=0,
        n_extract_threads=2,
        write_threads=2,
        zstd_level=3,
        rows_per_row_group=1,
        strategy="in_memory",
        stage_to_scratch=False,
    )


def test_chrom_pass_loads_via_datasets(synth_dataset, tmp_path):
    """A built dataset directory satisfies the HF-Hub-layout `load_dataset` contract.

    Checks: splits {train, validation}, info.splits counts match, features equal, the first
    example's shapes are right, and tracks.parquet is not a split.
    """
    from datasets import Features, List, Value, load_dataset

    output_dir = tmp_path / "out"
    _build_tiny_hf_dataset(synth_dataset, output_dir)

    expected_features = Features(
        {
            "sequence_tokens": List(Value("uint8"), length=CONTEXT_LEN),
            "signal": List(List(Value("float32"), length=N_PRED_BINS), length=N_TRACKS),
            "interval": Value(dtype="string"),
            "index": Value(dtype="int64"),
            "local_index": Value(dtype="int64"),
        }
    )

    ds = load_dataset(str(output_dir), streaming=True)
    assert set(ds.keys()) == {"train", "validation"}

    train_info_splits = ds["train"].info.splits
    assert train_info_splits is not None
    assert {split: info.num_examples for split, info in train_info_splits.items()} == {
        "train": 4,
        "validation": 4,
    }

    for split in ("train", "validation"):
        assert ds[split].features == expected_features
        example = next(iter(ds[split]))
        assert np.asarray(example["sequence_tokens"]).shape == (CONTEXT_LEN,)
        assert np.asarray(example["signal"]).shape == (N_TRACKS, N_PRED_BINS)

    assert "tracks_hf" not in ds
    assert not any(p.name == "tracks_hf.parquet" for p in (output_dir / "data").glob("*"))


def test_chrom_pass_write_threads_are_equivalent(synth_dataset, tmp_path):
    """Parallel Parquet writers should not change rows or loadability."""
    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]]
    sample_indices = list(range(len(bed_rows_tuples)))
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]
    hf_features_json = _features_json(N_TRACKS, N_PRED_BINS, CONTEXT_LEN)

    loaded = {}
    for write_threads in (1, 2):
        split_dir = tmp_path / f"split_threads_{write_threads}"
        split_dir.mkdir()
        write_parquet_split_chrom_pass(
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
            hf_features_json=hf_features_json,
            shard_size=2,
            n_threads=2,
            write_threads=write_threads,
            zstd_level=1,
        )
        shards = sorted(split_dir.glob("split-*.parquet"))
        assert len(shards) >= 4
        loaded[write_threads] = _load_parquet_files(split_dir, split="split")

    assert len(loaded[1]["index"]) == len(loaded[2]["index"]) == len(bed_rows_tuples)
    assert sorted(loaded[1]["index"]) == sorted(loaded[2]["index"]) == sample_indices
    assert sorted(loaded[1]["local_index"]) == sorted(loaded[2]["local_index"]) == sample_indices

    by_idx_1 = dict(zip(loaded[1]["index"], zip(loaded[1]["signal"], loaded[1]["sequence_tokens"])))
    by_idx_2 = dict(zip(loaded[2]["index"], zip(loaded[2]["signal"], loaded[2]["sequence_tokens"])))
    for gid in sample_indices:
        np.testing.assert_array_equal(by_idx_1[gid][0], by_idx_2[gid][0])
        np.testing.assert_array_equal(by_idx_1[gid][1], by_idx_2[gid][1])


def test_chrom_pass_shard_size_controls_file_count_and_row_groups(synth_dataset, tmp_path):
    """shard_size controls file count; rows_per_row_group controls num_row_groups.

    The fixture has 4 samples per chromosome (2 chromosomes, 8 rows total).
    shard_size=2 -> 2 shards/chrom (4 files); shard_size=4 -> 1 shard/chrom (2
    files). Row content must be unchanged either way, and each shard's
    ``num_row_groups`` must equal ``rows_in_shard / rows_per_row_group``.
    """
    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]]
    sample_indices = list(range(len(bed_rows_tuples)))
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]
    hf_features_json = _features_json(N_TRACKS, N_PRED_BINS, CONTEXT_LEN)

    cols_by_shard: dict[int, dict] = {}
    n_files: dict[int, int] = {}
    for shard_size in (2, 4):
        split_dir = tmp_path / f"split_shard_{shard_size}"
        split_dir.mkdir()
        write_parquet_split_chrom_pass(
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
            hf_features_json=hf_features_json,
            shard_size=shard_size,
            rows_per_row_group=1,
            n_threads=2,
            zstd_level=1,
        )
        shards = sorted(split_dir.glob("split-*.parquet"))
        n_files[shard_size] = len(shards)
        for shard in shards:
            rows_in_shard = pq.ParquetFile(shard).metadata.num_rows
            num_row_groups = pq.ParquetFile(shard).metadata.num_row_groups
            assert num_row_groups == rows_in_shard  # rows_per_row_group=1
        cols_by_shard[shard_size] = _load_parquet_files(split_dir, split="split")

    # Decoupling: larger shard_size -> strictly fewer shard files.
    assert n_files[2] == 4
    assert n_files[4] == 2

    # ...but identical row content (ordered by global index for a stable compare).
    small = cols_by_shard[2]
    large = cols_by_shard[4]
    assert sorted(small["index"]) == sorted(large["index"]) == sample_indices
    order_s = np.argsort(small["index"])
    order_l = np.argsort(large["index"])
    for si, li in zip(order_s, order_l, strict=True):
        np.testing.assert_array_equal(small["signal"][si], large["signal"][li])
        np.testing.assert_array_equal(small["sequence_tokens"][si], large["sequence_tokens"][li])


def test_chrom_pass_rows_per_row_group_matches_num_row_groups(synth_dataset, tmp_path):
    """num_row_groups in each shard must equal rows_in_shard / rows_per_row_group."""
    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]]
    sample_indices = list(range(len(bed_rows_tuples)))
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]
    hf_features_json = _features_json(N_TRACKS, N_PRED_BINS, CONTEXT_LEN)

    split_dir = tmp_path / "split_rrg"
    split_dir.mkdir()
    write_parquet_split_chrom_pass(
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
        hf_features_json=hf_features_json,
        shard_size=4,
        rows_per_row_group=2,
        n_threads=2,
        zstd_level=1,
    )
    shards = sorted(split_dir.glob("split-*.parquet"))
    assert shards
    for shard in shards:
        meta = pq.ParquetFile(shard).metadata
        assert meta.num_row_groups == meta.num_rows // 2


def test_chrom_pass_shared_scan_writes_multiple_splits(synth_dataset, tmp_path):
    """The shared chrom pass should write loadable split shards in one call."""
    from regulonado._rs import write_parquet_splits_chrom_pass  # type: ignore[import-not-found]

    bw_paths = synth_dataset["bw_paths"]
    bed_rows_tuples = [(c, int(s), int(e), "fold0") for (c, s, e) in synth_dataset["bed_rows"]]
    signal_intervals = [(c, int(s), int(e)) for (c, s, e) in synth_dataset["bed_rows"]]
    hf_features_json = _features_json(N_TRACKS, N_PRED_BINS, CONTEXT_LEN)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_indices = [0, 2, 4, 6]
    validation_indices = [1, 3, 5, 7]

    row_counts = write_parquet_splits_chrom_pass(
        bw_paths,
        [False] * N_TRACKS,
        signal_intervals,
        ["train", "validation"],
        str(data_dir),
        [train_indices, validation_indices],
        bed_rows_tuples,
        synth_dataset["fasta"],
        N_PRED_BINS,
        CONTEXT_LEN,
        BIN_SIZE,
        hf_features_json,
        n_threads=2,
        write_threads=2,
        zstd_level=1,
    )
    assert row_counts == {"train": 4, "validation": 4}

    for split, expected_indices in [("train", train_indices), ("validation", validation_indices)]:
        shards = sorted(data_dir.glob(f"{split}-*.parquet"))
        assert shards
        cols = _load_parquet_files(data_dir, split=split)
        assert len(cols["index"]) == len(expected_indices)
        assert sorted(cols["index"]) == expected_indices
        assert sorted(cols["local_index"]) == list(range(len(expected_indices)))


def test_chrom_pass_rejects_truncated_bigwig(tmp_path):
    from regulonado._rs import write_parquet_splits_chrom_pass  # type: ignore[import-not-found]

    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chrA\n" + "ACGT" * 256 + "\n")
    import pyfaidx

    pyfaidx.Fasta(str(fasta))
    bw = tmp_path / "track.bw"
    writer = pybigtools.open(str(bw), "w")
    writer.write({"chrA": 1024}, iter([("chrA", 0, 10, 5.0)]))
    bw.write_bytes(bw.read_bytes()[:100])
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    hf_features_json = _features_json(1, 64, 512)

    with pytest.raises(RuntimeError, match="failed|error|invalid|unexpected"):
        write_parquet_splits_chrom_pass(
            [str(bw)],
            [False],
            [("chrA", 0, 512)],
            ["split"],
            str(output_dir),
            [[0]],
            [("chrA", 0, 512, "fold0")],
            str(fasta),
            64,
            512,
            8,
            hf_features_json,
            n_threads=1,
        )
