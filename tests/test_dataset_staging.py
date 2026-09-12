from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner


def test_stage_files_avoids_basename_collisions(tmp_path):
    from regulonado.dataset.build import _stage_files

    src_a = tmp_path / "a" / "track.bw"
    src_b = tmp_path / "b" / "track.bw"
    src_a.parent.mkdir()
    src_b.parent.mkdir()
    src_a.write_text("a")
    src_b.write_text("b")

    staged = _stage_files([src_a, src_b], tmp_path / "stage", max_workers=2)

    assert len(staged) == 2
    assert staged[0] != staged[1]
    assert Path(staged[0]).read_text() == "a"
    assert Path(staged[1]).read_text() == "b"


def test_stage_files_copies_exact_duplicate_once_preserves_order(tmp_path):
    from regulonado.dataset.build import _stage_files

    src = tmp_path / "track.bw"
    src.write_text("track")

    staged = _stage_files([src, src, src], tmp_path / "stage", max_workers=2)

    assert staged[0] == staged[1] == staged[2]
    assert Path(staged[0]).read_text() == "track"


def test_concurrent_stage_files_are_isolated(tmp_path):
    from regulonado.dataset.build import _stage_files

    source_a = tmp_path / "source" / "a.bw"
    source_b = tmp_path / "source" / "b.bw"
    source_a.parent.mkdir()
    source_a.write_bytes(b"a" * 4096)
    source_b.write_bytes(b"b" * 4096)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_stage_files, [source_a, source_b], tmp_path / f"scratch-{i}")
            for i in range(2)
        ]
        staged = [future.result() for future in futures]

    assert Path(staged[0][0]).read_bytes() == source_a.read_bytes()
    assert Path(staged[0][1]).read_bytes() == source_b.read_bytes()
    assert Path(staged[1][0]).read_bytes() == source_a.read_bytes()
    assert Path(staged[1][1]).read_bytes() == source_b.read_bytes()
    assert set(staged[0]).isdisjoint(staged[1])


def test_track_dedupe_none_preserves_repeated_tracks(tmp_path):
    from regulonado.dataset.discovery import _resolve_bigwig_tracks

    src = tmp_path / "track.bw"
    src.write_text("track")

    paths, metadata = _resolve_bigwig_tracks(
        [src, src],
        drop_missing=False,
        dedupe_tracks="none",
    )

    assert paths == [str(src), str(src)]
    assert metadata["final_bigwig_paths"] == paths
    assert metadata["bigwig_paths"] == paths
    assert metadata["n_final_tracks"] == 2
    assert metadata["n_dropped_duplicate_tracks"] == 0


def test_track_dedupe_identity_drops_exact_repeated_path(tmp_path):
    from regulonado.dataset.discovery import _resolve_bigwig_tracks

    src = tmp_path / "track.bw"
    src.write_text("track")

    paths, metadata = _resolve_bigwig_tracks(
        [src, src, src],
        drop_missing=False,
        dedupe_tracks="identity",
    )

    assert paths == [str(src)]
    assert metadata["final_bigwig_paths"] == paths
    assert metadata["bigwig_paths"] == paths
    assert metadata["n_requested_tracks"] == 3
    assert metadata["n_final_tracks"] == 1
    assert metadata["n_dropped_duplicate_tracks"] == 2
    assert [r["track_index"] for r in metadata["final_track_records"]] == [0]
    assert {r["duplicate_of_track_index"] for r in metadata["dropped_duplicate_tracks"]} == {0}


def test_track_dedupe_content_drops_copied_identical_file(tmp_path):
    from regulonado.dataset.discovery import _resolve_bigwig_tracks

    src = tmp_path / "track-a.bw"
    copied = tmp_path / "track-b.bw"
    src.write_bytes(b"same bigwig bytes")
    shutil.copyfile(src, copied)

    paths, metadata = _resolve_bigwig_tracks(
        [src, copied],
        drop_missing=False,
        dedupe_tracks="content",
    )

    assert paths == [str(src)]
    dropped = metadata["dropped_duplicate_tracks"]
    assert metadata["n_final_tracks"] == 1
    assert metadata["n_dropped_duplicate_tracks"] == 1
    assert dropped[0]["source_index"] == 1
    assert dropped[0]["duplicate_of_source_index"] == 0
    assert dropped[0]["duplicate_of_track_index"] == 0
    assert dropped[0]["dedupe_method"] == "content"
    assert "content_hash" in dropped[0]
    assert metadata["dedupe_tracks"]["n_hashed_files"] == 2


def test_track_dedupe_content_keeps_same_basename_different_content(tmp_path):
    from regulonado.dataset.discovery import _resolve_bigwig_tracks

    src_a = tmp_path / "a" / "track.bw"
    src_b = tmp_path / "b" / "track.bw"
    src_a.parent.mkdir()
    src_b.parent.mkdir()
    src_a.write_bytes(b"aa")
    src_b.write_bytes(b"bb")

    paths, metadata = _resolve_bigwig_tracks(
        [src_a, src_b],
        drop_missing=False,
        dedupe_tracks="content",
    )

    assert paths == [str(src_a), str(src_b)]
    assert metadata["final_bigwig_paths"] == paths
    assert metadata["bigwig_paths"] == paths
    assert metadata["n_final_tracks"] == 2
    assert metadata["n_dropped_duplicate_tracks"] == 0
    assert [r["track_index"] for r in metadata["final_track_records"]] == [0, 1]


def test_publish_tree_same_filesystem_swaps_and_deletes_stale_entries(tmp_path):
    """os.replace path: dest is fully replaced by src, matching rsync --delete."""
    from regulonado.dataset.build import _publish_tree

    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    (src / "keep.txt").write_text("new")
    dest.mkdir()
    (dest / "stale.txt").write_text("old")

    _publish_tree(src, dest)

    assert (dest / "keep.txt").read_text() == "new"
    assert not (dest / "stale.txt").exists()
    assert not src.exists()  # swapped in place, not copied
    assert not (tmp_path / "dest.old").exists()
    assert not (tmp_path / "dest.staging").exists()


def test_publish_tree_same_filesystem_first_publish(tmp_path):
    """os.replace path when dest does not exist yet (first build)."""
    from regulonado.dataset.build import _publish_tree

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")
    dest = tmp_path / "does" / "not" / "exist" / "dest"

    _publish_tree(src, dest)

    assert (dest / "a.txt").read_text() == "a"


def test_publish_tree_cross_filesystem_copies_then_swaps(tmp_path, monkeypatch):
    """Cross-fs path: copy into a temp sibling of dest, then swap — same end result."""
    from regulonado.dataset import build as build_module

    monkeypatch.setattr(build_module, "_same_filesystem", lambda a, b: False)

    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    (src / "keep.txt").write_text("new")
    dest.mkdir()
    (dest / "stale.txt").write_text("old")

    build_module._publish_tree(src, dest)

    assert (dest / "keep.txt").read_text() == "new"
    assert not (dest / "stale.txt").exists()
    assert src.exists()  # cross-fs: src is copied, not moved
    assert not (tmp_path / "dest.staging").exists()
    assert not (tmp_path / "dest.old").exists()


def test_same_filesystem_true_within_one_tmp_dir(tmp_path):
    from regulonado.dataset.build import _same_filesystem

    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b" / "not-created-yet"

    assert _same_filesystem(a, b) is True


def test_build_cli_skips_final_reload_for_fast_path(tmp_path, monkeypatch):
    from regulonado.__main__ import app

    runner = CliRunner()
    captured: dict[str, object] = {}

    bed = tmp_path / "intervals.bed"
    fasta = tmp_path / "genome.fa"
    track_table = tmp_path / "tracks.parquet"
    output_dir = tmp_path / "out"
    bed.write_text("")
    fasta.write_text("")
    track_table.write_text("")

    def fake_build_dataset(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("regulonado.dataset.build_dataset", fake_build_dataset)

    result = runner.invoke(
        app,
        [
            "build",
            str(bed),
            str(fasta),
            str(output_dir),
            "--track-table",
            str(track_table),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["return_dataset"] is False


@pytest.mark.parametrize("old_name", ["chrom_pass", "fast"])
def test_old_strategy_names_are_rejected(tmp_path, old_name):
    """The retired strategy names raise instead of silently mapping to the new ones."""
    from regulonado.dataset.build import build_dataset

    with pytest.raises(ValueError, match="strategy must be 'in_memory' or 'streaming'"):
        build_dataset(
            tmp_path / "intervals.bed",
            tmp_path / "genome.fa",
            tmp_path / "tracks.parquet",
            tmp_path / "out",
            strategy=old_name,
        )


def test_build_dataset_rejects_bgzipped_fasta(tmp_path):
    from regulonado.dataset.build import build_dataset

    fasta = tmp_path / "genome.fa.gz"
    fasta.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00")  # gzip/bgzf magic

    with pytest.raises(ValueError, match="bgzf-compressed FASTA is not supported"):
        build_dataset(
            tmp_path / "intervals.bed",
            fasta,
            tmp_path / "tracks.parquet",
            tmp_path / "out",
        )


# --------------------------------------------------------------------------- #
# Pure planning-function unit tests (F15 decomposition)                       #
# --------------------------------------------------------------------------- #


def test_compute_geometry_pads_for_shift():
    from regulonado.dataset.build import _compute_geometry

    geo = _compute_geometry(context_length=100, bin_size=4, n_pred_bins=10, shift_max_bp=8)

    assert geo.stored_context == 116
    assert geo.shift_bins == 2
    assert geo.stored_n_bins == 14


def test_compute_geometry_no_shift_is_a_no_op():
    from regulonado.dataset.build import _compute_geometry

    geo = _compute_geometry(context_length=64, bin_size=4, n_pred_bins=16, shift_max_bp=0)

    assert geo.stored_context == 64
    assert geo.shift_bins == 0
    assert geo.stored_n_bins == 16


def test_edge_unsafe_row_indices_flags_rows_near_contig_start():
    from regulonado.dataset.build import _edge_unsafe_row_indices

    bed_rows = [
        ("chrA", 0, 2, "fold0"),  # center=1, half=8 -> -7 < 0 -> unsafe
        ("chrA", 100, 116, "fold0"),  # center=108, half=8 -> 100 -> safe
        ("chrA", 4, 20, "fold0"),  # center=12, half=8 -> 4 -> safe (not negative)
        ("chrA", 3, 4, "fold0"),  # center=3, half=8 -> -5 < 0 -> unsafe
    ]

    unsafe = _edge_unsafe_row_indices(bed_rows, n_pred_bins=4, bin_size=4, shift_max_bp=0)

    assert unsafe == [0, 3]


def test_edge_unsafe_row_indices_accounts_for_shift_max_bp():
    from regulonado.dataset.build import _edge_unsafe_row_indices

    bed_rows = [("chrA", 100, 116, "fold0")]  # center=108, half=8 without shift -> safe

    # With shift_max_bp=104 the half-window grows to 112, making the same row unsafe.
    unsafe = _edge_unsafe_row_indices(bed_rows, n_pred_bins=4, bin_size=4, shift_max_bp=104)

    assert unsafe == [0]


def test_compute_split_indices_excludes_edge_unsafe_rows(tmp_path):
    from regulonado.dataset.build import _compute_split_indices

    bed_rows = [
        ("chrA", 0, 2, "fold0"),
        ("chrA", 100, 116, "fold0"),
        ("chrA", 200, 216, "fold1"),
    ]
    splits = {"train": ["fold0"], "validation": ["fold1"]}

    split_indices, splits_to_build = _compute_split_indices(
        bed_rows, splits, None, tmp_path / "out", False, {0}
    )

    assert split_indices == {"train": [1], "validation": [2]}
    assert splits_to_build == ["train", "validation"]


def test_compute_split_indices_skips_already_built_splits(tmp_path):
    from regulonado.dataset.build import _compute_split_indices

    output_dir = tmp_path / "out"
    (output_dir / "train").mkdir(parents=True)
    (output_dir / "train" / "dataset_info.json").write_text("{}")
    bed_rows = [("chrA", 100, 116, "fold0"), ("chrA", 200, 216, "fold1")]
    splits = {"train": ["fold0"], "validation": ["fold1"]}

    split_indices, splits_to_build = _compute_split_indices(
        bed_rows, splits, None, output_dir, False, set()
    )

    assert splits_to_build == ["validation"]
    assert "train" not in split_indices


# --------------------------------------------------------------------------- #
# New-bug regression: edge-clamp misalignment (item 4)                        #
# --------------------------------------------------------------------------- #

_EDGE_BIN_SIZE = 4
_EDGE_N_PRED_BINS = 4  # pred_bp = 16, half-window = 8


def _build_edge_case_dataset(root):
    """One BED row unsafe (center 1, half-window 8) and one safe (center 108)."""
    import pandas as pd
    import pybigtools
    import pyfaidx

    root.mkdir(parents=True, exist_ok=True)
    chrom_len = 256
    fasta_path = root / "ref.fa"
    with fasta_path.open("w") as fh:
        fh.write(">chrA\n")
        seq = "ACGT" * (chrom_len // 4)
        for i in range(0, chrom_len, 80):
            fh.write(seq[i : i + 80] + "\n")
    pyfaidx.Fasta(str(fasta_path))

    bw_path = root / "track_0.bw"
    w = pybigtools.open(str(bw_path), "w")
    w.write({"chrA": chrom_len}, iter([("chrA", 0, chrom_len, 5.0)]))

    bed_path = root / "intervals.bed"
    with bed_path.open("w") as fh:
        fh.write("chrA\t0\t2\tfold0\n")  # unsafe: center=1
        fh.write("chrA\t100\t116\tfold0\n")  # safe: center=108

    tracks_parquet = root / "tracks.parquet"
    from regulonado.tracks_table import write_track_table

    write_track_table(
        pd.DataFrame(
            {
                "track_name": ["track_0"],
                "status": ["included"],
                "track_index": [0],
                "path": [str(bw_path)],
                "resolved_path": [str(bw_path)],
            }
        ),
        tracks_parquet,
    )

    return {
        "fasta": str(fasta_path),
        "bed_path": str(bed_path),
        "tracks_parquet": str(tracks_parquet),
    }


def test_build_dataset_drops_edge_unsafe_row_and_counts_it(tmp_path):
    """The unsafe row is dropped+counted at defaults; the build succeeds; other rows unaffected."""
    import json

    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    from regulonado.dataset.build import build_dataset

    inputs = _build_edge_case_dataset(tmp_path / "in")
    output_dir = tmp_path / "out"

    build_dataset(
        inputs["bed_path"],
        inputs["fasta"],
        inputs["tracks_parquet"],
        output_dir,
        splits={"train": ["fold0"]},
        context_length=16,
        bin_size=_EDGE_BIN_SIZE,
        n_pred_bins=_EDGE_N_PRED_BINS,
        shift_max_bp=0,
        n_extract_threads=1,
        arrow_batch_size=2,
        arrow_write_threads=1,
        strategy="in_memory",
        return_dataset=False,
    )

    shard = sorted((output_dir / "train").glob("data-*-of-*.arrow"))
    assert shard, "expected at least one Arrow shard"
    with shard[0].open("rb") as fh:
        reader = ipc.open_stream(fh)
        rows = [batch.to_pydict() for batch in reader]
    indices = [i for batch in rows for i in batch["index"]]
    intervals = [iv for batch in rows for iv in batch["interval"]]
    assert indices == [1]
    assert intervals == ["chrA:100-116"]

    table = pq.read_table(output_dir / "tracks.parquet")
    attrs = json.loads(table.schema.metadata[b"regulonado"])
    assert attrs["edge_dropped_rows"] == 1
    assert attrs["edge_dropped_examples"] == ["chrA:0-2"]
    assert "skipped_rows" not in attrs


def test_build_dataset_streaming_strategy_also_drops_edge_unsafe_row(tmp_path):
    """The edge-safety filter runs before strategy dispatch, so streaming is covered too."""
    import pyarrow.ipc as ipc
    from regulonado.dataset.build import build_dataset

    inputs = _build_edge_case_dataset(tmp_path / "in")
    output_dir = tmp_path / "out"

    build_dataset(
        inputs["bed_path"],
        inputs["fasta"],
        inputs["tracks_parquet"],
        output_dir,
        splits={"train": ["fold0"]},
        context_length=16,
        bin_size=_EDGE_BIN_SIZE,
        n_pred_bins=_EDGE_N_PRED_BINS,
        shift_max_bp=0,
        n_extract_threads=1,
        arrow_batch_size=2,
        strategy="streaming",
        return_dataset=False,
    )

    shard = output_dir / "train" / "data-00000-of-00001.arrow"
    assert shard.exists()
    with shard.open("rb") as fh:
        reader = ipc.open_stream(fh)
        rows = [batch.to_pydict() for batch in reader]
    indices = [i for batch in rows for i in batch["index"]]
    assert indices == [1]
