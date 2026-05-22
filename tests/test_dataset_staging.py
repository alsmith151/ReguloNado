from __future__ import annotations

from pathlib import Path
import shutil


def test_stage_files_avoids_basename_collisions(tmp_path):
    from regulonado.dataset import _stage_files

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
    from regulonado.dataset import _stage_files

    src = tmp_path / "track.bw"
    src.write_text("track")

    staged = _stage_files([src, src, src], tmp_path / "stage", max_workers=2)

    assert staged[0] == staged[1] == staged[2]
    assert Path(staged[0]).read_text() == "track"


def test_track_dedupe_none_preserves_repeated_tracks(tmp_path):
    from regulonado.dataset import _resolve_bigwig_tracks

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
    from regulonado.dataset import _resolve_bigwig_tracks

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
    assert {
        r["duplicate_of_track_index"] for r in metadata["dropped_duplicate_tracks"]
    } == {0}


def test_track_dedupe_content_drops_copied_identical_file(tmp_path):
    from regulonado.dataset import _resolve_bigwig_tracks

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
    from regulonado.dataset import _resolve_bigwig_tracks

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
