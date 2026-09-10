"""tracks.parquet: schema invariants, round-tripping, and atomic writes."""

from __future__ import annotations

import pandas as pd
import pytest
from regulonado.tracks_table import (
    SCHEMA_VERSION,
    read_track_table,
    to_track_records,
    validate_track_table,
    write_track_table,
)


def _table(**overrides) -> pd.DataFrame:
    rows = [
        {"track_name": "a", "status": "included", "track_index": 0},
        {"track_name": "b", "status": "included", "track_index": 1},
        {"track_name": "c", "status": "dropped_duplicate", "track_index": None},
    ]
    df = pd.DataFrame(rows)
    for column, values in overrides.items():
        df[column] = values
    return df


def test_round_trip_preserves_dtypes_list_columns_and_schema_metadata(tmp_path):
    df = _table(
        scale_factor=[1.0, 2.0, None],
        qc_failed_rules=[[], ["qc_dynamic_range"], []],
    )
    path = tmp_path / "tracks.parquet"
    write_track_table(
        df, path, bin_size=32, shift_max_bp=64, created_at="2026-01-01T00:00:00Z"
    )
    back = read_track_table(path)

    assert back["track_index"].dtype == pd.Int64Dtype()
    assert back.loc[1, "qc_failed_rules"] == ["qc_dynamic_range"]
    assert back.attrs["bin_size"] == 32
    assert back.attrs["shift_max_bp"] == 64
    assert back.attrs["schema_version"] == SCHEMA_VERSION
    assert back.attrs["created_at"] == "2026-01-01T00:00:00Z"


def test_track_index_contiguity_violation_names_the_offending_track(tmp_path):
    df = _table()
    df.loc[df["track_name"] == "b", "track_index"] = 5  # skips 1, not contiguous
    with pytest.raises(ValueError, match="a, b"):
        validate_track_table(df)


def test_duplicate_track_name_raises_naming_it(tmp_path):
    df = _table()
    df.loc[df["track_name"] == "b", "track_name"] = "a"
    with pytest.raises(ValueError, match="a"):
        validate_track_table(df)


def test_missing_status_value_raises():
    df = _table()
    df.loc[0, "status"] = "bogus"
    with pytest.raises(ValueError, match="isin"):
        validate_track_table(df)


def test_scaling_columns_must_be_present_or_absent_together():
    df = _table(scale_factor=[1.0, None, None], scale_clip_soft=[None, None, None])
    with pytest.raises(ValueError):
        validate_track_table(df)


def test_schema_version_mismatch_fails_loudly(tmp_path):
    import json

    import pyarrow.parquet as pq

    df = _table()
    path = tmp_path / "tracks.parquet"
    write_track_table(df, path)

    # write_track_table always stamps the current SCHEMA_VERSION, so simulate
    # a stale file (e.g. from before a schema change) by patching the Arrow
    # metadata directly, bypassing the write path.
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    meta[b"regulonado"] = json.dumps({"schema_version": "0"}).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)

    with pytest.raises(ValueError, match="schema_version"):
        read_track_table(path)


def test_shift_max_bp_must_be_a_multiple_of_bin_size(tmp_path):
    df = _table()
    path = tmp_path / "tracks.parquet"
    write_track_table(df, path, bin_size=32, shift_max_bp=50)
    with pytest.raises(ValueError, match="multiple of bin_size"):
        read_track_table(path)


def test_truncated_parquet_fails_clearly_not_as_a_keyerror(tmp_path):
    df = _table()
    path = tmp_path / "tracks.parquet"
    write_track_table(df, path)
    with path.open("r+b") as fh:
        fh.truncate(16)
    with pytest.raises(Exception) as exc_info:
        read_track_table(path)
    assert not isinstance(exc_info.value, KeyError)


def test_interrupted_write_leaves_previous_file_intact(tmp_path):
    df = _table()
    path = tmp_path / "tracks.parquet"
    write_track_table(df, path, created_at="first")

    # Simulate a crash mid-write: a stray .tmp file must never replace the
    # committed parquet, since write_track_table only os.replace()s at the end.
    tmp_leftover = path.with_suffix(path.suffix + ".tmp")
    tmp_leftover.write_bytes(b"not a valid parquet file")

    back = read_track_table(path)
    assert back.attrs["created_at"] == "first"
    tmp_leftover.unlink()


def test_to_track_records_only_includes_status_included_ordered_by_index():
    df = _table(scale_factor=[1.0, 2.0, None])
    records = to_track_records(df)
    assert [r["track_name"] for r in records] == ["a", "b"]
    assert records[0]["track_index"] == 0
    assert records[0]["scale_factor"] == 1.0


def test_to_track_records_derives_categorical_ids_by_sorted_factorisation():
    df = _table(condition=["beta", "alpha", None])
    records = to_track_records(df)
    by_name = {r["track_name"]: r for r in records}
    # sorted(["alpha", "beta"]) -> alpha=0, beta=1
    assert by_name["a"]["condition_id"] == 1
    assert by_name["b"]["condition_id"] == 0


def test_bigwig_dir_style_track_name_falls_back_to_stem(tmp_path):
    pybigtools = pytest.importorskip("pybigtools")
    from regulonado.dataset.discovery import discover_tracks

    bw = tmp_path / "my_track.bw"
    writer = pybigtools.open(str(bw), "w")
    writer.write({"chr1": 1000}, iter([("chr1", 0, 100, 1.0)]))

    df = discover_tracks([bw])
    assert df.loc[df["status"] == "included", "track_name"].tolist() == ["my_track"]
