from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from regulonado.recompress import recompress_dataset


def _write_shard(path: Path, values: list[int]) -> None:
    table = pa.table({"value": pa.array(values, type=pa.int64())})
    with path.open("wb") as fh:
        with ipc.new_stream(fh, table.schema) as writer:
            for batch in table.to_batches(max_chunksize=1):
                writer.write_batch(batch)


def _make_dataset(path: Path, values: list[int]) -> None:
    split = path / "train"
    split.mkdir(parents=True)
    _write_shard(split / "data.arrow", values)
    (split / "dataset_info.json").write_text("{}")
    (path / "dataset_dict.json").write_text(json.dumps({"splits": ["train"]}))


def test_recompress_rejects_equal_and_nested_paths(tmp_path):
    source = tmp_path / "source"
    _make_dataset(source, [1, 2])
    _make_dataset(source / "nested", [3, 4])

    with pytest.raises(ValueError, match="distinct, non-nested"):
        recompress_dataset(source, source, overwrite=True)
    with pytest.raises(ValueError, match="distinct, non-nested"):
        recompress_dataset(source, source / "nested")
    with pytest.raises(ValueError, match="distinct, non-nested"):
        recompress_dataset(source / "nested", source)


def test_recompress_overwrite_preserves_source(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_dataset(source, [1, 2, 3])
    _make_dataset(destination, [99])

    recompress_dataset(source, destination, overwrite=True)

    assert (source / "train" / "data.arrow").exists()
    with (destination / "train" / "data.arrow").open("rb") as fh:
        table = pa.Table.from_batches(list(ipc.open_stream(fh)))
    assert table.column("value").to_pylist() == [1, 2, 3]


def test_recompress_failure_preserves_existing_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_dataset(source, [1, 2])
    _make_dataset(destination, [99])

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr("regulonado.recompress.recompress_split", fail)

    with pytest.raises(RuntimeError, match="injected failure"):
        recompress_dataset(source, destination, overwrite=True)

    with (destination / "train" / "data.arrow").open("rb") as fh:
        table = pa.Table.from_batches(list(ipc.open_stream(fh)))
    assert table.column("value").to_pylist() == [99]


def test_recompress_mid_stream_failure_cleans_temporary_output(tmp_path, monkeypatch):
    import regulonado.recompress as recompress_module

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_dataset(source, [1, 2])
    split = source / "train"
    _write_shard(split / "data-01.arrow", [3, 4])
    _write_shard(split / "data-02.arrow", [5, 6])
    _make_dataset(destination, [99])

    original = recompress_module.recompress_shard
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("mid-stream failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(recompress_module, "recompress_shard", fail_on_second)
    with pytest.raises(RuntimeError, match="mid-stream failure"):
        recompress_dataset(source, destination, overwrite=True)

    assert not list(tmp_path.glob(f".{destination.name}.*"))
    with (destination / "train" / "data.arrow").open("rb") as fh:
        table = pa.Table.from_batches(list(ipc.open_stream(fh)))
    assert table.column("value").to_pylist() == [99]
