from __future__ import annotations

import bisect
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def stack_batch_tensors(
    batch: Sequence[Mapping[str, object]],
    *,
    tensor_keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    stacked: dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        values = [item[key] for item in batch]
        stacked[key] = torch.stack(
            [
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                for value in values
            ]
        )
    return stacked


def _decode_fixed_size_list_column(table: pa.Table, name: str) -> np.ndarray:
    """Decode a (possibly nested) fixed-size-list column to a ``(rows, *dims)`` array.

    ``FixedSizeListArray.flatten()`` drops one level of nesting per call, so calling it
    once per stored dimension turns ``sequence_tokens`` (one level) or ``signal`` (two
    levels, tracks then bins) into a flat primitive array that reshapes back cleanly —
    no Python-level iteration over rows.
    """
    column = table.column(name).combine_chunks()
    dims: list[int] = []
    array_type = table.schema.field(name).type
    while pa.types.is_fixed_size_list(array_type):
        dims.append(array_type.list_size)
        array_type = array_type.value_type
    flat = column
    for _ in dims:
        flat = flat.flatten()
    # zero_copy_only=False allows a copy when needed but doesn't force one — Arrow's
    # buffer is otherwise read-only, which trips a PyTorch warning at collate time.
    values = flat.to_numpy(zero_copy_only=False).copy()
    return values.reshape(table.num_rows, *dims)


def _row_to_example(table: pa.Table, local_row: int, columns: Sequence[str]) -> dict:
    """Extract one row of ``table`` (as read for a single row group) into a plain dict."""
    example: dict = {}
    for name in columns:
        field_type = table.schema.field(name).type
        if pa.types.is_fixed_size_list(field_type):
            example[name] = _decode_fixed_size_list_column(table, name)[local_row]
        else:
            example[name] = table.column(name)[local_row].as_py()
    return example


class WindowParquetDataset(torch.utils.data.Dataset):
    """Map-style random access over one split's HF-layout Parquet shards.

    Reads ``<data_dir>/data/{split}-*.parquet`` one row group at a time. Shard footers
    (``pq.read_metadata``) are read once, in the main process, to build a cumulative
    row-group index; no Parquet file handle is opened until the first ``__getitem__``
    call, so nothing keeps an open file descriptor across the DataLoader worker fork.
    Each worker lazily opens (and caches) its own ``pq.ParquetFile`` per shard.

    Raises ``FileNotFoundError`` at construction when no shards match the split, and
    ``IndexError`` from ``__getitem__`` for an out-of-range index.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        columns: Sequence[str] = ("sequence_tokens", "signal", "interval", "index", "local_index"),
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self._columns = list(columns)

        shard_paths = sorted((self.data_dir / "data").glob(f"{split}-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No Parquet shards found for split {split!r} under "
                f"{self.data_dir / 'data'}; rebuild the dataset with `regulonado dataset`."
            )
        self._paths = shard_paths
        self._metadata = [pq.read_metadata(str(path)) for path in shard_paths]

        # Cumulative row-group index: group i starts at _group_row_starts[i] and lives in
        # shard/row-group _groups[i]. A row group may hold more than one row
        # (rows_per_row_group > 1), so offsets are counted in rows, not groups.
        self._groups: list[tuple[int, int]] = []
        self._group_row_starts: list[int] = []
        total_rows = 0
        for shard_idx, metadata in enumerate(self._metadata):
            for row_group_idx in range(metadata.num_row_groups):
                self._groups.append((shard_idx, row_group_idx))
                self._group_row_starts.append(total_rows)
                total_rows += metadata.row_group(row_group_idx).num_rows
        self._total_rows = total_rows
        self._files: dict[int, pq.ParquetFile] = {}

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema of the stored (pre-transform) columns."""
        return self._metadata[0].schema.to_arrow_schema()

    def __len__(self) -> int:
        return self._total_rows

    def _file(self, shard_idx: int) -> pq.ParquetFile:
        file = self._files.get(shard_idx)
        if file is None:
            file = pq.ParquetFile(
                str(self._paths[shard_idx]),
                metadata=self._metadata[shard_idx],
                pre_buffer=True,
            )
            self._files[shard_idx] = file
        return file

    def _locate(self, index: int) -> tuple[int, int, int]:
        group_idx = bisect.bisect_right(self._group_row_starts, index) - 1
        shard_idx, row_group_idx = self._groups[group_idx]
        local_row = index - self._group_row_starts[group_idx]
        return shard_idx, row_group_idx, local_row

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self._total_rows
        if not 0 <= index < self._total_rows:
            raise IndexError(f"Index {index} out of range for dataset of length {self._total_rows}")
        shard_idx, row_group_idx, local_row = self._locate(index)
        table = self._file(shard_idx).read_row_group(row_group_idx, columns=self._columns)
        example = _row_to_example(table, local_row, self._columns)
        return self.transform(example) if self.transform is not None else example

    def __getstate__(self) -> dict:
        # Never carry open file handles across a pickle/fork boundary. FileMetaData is
        # picklable in the pyarrow versions this project targets; fall back to re-reading
        # footers lazily in the worker if that ever stops being true.
        state = self.__dict__.copy()
        state["_files"] = {}
        try:
            pickle.dumps(state["_metadata"])
        except Exception:
            state["_metadata"] = [None] * len(self._paths)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if any(metadata is None for metadata in self._metadata):
            self._metadata = [pq.read_metadata(str(path)) for path in self._paths]
