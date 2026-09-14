from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

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


def count_arrow_split_rows(split_dir: Path) -> int:
    """Count rows in a saved Arrow split without decoding any column data.

    ``load_from_disk`` calls ``read_all()`` on every shard, which decompresses
    compressed shards into RAM. Reading memory-mapped shards with no included
    fields still yields each record batch's ``num_rows`` but skips every buffer.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    state = json.loads((split_dir / "state.json").read_text())
    options = ipc.IpcReadOptions(included_fields=[])
    rows = 0
    for data_file in state["_data_files"]:
        with pa.memory_map(str(split_dir / data_file["filename"])) as source:
            rows += sum(batch.num_rows for batch in ipc.open_stream(source, options=options))
    return rows
