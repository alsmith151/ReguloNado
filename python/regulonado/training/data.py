from __future__ import annotations

from collections.abc import Mapping, Sequence

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
