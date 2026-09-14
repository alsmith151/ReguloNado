"""Cache trunk features and transformed labels for head-only calibration sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer

app = typer.Typer(help=__doc__)


def save_feature_cache(features: torch.Tensor, labels: torch.Tensor, path: str | Path) -> None:
    """Save a CPU feature/label cache with an atomic-ish single torch artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.detach().to("cpu", dtype=torch.bfloat16),
                "labels": labels.detach().to("cpu")}, destination)


@app.command()
def main(
    input: Annotated[Path, typer.Option("--input", help="Torch file with features and labels.")],
    output: Annotated[Path, typer.Option("--output", help="Feature cache output path.")],
) -> None:
    payload = torch.load(input, map_location="cpu", weights_only=True)
    save_feature_cache(payload["features"], payload["labels"], output)


if __name__ == "__main__":
    app()
