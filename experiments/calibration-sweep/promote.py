"""Emit tier-2 override files for the best completed W&B runs."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import wandb

app = typer.Typer(help=__doc__)


@app.command()
def main(
    sweep_id: Annotated[str, typer.Argument(help="W&B entity/project/sweep ID.")],
    top_n: Annotated[int, typer.Option("--top-n", min=1, help="Number of variants to promote.")] = 3,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directory for promoted override files.")
    ] = Path("experiments/calibration-sweep/results/promoted"),
) -> None:
    runs = [r for r in wandb.Api().sweep(sweep_id).runs if r.state == "finished"]
    runs.sort(key=lambda r: abs(float(r.summary.get("total_ratio_median", float("inf"))) - 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs[:top_n], 1):
        overrides = run.config.get("overrides", "")
        (output_dir / f"variant_{index}.txt").write_text(str(overrides) + "\n")


if __name__ == "__main__":
    app()
