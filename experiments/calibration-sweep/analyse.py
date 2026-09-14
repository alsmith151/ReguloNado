"""Print magnitude-ranked runs from a W&B sweep."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import wandb

app = typer.Typer(help=__doc__)


def load_runs(sweep_id: str) -> pl.DataFrame:
    api = wandb.Api()
    rows = []
    for run in api.sweep(sweep_id).runs:
        summary = dict(run.summary)
        rows.append({"run": run.name, "state": run.state,
                     "total_ratio_median": summary.get("total_ratio_median"),
                     "log_ratio_total_median": summary.get("log_ratio_total_median"),
                     "amplitude_ratio_median": summary.get("amplitude_ratio_median"),
                     "dispersion_slope": summary.get("dispersion_slope"),
                     "pearson_bin_median": summary.get("pearson_bin_median"),
                     "config": str(run.config.get("overrides", ""))})
    return pl.DataFrame(rows)


@app.command()
def main(
    sweep_id: Annotated[str, typer.Argument(help="W&B entity/project/sweep ID.")],
    output: Annotated[
        Path, typer.Option("--output", help="Local parquet report path.")
    ] = Path("experiments/calibration-sweep/results/runs.parquet"),
) -> None:
    frame = load_runs(sweep_id)
    frame = frame.with_columns((pl.col("total_ratio_median") - 1).abs().alias("total_ratio_error"))
    print(frame.sort("total_ratio_error").to_pandas().to_string(index=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)


if __name__ == "__main__":
    app()
