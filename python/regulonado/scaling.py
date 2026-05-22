from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

BAMNADO = os.environ.get("BAMNADO", "bamnado")


def infer_scale_factor(bw: Path) -> pd.Series:
    path = bw.resolve()
    cmd = [BAMNADO, "bigwig-infer-scale", "--bigwig", str(path), "--format", "json"]
    output = subprocess.run(cmd, check=True, capture_output=True, text=True)
    result = json.loads(output.stdout)
    result["samplename"] = bw.stem
    result["path"] = str(path)
    return pd.Series(result)


def infer_scale_factors(
    bw_files: list[Path],
    max_workers: int = 16,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(infer_scale_factor, bw): bw for bw in bw_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring scale factors"):
            rows.append(future.result())
    return pd.concat(rows, axis=1).T.reset_index(drop=True)


def save_scale_factors(
    df: pd.DataFrame,
    output: Path,
    fmt: Literal["csv", "parquet"] = "parquet",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)
