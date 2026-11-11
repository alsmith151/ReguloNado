"""BigWig region summaries, stored separately from discrete BAM event counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from regulonado.counts.dataset import RegionCountData
from regulonado.tracks_table import read_track_table, require_track_format


def signal_tracks(path: str | Path) -> pl.DataFrame:
    table = read_track_table(path)
    require_track_format(table, "bigwig", "BigWig region summaries")
    included = table[table.status == "included"].sort_values("track_index")
    return pl.from_pandas(included)


def count_signal_regions(counter, tracks, out_dir, aggregation="mean", track=None):
    from regulonado._rs import bigwig_regions

    if aggregation not in {"mean", "max", "count", "logcount"}:
        raise ValueError("aggregation must be mean, max, count, or logcount")
    windows = list(counter.regions.select("chrom", "target_start", "target_end").iter_rows())
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = tracks.to_dicts()
    if track is not None:
        rows = [r for r in rows if r["track_name"] == track]
        if not rows:
            raise ValueError(f"no track named {track!r}")
    for row in rows:
        path = directory / f"{row['track_name']}.parquet"
        source = Path(row["resolved_path"])
        stat = source.stat()
        identity = json.dumps(
            {
                "source": str(source.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "aggregation": aggregation,
                "windows": windows,
            },
            sort_keys=True,
        )
        import hashlib

        key = hashlib.sha256(identity.encode()).hexdigest()
        import pyarrow as pa
        import pyarrow.parquet as pq

        if path.exists():
            cached = pq.read_metadata(path).metadata or {}
            if cached.get(b"target_identity") == key.encode():
                continue
        values = np.frombuffer(bigwig_regions(str(source), windows, aggregation), dtype="<f8")
        table = pa.table({"signal": values}).replace_schema_metadata(
            {
                b"target_identity": key.encode(),
                b"aggregation": aggregation.encode(),
                b"windows": json.dumps(windows).encode(),
            }
        )
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)


def gather_signal_regions(counter, tracks, out_dir, dataset_dir, aggregation="mean"):
    import pyarrow.parquet as pq

    names = tracks["track_name"].to_list()
    values = []
    windows = list(counter.regions.select("chrom", "target_start", "target_end").iter_rows())
    for name in names:
        path = Path(out_dir) / f"{name}.parquet"
        table = pq.read_table(path)
        if (table.schema.metadata or {}).get(b"aggregation") != aggregation.encode():
            raise ValueError(f"{name}: cached aggregation does not match {aggregation}")
        if (table.schema.metadata or {}).get(b"windows") != json.dumps(windows).encode():
            raise ValueError(f"{name}: cached scored regions do not match")
        column = table["signal"].to_numpy()
        if len(column) != counter.n_regions:
            raise ValueError(f"{name}: cached region count does not match")
        values.append(column)
    meta = tracks.with_columns(
        pl.lit("bigwig").alias("target_source"),
        pl.lit(aggregation).alias("aggregation"),
        pl.lit("signal").alias("count_unit"),
        pl.lit(0.0).alias("log_size_factor"),
    )
    data = RegionCountData.from_arrays(counter.regions, np.column_stack(values), names, meta)
    data.write(dataset_dir)
    return data
