"""``tracks.parquet``: the one canonical track table.

A strict superset of every pipeline stage's output (discovery, scaling, QC) —
every column any stage produces appears here for every track any stage saw.
The stage files under ``tracks/_stages/`` are rule plumbing kept only so
re-running assembly does not re-scan every BigWig; nothing should read them
directly. See CLAUDE.md's "Track table" section.

Column prefixes namespace a wide table: ``fp_`` fingerprint, ``scale_``
scaling, ``qc_`` QC. Identity/annotation/dedupe columns keep bare names.
``track_name`` is the join key everywhere; ``track_index`` is positional and
renumbers under filtering.

Clip thresholds: a track's stored signal can be read in one of three unit spaces
depending on ``data.label_space``/scaling method, so there are three clip-threshold
column families rather than one ambiguous pair (see ``training.runner.
resolve_scale_and_clip``, which picks between them):

- ``scale_clip_soft_counts``/``scale_clip_hard_counts`` — stored mean-coverage units
  (the BigWig's own unit); QC-derived interval-mean quantiles, independent of
  ``data.count_unit`` and of which scaling method (if any) ran.
- ``scale_clip_soft_anchor``/``scale_clip_hard_anchor`` — anchor units (1.0 ==
  housekeeping-promoter level); populated only by the anchor scaling method.
- ``scale_clip_soft_squash``/``scale_clip_hard_squash`` — raw-count units, pre-squash;
  populated by every other scaling method.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

SCHEMA_VERSION = "1"

STATUSES = ("included", "dropped_duplicate", "missing", "qc_failed", "excluded")

# Sheet/discovery label columns -> the categorical id field derived at load time.
# Canonical definition; regulonado.tracks imports this. Ids are never stored,
# only labels, so they are always recomputed by sorted factorisation (see
# TrackSheet.to_track_records).
CATEGORICAL_FIELDS: dict[str, str] = {
    "condition": "condition_id",
    "source": "source_id",
    "assay": "assay_type_id",
    "ip": "target_id",
}

_METADATA_ARROW_KEY = b"regulonado"


class TrackTableSchema(pa.DataFrameModel):
    """Contract for ``tracks.parquet``. Additive stage-specific columns pass through."""

    track_name: Series[str] = pa.Field(unique=True, nullable=False)
    status: Series[str] = pa.Field(isin=list(STATUSES))
    track_index: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0)

    class Config:
        coerce = True
        strict = False  # stage-specific metric columns are additive

    @pa.dataframe_check
    def track_index_is_contiguous(cls, df: pd.DataFrame) -> Series[bool]:
        """``track_index`` is 0..n-1 over included rows, null everywhere else."""
        included = df["status"] == "included"
        idx = df["track_index"]
        ok = pd.Series(True, index=df.index)
        ok.loc[~included] = idx[~included].isna().to_numpy(dtype=bool)
        included_idx = idx[included]
        if included_idx.isna().any():
            ok[included & idx.isna()] = False
            included_idx = included_idx.dropna()
        expected = pd.RangeIndex(len(included_idx))
        matches = pd.Series(
            sorted(included_idx.astype("int64")) == list(expected), index=[True]
        ).iloc[0]
        if not matches:
            ok[included] = False
        return ok

    @pa.dataframe_check
    def scaling_columns_present_together(cls, df: pd.DataFrame) -> Series[bool]:
        """Per row and per clip-unit family: soft/hard agree, and a populated family
        implies ``scale_factor`` is populated too (not the reverse).

        Three independent clip families, one per unit space a track's stored signal
        might be read in (see the module docstring's "Clip thresholds" note):

        - ``scale_clip_soft_anchor``/``scale_clip_hard_anchor`` (anchor units):
          populated only by the anchor scaling method.
        - ``scale_clip_soft_squash``/``scale_clip_hard_squash`` (raw-count units,
          pre-squash): populated by every other scaling method.
        - ``scale_clip_soft_counts``/``scale_clip_hard_counts`` (stored mean-coverage
          units): checked only against each other, *not* ``scale_factor`` — these
          come from a QC run's interval-quantile computation, independent of which
          (if any) scaling method ran, and ``label_space: counts`` has no
          "scale factor" concept at all.

        ``scale_factor`` is shared by the anchor and squash families (exactly one of
        the two is populated alongside it, depending on which scaling method ran), so
        the check cannot be symmetric the way it is for a single family: "scale_factor
        present" does not imply either family is present, only the other way round.
        ``scale_background`` is excluded entirely: it is only ever populated by the
        anchor method, so it is legitimately absent even when a track has a
        ``scale_factor``/anchor-clip triple.
        """
        # Not symmetric: scale_factor is shared by both the anchor and squash families
        # (whichever scaling method ran populates scale_factor plus exactly one of the
        # two), so "scale_factor present" cannot imply "this family present" -- only
        # "this family present" implies "scale_factor present".
        ok = pd.Series(True, index=df.index)
        for suffix in ("anchor", "squash"):
            soft_col, hard_col = f"scale_clip_soft_{suffix}", f"scale_clip_hard_{suffix}"
            if soft_col not in df.columns or hard_col not in df.columns:
                continue
            soft_null = df[soft_col].isna()
            hard_null = df[hard_col].isna()
            ok &= soft_null == hard_null  # soft/hard of one family always agree
            if "scale_factor" in df.columns:
                populated = ~soft_null
                ok &= ~(populated & df["scale_factor"].isna())
        counts_soft, counts_hard = "scale_clip_soft_counts", "scale_clip_hard_counts"
        if counts_soft in df.columns and counts_hard in df.columns:
            ok &= df[counts_soft].isna() == df[counts_hard].isna()
        return ok


def _schema_metadata_dict(df: pd.DataFrame) -> dict[str, Any]:
    return {key: value for key, value in df.attrs.items()}


def _translate_failure_cases(exc: pa.errors.SchemaErrors) -> str:
    """Name the offending ``track_name`` for each failing row, where derivable."""
    lines = [str(exc)]
    cases = getattr(exc, "failure_cases", None)
    if cases is None or "column" not in cases.columns:
        return "\n".join(lines)
    names = cases.loc[cases["column"] == "track_name", "failure_case"].unique().tolist()
    if names:
        lines.append(f"Offending track_name(s): {', '.join(map(str, names))}")
    return "\n".join(lines)


def validate_track_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against :class:`TrackTableSchema`, naming failing tracks."""
    try:
        return TrackTableSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(_translate_failure_cases(exc)) from exc


def write_track_table(df: pd.DataFrame, path: str | Path, **scalars: Any) -> Path:
    """Validate, attach run-level scalar metadata, and atomically write ``df``.

    ``scalars`` are merged into ``df.attrs`` and serialised into the Arrow
    schema metadata under the ``b"regulonado"`` key, so they round-trip through
    ``pq.write_table``/``read_table`` without becoming spurious columns.
    """
    import pyarrow as pa_arrow
    import pyarrow.parquet as pq

    df = validate_track_table(df)
    attrs = {**df.attrs, **scalars, "schema_version": SCHEMA_VERSION}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa_arrow.Table.from_pandas(df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, _METADATA_ARROW_KEY: json.dumps(attrs).encode()}
    table = table.replace_schema_metadata(new_meta)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, path)
    return path


def read_track_table(path: str | Path) -> pd.DataFrame:
    """Read and validate ``tracks.parquet``, lifting run-level scalars into ``df.attrs``."""
    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    raw = meta.get(_METADATA_ARROW_KEY)
    attrs: dict[str, Any] = json.loads(raw) if raw else {}

    df = table.to_pandas()
    df.attrs = attrs
    df = validate_track_table(df)

    schema_version = attrs.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {schema_version!r} != expected {SCHEMA_VERSION!r}; "
            f"rebuild the track table"
        )
    bin_size = attrs.get("bin_size")
    shift_max_bp = attrs.get("shift_max_bp")
    if bin_size is not None and shift_max_bp is not None and shift_max_bp % bin_size != 0:
        raise ValueError(
            f"{path}: shift_max_bp ({shift_max_bp}) must be a multiple of bin_size ({bin_size})"
        )
    return df


def bigwig_fingerprint(path: str | Path) -> dict[str, Any]:
    """Cheap content-derived fingerprint: verified against disk, never joined on.

    ``size_bytes``/``mtime_ns`` catch a rewrite instantly; ``bases_covered``/
    ``genome_sum`` (the BigWig header summary) catch a same-size rewrite that
    those two would miss.
    """
    import pybigtools

    resolved = Path(path).resolve()
    st = resolved.stat()
    reader = pybigtools.open(str(resolved))
    try:
        summary = reader.info()["summary"]
    finally:
        close = getattr(reader, "close", None)
        if close:
            close()
    return {
        "fp_size_bytes": int(st.st_size),
        "fp_mtime_ns": int(st.st_mtime_ns),
        "fp_bases_covered": int(summary.get("basesCovered", 0)),
        "fp_genome_sum": float(summary.get("sum", 0.0)),
    }


def verify_fingerprint(path: str | Path, expected: dict[str, Any]) -> list[str]:
    """Return mismatch descriptions between disk and a stored fingerprint; empty if consistent."""
    current = bigwig_fingerprint(path)
    problems = []
    for key in ("fp_bases_covered", "fp_genome_sum"):
        if expected.get(key) is not None and current[key] != expected[key]:
            problems.append(f"{key}: recorded={expected[key]!r} disk={current[key]!r}")
    if not problems and expected.get("fp_size_bytes") is not None:
        if current["fp_size_bytes"] != expected["fp_size_bytes"]:
            problems.append(
                f"fp_size_bytes: recorded={expected['fp_size_bytes']!r} "
                f"disk={current['fp_size_bytes']!r}"
            )
    return problems


def to_track_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Included rows, ordered by ``track_index``, as the flat dicts training code reads.

    Bridges the wide, namespaced table back to the field names
    ``regulonado.training.runner`` / ``regulonado.design.objective`` /
    ``regulonado.inference`` already expect (``scale_factor``, ``clip_soft``,
    ``clip_hard``, ``background``, ``condition_id``, ...). Categorical ids are
    *derived* here by sorted factorisation over the label columns — labels are
    the only thing ever stored, so an id means the same thing regardless of
    which stage populated the label.
    """
    included = df[df["status"] == "included"].sort_values("track_index")

    numeric_rename = {
        "scale_factor": "scale_factor",
        "scale_background": "background",
        # Three unit-specific clip families (see the module docstring's "Clip
        # thresholds" note and TrackTableSchema.scaling_columns_present_together) —
        # training code (runner.resolve_scale_and_clip) picks the one matching the
        # active label space/scaling method rather than reading one ambiguous pair.
        "scale_clip_soft_counts": "clip_soft_counts",
        "scale_clip_hard_counts": "clip_hard_counts",
        "scale_clip_soft_anchor": "clip_soft_anchor",
        "scale_clip_hard_anchor": "clip_hard_anchor",
        "scale_clip_soft_squash": "clip_soft_squash",
        "scale_clip_hard_squash": "clip_hard_squash",
        "scale_anchor_reference": "anchor_reference",
        "scale_library_size": "library_size",
        "fp_genome_sum": "genome_sum",
        "fragment_length": "fragment_length",
    }
    records: list[dict[str, Any]] = []
    for _, row in included.iterrows():
        record: dict[str, Any] = {
            "track_name": row["track_name"],
            "track_index": int(row["track_index"]),
        }
        for column in ("sample_id", "path", "resolved_path"):
            if column in row and pd.notna(row[column]):
                record[column] = row[column]
        for src, dst in numeric_rename.items():
            if src in row and pd.notna(row[src]):
                record[dst] = float(row[src])
        for column in (
            "project",
            "scaling_group",
            "consensus_group",
            "control",
            "group",
            "condition",
            "source",
            "assay",
            "assay_class",
            "ip",
        ):
            if column in row and pd.notna(row[column]):
                record[column] = row[column]
        if "timepoint_minutes" in row and pd.notna(row["timepoint_minutes"]):
            record["timepoint_minutes"] = float(row["timepoint_minutes"])
        records.append(record)

    for label_column, id_field in CATEGORICAL_FIELDS.items():
        if label_column not in included.columns:
            continue
        labels = sorted({r[label_column] for r in records if label_column in r})
        lookup = {label: i for i, label in enumerate(labels)}
        for record in records:
            record[id_field] = lookup.get(record.get(label_column), -1)

    return records
