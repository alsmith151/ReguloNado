"""Track discovery: resolve raw BigWig sources into the ``tracks/_stages/discovered.parquet``.

Existence filtering, dedupe, fingerprinting and track naming all happen here so
every downstream stage (scaling, QC, assembly, build) can assume a validated,
uniquely-named track list and never has to re-derive it.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from regulonado.tracks_table import bigwig_fingerprint

logger = logging.getLogger(__name__)

_DEDUPE_TRACK_MODES = {"none", "identity", "content"}


def _hash_file_blake2b(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _track_file_record(source_index: int, path: str) -> dict:
    p = Path(path).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p.absolute()

    try:
        st = resolved.stat()
        size_bytes = int(st.st_size)
        identity_key = f"inode:{st.st_dev}:{st.st_ino}"
    except OSError:
        size_bytes = None
        identity_key = f"path:{resolved}"

    return {
        "source_index": source_index,
        "path": path,
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "identity_key": identity_key,
    }


def _resolve_bigwig_tracks(
    bigwig_paths: Sequence[str | Path],
    *,
    drop_missing: bool,
    dedupe_tracks: str,
    annotations: dict[str, dict] | None = None,
) -> tuple[list[str], dict]:
    """Filter requested tracks and return final paths plus provenance metadata.

    ``annotations`` maps a resolved BigWig path to biological annotation
    (``condition``, ``track_name``, …) from a track sheet. Annotation is merged
    into each surviving track record; provenance fields always win on a key
    clash, so a sheet can never overwrite dedupe bookkeeping.
    """
    if dedupe_tracks not in _DEDUPE_TRACK_MODES:
        raise ValueError(
            f"dedupe_tracks must be one of {sorted(_DEDUPE_TRACK_MODES)}, got {dedupe_tracks!r}"
        )

    requested = [str(p).strip().strip('"').strip("'") for p in bigwig_paths]
    existing: list[dict] = []
    missing_records: list[dict] = []
    for source_index, path in enumerate(requested):
        if Path(path).expanduser().exists():
            existing.append(_track_file_record(source_index, path))
        else:
            missing_records.append({"source_index": source_index, "path": path})

    if missing_records:
        if drop_missing:
            logger.warning(
                f"Dropping {len(missing_records)}/{len(requested)} missing bigwig paths:\n"
                + "\n".join(f"  {r['path']}" for r in missing_records)
            )
        else:
            raise FileNotFoundError(
                f"{len(missing_records)}/{len(requested)} bigwig paths do not exist:\n"
                + "\n".join(f"  {r['path']}" for r in missing_records)
            )

    identity_canonical: dict[int, int] = {}
    identity_first: dict[str, int] = {}
    survivors: list[dict] = []
    n_identity_dropped = 0
    if dedupe_tracks in {"identity", "content"}:
        for rec in existing:
            source_index = int(rec["source_index"])
            key = str(rec["identity_key"])
            if key in identity_first:
                identity_canonical[source_index] = identity_first[key]
                n_identity_dropped += 1
            else:
                identity_first[key] = source_index
                identity_canonical[source_index] = source_index
                survivors.append(rec)
    else:
        for rec in existing:
            source_index = int(rec["source_index"])
            identity_canonical[source_index] = source_index
            survivors.append(rec)

    content_canonical: dict[int, int] = {
        int(rec["source_index"]): int(rec["source_index"]) for rec in survivors
    }
    content_hash_by_source: dict[int, str] = {}
    n_hashed = 0
    n_content_dropped = 0
    hash_algorithm = "blake2b-256"
    if dedupe_tracks == "content":
        by_size: dict[int, list[dict]] = {}
        for rec in survivors:
            size = rec.get("size_bytes")
            if size is not None:
                by_size.setdefault(int(size), []).append(rec)

        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            first_for_hash: dict[str, int] = {}
            for rec in same_size:
                source_index = int(rec["source_index"])
                digest = _hash_file_blake2b(Path(str(rec["resolved_path"])))
                content_hash_by_source[source_index] = digest
                n_hashed += 1
                if digest in first_for_hash:
                    content_canonical[source_index] = first_for_hash[digest]
                    n_content_dropped += 1
                else:
                    first_for_hash[digest] = source_index

    final_canonical: dict[int, int] = {}
    for rec in existing:
        source_index = int(rec["source_index"])
        identity_source = identity_canonical[source_index]
        final_canonical[source_index] = content_canonical.get(identity_source, identity_source)

    final_source_indices = {
        source_index
        for source_index, canonical in final_canonical.items()
        if source_index == canonical
    }
    final_records: list[dict] = []
    final_track_index_by_source: dict[int, int] = {}
    rec_by_source = {int(rec["source_index"]): rec for rec in existing}
    for rec in existing:
        source_index = int(rec["source_index"])
        if source_index not in final_source_indices:
            continue
        track_index = len(final_records)
        final_track_index_by_source[source_index] = track_index
        content_hash = content_hash_by_source.get(source_index)
        dedupe_method = (
            "content"
            if content_hash is not None
            else ("identity" if dedupe_tracks in {"identity", "content"} else "none")
        )
        dedupe_key = (
            f"content:{content_hash}" if content_hash is not None else str(rec["identity_key"])
        )
        out = {
            "track_index": track_index,
            "source_index": source_index,
            "path": rec["path"],
            "resolved_path": rec["resolved_path"],
            "size_bytes": rec["size_bytes"],
            "dedupe_key": dedupe_key,
            "dedupe_method": dedupe_method,
        }
        if content_hash is not None:
            out["content_hash"] = content_hash
        if annotations:
            annotation = annotations.get(str(rec["resolved_path"]))
            if annotation:
                # Provenance keys take precedence over sheet-supplied ones.
                out = {**annotation, **out}
        final_records.append(out)

    dropped_records: list[dict] = []
    for rec in existing:
        source_index = int(rec["source_index"])
        if source_index in final_source_indices:
            continue
        duplicate_of_source_index = final_canonical[source_index]
        identity_source = identity_canonical[source_index]
        content_hash = content_hash_by_source.get(identity_source)
        dedupe_method = "identity" if identity_source != source_index else "content"
        dropped = {
            "source_index": source_index,
            "path": rec["path"],
            "resolved_path": rec["resolved_path"],
            "size_bytes": rec["size_bytes"],
            "duplicate_of_track_index": final_track_index_by_source[duplicate_of_source_index],
            "duplicate_of_source_index": duplicate_of_source_index,
            "dedupe_method": dedupe_method,
            "dedupe_key": (
                f"content:{content_hash}"
                if dedupe_method == "content" and content_hash is not None
                else str(rec["identity_key"])
            ),
        }
        if dedupe_method == "content" and content_hash is not None:
            dropped["content_hash"] = content_hash
        duplicate_of = rec_by_source[duplicate_of_source_index]
        dropped["duplicate_of_path"] = duplicate_of["path"]
        dropped["duplicate_of_resolved_path"] = duplicate_of["resolved_path"]
        dropped_records.append(dropped)

    final_paths = [str(rec["path"]) for rec in final_records]

    if annotations:
        annotated = {
            str(rec["resolved_path"])
            for rec in final_records
            if str(rec["resolved_path"]) in annotations
        }
        unannotated = len(final_records) - len(annotated)
        if unannotated:
            logger.warning(
                f"{unannotated}/{len(final_records)} track(s) have no track-sheet "
                f"annotation; their categorical ids will be -1"
            )
        unused = set(annotations) - annotated
        if unused:
            sample = "\n".join(f"  {path}" for path in sorted(unused)[:5])
            logger.warning(
                f"{len(unused)} track-sheet row(s) matched no built track "
                f"(dropped as missing or duplicate):\n{sample}"
            )

    if dedupe_tracks != "none":
        logger.info(
            f"Track dedupe mode={dedupe_tracks}: {len(final_paths)} final track(s) from "
            f"{len(requested)} requested; dropped {len(dropped_records)} duplicate(s) "
            f"({n_identity_dropped} identity, {n_content_dropped} content); "
            f"hashed {n_hashed} file(s)"
        )

    provenance = {
        "bigwig_paths": final_paths,
        "final_bigwig_paths": final_paths,
        "requested_bigwig_paths": requested,
        "final_track_records": final_records,
        "dropped_duplicate_tracks": dropped_records,
        "missing_bigwig_paths": missing_records,
        "n_requested_tracks": len(requested),
        "n_missing_tracks": len(missing_records),
        "n_dropped_duplicate_tracks": len(dropped_records),
        "n_final_tracks": len(final_paths),
        "dedupe_tracks": {
            "mode": dedupe_tracks,
            "keep": "first",
            "identity_method": "stat(st_dev,st_ino) after resolve; fallback resolved_path",
            "hash_algorithm": hash_algorithm if dedupe_tracks == "content" else None,
            "hash_limited_to_same_size_groups": dedupe_tracks == "content",
            "n_hashed_files": n_hashed,
            "n_identity_duplicates": n_identity_dropped,
            "n_content_duplicates": n_content_dropped,
        },
    }
    return final_paths, provenance


def _require_unique_included_names(names: Sequence[str]) -> None:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate track name(s) from discovery: {', '.join(duplicates)}. "
            f"Give the source files distinct names, or add a track sheet with explicit "
            f"'track_name' values to disambiguate."
        )


def _dedupe_name(name: str, used: set[str]) -> str:
    """Disambiguate a non-included row's name against ones already placed in the table."""
    if name not in used:
        used.add(name)
        return name
    for candidate in (f"{name}~{i}" for i in itertools.count(1)):
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover


def discover_tracks(
    bigwig_paths: Sequence[str | Path],
    *,
    drop_missing: bool = False,
    dedupe_tracks: str = "none",
    annotations: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Resolve raw BigWig paths into the discovery-stage track table.

    One row per requested path: ``included`` (survives existence + dedupe
    filtering), ``dropped_duplicate``, or ``missing``. ``track_name`` comes
    from the track sheet when ``annotations`` supplies one, else the file
    stem; colliding stems among *included* tracks raise here, matching
    :meth:`regulonado.tracks.TrackSheet._require_unique_track_names`.
    """
    final_paths, provenance = _resolve_bigwig_tracks(
        bigwig_paths,
        drop_missing=drop_missing,
        dedupe_tracks=dedupe_tracks,
        annotations=annotations,
    )

    used_names: set[str] = set()
    included_names: list[str] = []
    rows: list[dict] = []
    for rec in provenance["final_track_records"]:
        row = dict(rec)
        track_name = row.pop("track_name", None) or Path(row["path"]).stem
        included_names.append(track_name)
        used_names.add(track_name)
        fp = bigwig_fingerprint(row["resolved_path"])
        content_hash = row.pop("content_hash", None)
        row.update(fp)
        if content_hash is not None:
            row["fp_content_hash"] = content_hash
        rows.append({"track_name": track_name, "status": "included", **row})
    _require_unique_included_names(included_names)

    for rec in provenance["dropped_duplicate_tracks"]:
        name = _dedupe_name(Path(rec["path"]).stem, used_names)
        rows.append(
            {
                "track_name": name,
                "status": "dropped_duplicate",
                "path": rec["path"],
                "resolved_path": rec["resolved_path"],
                "dedupe_key": rec["dedupe_key"],
                "dedupe_method": rec["dedupe_method"],
                "duplicate_of": Path(rec["duplicate_of_path"]).stem,
            }
        )

    for rec in provenance["missing_bigwig_paths"]:
        name = _dedupe_name(Path(rec["path"]).stem, used_names)
        rows.append({"track_name": name, "status": "missing", "path": rec["path"]})

    df = pd.DataFrame(rows)
    df.attrs["dedupe_tracks"] = provenance["dedupe_tracks"]
    return df
