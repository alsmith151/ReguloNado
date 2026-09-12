"""Track sheets: the mapping from BigWig files to biological annotation.

ReguloNado's training code reads per-track categorical ids (``condition_id``,
``source_id``, ``assay_type_id``, ``target_id``) and ``timepoint_minutes``
from the dataset metadata, but nothing populated them — tracks were discovered by
globbing a directory. A track sheet fills that gap.

Column names deliberately match SeqNado's design sheet
(``seqnado/inputs/validation.py``) so a SeqNado user recognises the file and a
SeqNado design is close to a drop-in: ``sample_id``, ``condition``,
``scaling_group``, ``consensus_group``, ``group``, ``ip``, ``control``, ``assay``.
Columns ReguloNado adds are marked below.

Sheets can be written by hand or derived from one or more SeqNado projects; see
:meth:`TrackSheet.from_seqnado_projects` for what aggregating several projects
does and does not guarantee.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from regulonado.tracks_table import CATEGORICAL_FIELDS

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# SeqNado constrains label columns to this alphabet because they are interpolated
# into output paths. We enforce the same rule so a sheet round-trips between the
# two tools, and because our track names end up in file names too.
LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Separator between the project namespace and the sample name when aggregating.
# Stays within LABEL_PATTERN so composed names remain valid SeqNado labels.
PROJECT_SEPARATOR = "__"

_COLUMNS = (
    # Shared with SeqNado's design sheet.
    "sample_id",
    "condition",
    "scaling_group",
    "consensus_group",
    "group",
    "ip",
    "control",
    "assay",
    # ReguloNado-only.
    "track_name",
    "bigwig",
    "bam",
    "source",
    "timepoint_minutes",
    "project",
    "method",
    "scale",
)


def _clean(value: Any) -> str | None:
    """Normalise a sheet cell to a non-empty string or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "-"}:
        return None
    return text


def _track_record_from_csv_row(
    row: dict[str, Any], *, index: int, base: Path, path: Path
) -> "TrackRecord":
    """Parse and normalise one raw CSV row into a :class:`TrackRecord`.

    Cleans every recognised column, resolves ``bigwig``/``bam`` paths relative
    to the sheet's own directory, and derives ``sample_id``/``track_name``
    from each other when only one of them (or ``bigwig``) is present.
    """
    data: dict[str, Any] = {key: _clean(value) for key, value in row.items() if key in _COLUMNS}

    for column in ("bigwig", "bam"):
        value = data.get(column)
        if value:
            resolved = Path(str(value)).expanduser()
            if not resolved.is_absolute():
                resolved = (base / resolved).resolve()
            data[column] = resolved
        else:
            data[column] = None

    # sample_id, track_name and bigwig can each stand in for the others;
    # what we cannot do is proceed with none of them.
    bigwig = data.get("bigwig")
    if not data.get("sample_id"):
        data["sample_id"] = data.get("track_name") or (
            bigwig.stem if bigwig is not None else None
        )
    if not data.get("sample_id"):
        raise ValueError(
            f"{path}: row {index + 1} has none of 'sample_id', 'track_name' "
            f"or 'bigwig'; at least one is needed to identify the track"
        )
    if not data.get("track_name") and bigwig is not None:
        data["track_name"] = data["sample_id"]

    return TrackRecord(**data)


def _filter_seqnado_index(
    index: "pd.DataFrame",
    *,
    method: str | None,
    scale: str | None,
    merged: bool | None,
    keep_stranded: bool,
) -> "pd.DataFrame":
    """Apply method/scale/merged/stranded filters to a SeqNado bigwig index."""
    if method is not None:
        index = index[index["method"] == method]
    if scale is not None:
        index = index[index["scale"] == scale]
    if merged is not None:
        index = index[index["merged"] == merged]

    if not keep_stranded:
        stranded = index["strand"].notna()
        if stranded.any():
            # A +/- pair is two tracks in our model, which is fine — but
            # silently doubling the track count is not, so require opt-in.
            logger.warning(
                f"Dropping {int(stranded.sum())} stranded BigWigs "
                f"(pass keep_stranded=True to include them as separate tracks)"
            )
            index = index[~stranded]

    return index


def _apply_sample_filters(
    index: "pd.DataFrame", project: Any, filters: dict[str, Any]
) -> "pd.DataFrame":
    """Restrict an index to samples selected by SeqNado's own design-file filters."""
    if not filters:
        return index
    # Sample-level filters (condition/antibody/group) are resolved by SeqNado
    # against its design file, so ask it rather than reimplementing.
    allowed = {path.resolve() for path in project.bigwigs(**filters)}
    return index[index["path"].map(lambda p: Path(p).resolve() in allowed)]


def _track_records_from_seqnado_index(
    index: "pd.DataFrame",
    project: Any,
    bams: dict[str, Path],
    *,
    project_name: str | None,
    assay: str | None,
) -> list["TrackRecord"]:
    """Turn a filtered SeqNado bigwig index into track records."""
    records: list[TrackRecord] = []
    for row in index.sort_values(["sample", "method", "scale"]).itertuples():
        sample = str(row.sample)
        meta = project.metadata_for(sample) or {}

        track_name = sample
        if project_name:
            track_name = f"{project_name}{PROJECT_SEPARATOR}{sample}"

        records.append(
            TrackRecord(
                track_name=track_name,
                bigwig=Path(row.path),
                bam=bams.get(sample),
                sample_id=sample,
                # SeqNado renames design's `ip` to `antibody` internally;
                # map it back so the sheet uses the design-sheet name.
                ip=_clean(meta.get("antibody")),
                condition=_clean(meta.get("condition")),
                scaling_group=_clean(meta.get("group")),
                assay=_clean(assay),
                project=project_name,
                method=_clean(row.method),
                scale=_clean(row.scale),
            )
        )
    return records


def _unique_project_names(entries: Sequence[dict[str, Any]]) -> list[str]:
    """Return each entry's project name, in order, raising if any repeat."""
    names = [str(entry.get("name") or Path(entry["path"]).parent.name) for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Project names must be unique; repeated: {', '.join(duplicates)}")
    return names


def _check_genome_consistency(
    genomes: dict[str, str | None],
    unknown_genome: list[str],
    *,
    single: bool,
    assume_same_genome: bool,
) -> None:
    """Raise (or warn) when aggregated SeqNado projects disagree on reference genome."""
    distinct = {genome for genome in genomes.values() if genome}
    if len(distinct) > 1:
        listing = ", ".join(f"{name}={genome}" for name, genome in sorted(genomes.items()))
        raise GenomeMismatchError(
            f"SeqNado projects use different reference genomes ({listing}). "
            f"Signal bins only correspond across projects built on the same "
            f"reference; rebuild them on one genome before aggregating."
        )

    # With one project there is nothing to be inconsistent with, so an
    # unreadable config is not a problem worth blocking on.
    if unknown_genome and not single:
        listing = ", ".join(sorted(unknown_genome))
        if not assume_same_genome:
            raise GenomeMismatchError(
                f"Could not read the reference genome for: {listing}. "
                f"Pass assume_same_genome=True (CLI: --assume-same-genome) if "
                f"you are certain every project used the same reference."
            )
        logger.warning(
            f"Assuming a shared reference genome for projects with no readable "
            f"config: {listing}"
        )


def _build_seqnado_catalogue(
    projects: dict[str, str | Path],
    *,
    method: str | None,
    scale: str | None,
    merged: bool | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index every configured SeqNado project by sample id.

    Returns, per project name, ``sample_id -> {bigwig, bam, assay, method,
    scale, metadata}`` for the first-seen sub-project entry of each sample.
    """
    from regulonado._seqnado import is_multi_project, open_project

    catalogue: dict[str, dict[str, dict[str, Any]]] = {}
    for name, project_dir in projects.items():
        opened = open_project(project_dir)
        subs = list(opened.items()) if is_multi_project(opened) else [(opened.assay, opened)]
        entries: dict[str, dict[str, Any]] = {}
        for assay, sub in subs:
            index = sub.bigwig_dataframe()
            if index.empty:
                continue
            if method is not None:
                index = index[index["method"] == method]
            if scale is not None:
                index = index[index["scale"] == scale]
            if merged is not None:
                index = index[index["merged"] == merged]
            bams = {path.stem: path for path in sub.bams()}
            for row in index.itertuples():
                sample = str(row.sample)
                if sample in entries:
                    continue
                entries[sample] = {
                    "bigwig": Path(row.path),
                    "bam": bams.get(sample),
                    "assay": _clean(assay),
                    "method": _clean(row.method),
                    "scale": _clean(row.scale),
                    "metadata": sub.metadata_for(sample) or {},
                }
        catalogue[name] = entries
    return catalogue


def _resolve_track_record(
    record: "TrackRecord",
    catalogue: dict[str, dict[str, dict[str, Any]]],
    *,
    default_project: str | None,
    namespaced: bool,
    method: str | None,
    scale: str | None,
) -> None:
    """Fill one unresolved record's bigwig/bam/annotation from the catalogue, in place.

    Sheet values already present always win; the catalogue only fills blanks.
    """
    project_name = record.project or default_project
    if project_name is None:
        raise ValueError(
            f"Track '{record.sample_id}' has no 'bigwig' path and no "
            f"'project' column value; with several projects configured "
            f"({', '.join(sorted(catalogue))}) the row must say which one."
        )
    if project_name not in catalogue:
        raise ValueError(
            f"Track '{record.sample_id}' names project {project_name!r}, "
            f"which is not configured. Known: {', '.join(sorted(catalogue)) or 'none'}"
        )

    entry = catalogue[project_name].get(record.sample_id)
    if entry is None:
        available = sorted(catalogue[project_name])
        hint = ", ".join(available[:5]) + (" …" if len(available) > 5 else "")
        raise ValueError(
            f"Sample {record.sample_id!r} not found in project "
            f"{project_name!r} for method={method!r}, scale={scale!r}. "
            f"Samples present: {hint or 'none'}"
        )

    record.project = project_name
    record.bigwig = entry["bigwig"]
    if record.bam is None:
        record.bam = entry["bam"]
    if record.track_name is None:
        record.track_name = (
            f"{project_name}{PROJECT_SEPARATOR}{record.sample_id}"
            if namespaced
            else record.sample_id
        )

    metadata = entry["metadata"]
    if record.condition is None:
        record.condition = _clean(metadata.get("condition"))
    if record.ip is None:
        # SeqNado renames the design sheet's `ip` to `antibody` internally.
        record.ip = _clean(metadata.get("antibody"))
    if record.scaling_group is None:
        record.scaling_group = _clean(metadata.get("group")) or (
            project_name if namespaced else None
        )
    if record.assay is None:
        record.assay = entry["assay"]
    if record.method is None:
        record.method = entry["method"]
    if record.scale is None:
        record.scale = entry["scale"]


class TrackRecord(BaseModel):
    """One BigWig track and its annotation.

    ``track_name``, ``bigwig`` and ``bam`` are left unset when the sheet is an
    overlay on a SeqNado project — SeqNado already derives them from
    ``sample_id``, so :meth:`TrackSheet.resolve_from_projects` fills them in.
    They are required by the time the sheet reaches the dataset builder.
    """

    track_name: str | None = None
    bigwig: Path | None = None
    bam: Path | None = None

    # Shared SeqNado vocabulary.
    sample_id: str
    condition: str | None = None
    scaling_group: str | None = None
    consensus_group: str | None = None
    group: str | None = None
    ip: str | None = None
    control: str | None = None
    assay: str | None = None

    # ReguloNado additions.
    source: str | None = None
    timepoint_minutes: float | None = None
    project: str | None = None
    method: str | None = None
    scale: str | None = None

    @field_validator(
        "track_name",
        "sample_id",
        "condition",
        "scaling_group",
        "consensus_group",
        "group",
        "ip",
        "control",
        "assay",
        "source",
        "project",
    )
    @classmethod
    def _validate_label(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if not LABEL_PATTERN.match(value):
            raise ValueError(
                f"{info.field_name}={value!r} contains characters outside "
                f"[a-zA-Z0-9_-]; these labels are interpolated into file paths "
                f"(same rule as SeqNado's design sheet)"
            )
        return value


class GenomeMismatchError(ValueError):
    """Raised when aggregated projects were not built against one reference."""


class TrackSheet(BaseModel):
    """An ordered collection of annotated tracks."""

    records: list[TrackRecord] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def bigwig_paths(self) -> list[Path]:
        """Track BigWig paths, in sheet order. Order defines track order."""
        self.require_resolved()
        return [record.bigwig for record in self.records if record.bigwig is not None]

    @property
    def bam_paths(self) -> list[Path | None]:
        return [record.bam for record in self.records]

    def require_resolved(self) -> None:
        """Raise unless every row has a BigWig path.

        Rows may legitimately arrive without one — a sheet that overlays a
        SeqNado project only needs ``sample_id`` — but nothing downstream can
        proceed until :meth:`resolve_from_projects` has filled them in.
        """
        missing = [record.sample_id for record in self.records if record.bigwig is None]
        if missing:
            listing = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
            raise ValueError(
                f"{len(missing)} track sheet row(s) have no 'bigwig' path and were "
                f"not resolved against a SeqNado project: {listing}. Either add a "
                f"'bigwig' column or configure the project the 'project' column names."
            )

    # ------------------------------------------------------------------ #
    #  CSV round-trip                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        projects: dict[str, str | Path] | None = None,
        **resolve_options: Any,
    ) -> "TrackSheet":
        """Read a track sheet.

        A row needs either a ``bigwig`` path or a ``sample_id`` that a SeqNado
        project can resolve. SeqNado already derives track paths, BAMs and
        per-sample annotation from a sample name, so a sheet that overlays a
        project only has to carry what SeqNado cannot know (``source``,
        ``timepoint_minutes``) plus any deliberate overrides:

        .. code-block:: text

            project,sample_id,source,timepoint_minutes
            expA,CTCF_DMSO,K562,0
            expA,CTCF_IAA,K562,120

        ``projects`` maps a project name to its SeqNado output directory; extra
        keyword arguments are forwarded to :meth:`resolve_from_projects`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Track sheet not found: {path}")

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = set(reader.fieldnames or ())

        if not rows:
            raise ValueError(f"Track sheet is empty: {path}")

        unknown = columns - set(_COLUMNS)
        if unknown:
            logger.warning(f"{path}: ignoring unrecognised column(s): {', '.join(sorted(unknown))}")

        base = path.parent
        records = [
            _track_record_from_csv_row(row, index=index, base=base, path=path)
            for index, row in enumerate(rows)
        ]

        sheet = cls(records=records)
        if projects:
            sheet.resolve_from_projects(projects, **resolve_options)
        else:
            sheet._fill_track_names()
        sheet._require_unique_track_names()
        return sheet

    def to_csv(self, path: str | Path) -> Path:
        """Write the sheet, omitting columns that are empty for every track."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                column: ("" if (value := getattr(record, column, None)) is None else str(value))
                for column in _COLUMNS
            }
            for record in self.records
        ]
        used = [column for column in _COLUMNS if any(row[column] for row in rows)]

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=used, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    # ------------------------------------------------------------------ #
    #  SeqNado projects                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_seqnado_project(
        cls,
        project_dir: str | Path,
        *,
        name: str | None = None,
        method: str | None = "deeptools",
        scale: str | None = "unscaled",
        merged: bool | None = False,
        keep_stranded: bool = False,
        **filters: Any,
    ) -> "TrackSheet":
        """Build a sheet from one SeqNado output directory.

        Delegates all discovery to SeqNado's project API rather than reproducing
        its path conventions: ``bigwig_dataframe()`` for the track index,
        ``bams()`` for alignments, ``metadata_for()`` for per-sample annotation.

        Parameters mirror ``SeqNadoProject.bigwigs``. ``name`` namespaces the
        tracks; it is only required when aggregating several projects.
        """
        from regulonado._seqnado import is_multi_project, open_project

        project_dir = Path(project_dir)
        project = open_project(project_dir)

        if is_multi_project(project):
            records: list[TrackRecord] = []
            for assay, sub in project.items():
                records.extend(
                    cls._records_for_project(
                        sub,
                        project_name=name,
                        assay=assay,
                        method=method,
                        scale=scale,
                        merged=merged,
                        keep_stranded=keep_stranded,
                        filters=filters,
                    )
                )
        else:
            records = cls._records_for_project(
                project,
                project_name=name,
                assay=project.assay,
                method=method,
                scale=scale,
                merged=merged,
                keep_stranded=keep_stranded,
                filters=filters,
            )

        if not records:
            raise ValueError(
                f"No BigWig tracks found in {project_dir} for "
                f"method={method!r}, scale={scale!r}, merged={merged!r}. "
                f"Check 'seqnado' output layout or relax the filters."
            )

        sheet = cls(records=records)
        sheet._require_unique_track_names()
        return sheet

    @classmethod
    def _records_for_project(
        cls,
        project: Any,
        *,
        project_name: str | None,
        assay: str | None,
        method: str | None,
        scale: str | None,
        merged: bool | None,
        keep_stranded: bool,
        filters: dict[str, Any],
    ) -> list[TrackRecord]:
        """Turn one SeqNadoProject into track records."""
        index = project.bigwig_dataframe()
        if index.empty:
            return []

        index = _filter_seqnado_index(
            index, method=method, scale=scale, merged=merged, keep_stranded=keep_stranded
        )
        index = _apply_sample_filters(index, project, filters)
        if index.empty:
            return []

        bams = {path.stem: path for path in project.bams()}
        return _track_records_from_seqnado_index(
            index, project, bams, project_name=project_name, assay=assay
        )

    @classmethod
    def from_seqnado_projects(
        cls,
        entries: Sequence[dict[str, Any]],
        *,
        assume_same_genome: bool = False,
    ) -> "TrackSheet":
        """Aggregate several independent SeqNado projects into one sheet.

        ``entries`` is a sequence of dicts with a ``path`` and a ``name``
        (the namespace label), plus any keyword accepted by
        :meth:`from_seqnado_project`.

        Aggregation is not a plain concatenation:

        * Track names are namespaced ``{project}__{sample}``, because sample
          names such as ``input_1`` recur across projects.
        * Every project must have been built against the same reference genome —
          otherwise bins do not correspond and the merged dataset is meaningless.
          Mismatches raise; ``assume_same_genome`` downgrades an *unreadable*
          config to a warning but never overrides a genuine mismatch.
        * Categorical ids are assigned by :meth:`to_track_records` over the whole
          sheet, never per project. Factorising per project would make
          ``condition_id=0`` mean different things in different projects.
        * ``scaling_group`` defaults to the project name, so tracks are scaled
          within their source batch unless the sheet says otherwise.
        """
        if not entries:
            raise ValueError("No SeqNado projects given")

        names = _unique_project_names(entries)
        single = len(entries) == 1
        genomes: dict[str, str | None] = {}
        unknown_genome: list[str] = []
        records: list[TrackRecord] = []

        for entry, name in zip(entries, names):
            options = {k: v for k, v in entry.items() if k not in {"path", "name"}}
            sheet = cls.from_seqnado_project(
                entry["path"],
                # A single project keeps bare sample names; namespacing only
                # earns its keep when names can actually collide.
                name=None if single else name,
                **options,
            )

            genome = _project_genome(entry["path"])
            if genome is None:
                unknown_genome.append(name)
            else:
                genomes[name] = genome

            for record in sheet.records:
                record.project = name
                if record.scaling_group is None:
                    record.scaling_group = name
                records.append(record)

        _check_genome_consistency(
            genomes, unknown_genome, single=single, assume_same_genome=assume_same_genome
        )

        sheet = cls(records=records)
        sheet._require_unique_track_names()
        return sheet

    def resolve_from_projects(
        self,
        projects: dict[str, str | Path],
        *,
        method: str | None = "deeptools",
        scale: str | None = "unscaled",
        merged: bool | None = False,
    ) -> "TrackSheet":
        """Fill in columns SeqNado already derives, for rows that omit them.

        For every row without a ``bigwig`` path, look up its ``sample_id`` in the
        named SeqNado project and take the track path, the matching BAM, and the
        per-sample annotation (``condition``, ``ip``, ``scaling_group``) from
        there. Values written explicitly in the sheet always win — the sheet is
        an overlay, so it can correct or extend what SeqNado recorded.

        Modifies the sheet in place and returns it.
        """
        if not projects:
            return self

        catalogue = _build_seqnado_catalogue(projects, method=method, scale=scale, merged=merged)
        default_project = next(iter(catalogue)) if len(catalogue) == 1 else None
        namespaced = len(catalogue) > 1

        for record in self.records:
            if record.bigwig is not None:
                if record.track_name is None:
                    record.track_name = record.sample_id
                continue
            _resolve_track_record(
                record,
                catalogue,
                default_project=default_project,
                namespaced=namespaced,
                method=method,
                scale=scale,
            )

        self._fill_track_names()
        return self

    def _fill_track_names(self) -> None:
        for record in self.records:
            if record.track_name is None:
                record.track_name = record.sample_id

    # ------------------------------------------------------------------ #
    #  Training metadata                                                   #
    # ------------------------------------------------------------------ #

    def to_track_records(self) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Return per-track annotation dicts plus the id vocabulary.

        Categorical columns are factorised once across the entire sheet, so an id
        means the same thing for every track regardless of source project. The
        returned vocabulary maps each field to its ordered label list, so ids stay
        decodable later — ``runner.infer_cardinality`` can only recover ``max+1``.

        Tracks with a missing label get ``-1``, which the training code already
        treats as absent.
        """
        vocab: dict[str, list[str]] = {}
        lookup: dict[str, dict[str, int]] = {}
        for column in CATEGORICAL_FIELDS:
            labels = sorted({
                value for record in self.records if (value := getattr(record, column)) is not None
            })
            if labels:
                vocab[column] = labels
            lookup[column] = {label: index for index, label in enumerate(labels)}

        annotations: list[dict[str, Any]] = []
        for record in self.records:
            entry: dict[str, Any] = {
                "track_name": record.track_name,
                "sample_id": record.sample_id,
            }
            for column, field in CATEGORICAL_FIELDS.items():
                value = getattr(record, column)
                entry[field] = lookup[column].get(value, -1) if value is not None else -1
                if value is not None:
                    entry[column] = value
            if record.timepoint_minutes is not None:
                entry["timepoint_minutes"] = float(record.timepoint_minutes)
            for column in ("project", "scaling_group", "consensus_group", "control", "group"):
                value = getattr(record, column)
                if value is not None:
                    entry[column] = value
            annotations.append(entry)

        return annotations, vocab

    def annotations_by_path(self) -> dict[str, dict[str, Any]]:
        """Annotation dicts keyed by resolved BigWig path, for the dataset builder."""
        self.require_resolved()
        annotations, _ = self.to_track_records()
        return {
            str(record.bigwig.resolve()): annotation
            for record, annotation in zip(self.records, annotations)
            if record.bigwig is not None
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _require_unique_track_names(self) -> None:
        seen: dict[str, Path | None] = {}
        for record in self.records:
            name = record.track_name or record.sample_id
            if name in seen:
                raise ValueError(
                    f"Duplicate track name {name!r}:\n"
                    f"  {seen[name]}\n  {record.bigwig}\n"
                    f"Give the source projects distinct names to namespace them."
                )
            seen[name] = record.bigwig


def _project_genome(project_dir: str | Path) -> str | None:
    """Return the reference genome name recorded in a SeqNado project's config.

    ``SeqNadoProject._load_config`` returns None rather than raising when no
    config can be found or parsed, so an unreadable config is reported as unknown
    rather than as a mismatch.
    """
    from regulonado._seqnado import is_multi_project, open_project

    project = open_project(project_dir)
    projects = [sub for _, sub in project.items()] if is_multi_project(project) else [project]

    names = set()
    for sub in projects:
        config = getattr(sub, "config", None)
        genome = getattr(getattr(config, "genome", None), "name", None)
        if genome:
            names.add(str(genome))

    if not names:
        return None
    if len(names) > 1:
        raise GenomeMismatchError(
            f"{project_dir} mixes reference genomes across its assays: "
            f"{', '.join(sorted(names))}"
        )
    return names.pop()


def summarise_projects(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-project counts for reporting before an expensive build."""
    from regulonado._seqnado import is_multi_project, open_project

    summary: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry["path"])
        name = str(entry.get("name") or path.parent.name)
        try:
            project = open_project(path)
        except Exception as exc:
            summary.append({"project": name, "path": str(path), "error": str(exc)})
            continue

        subs = [sub for _, sub in project.items()] if is_multi_project(project) else [project]
        summary.append(
            {
                "project": name,
                "path": str(path),
                "genome": _project_genome(path),
                "samples": sum(len(sub.samples) for sub in subs),
                "bigwigs": sum(len(sub.bigwig_dataframe()) for sub in subs),
            }
        )
    return summary
