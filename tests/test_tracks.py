"""Track sheets and SeqNado project aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml
from regulonado._seqnado import seqnado_available
from regulonado.tracks import (
    PROJECT_SEPARATOR,
    GenomeMismatchError,
    TrackRecord,
    TrackSheet,
    _apply_sample_filters,
    _check_genome_consistency,
    _filter_seqnado_index,
    _resolve_track_record,
    _track_record_from_csv_row,
    _track_records_from_seqnado_index,
    _unique_project_names,
)

requires_seqnado = pytest.mark.skipif(
    not seqnado_available(),
    reason="SeqNado's project API (seqnado.open_project) is an optional dependency",
)

# Columns SeqNado's design sheet may carry; only those a sample actually sets are
# written, because SeqNado's Metadata model rejects the NaN that pandas produces
# for a blank cell.
_DESIGN_COLUMNS = ("sample_id", "r1", "condition", "ip", "scaling_group")

_BOWTIE_SUFFIXES = ("1.bt2", "2.bt2", "3.bt2", "4.bt2", "rev.1.bt2", "rev.2.bt2")


def _write_design(path: Path, samples: list[dict[str, str]]) -> None:
    columns = [c for c in _DESIGN_COLUMNS if any(sample.get(c) for sample in samples)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            writer.writerow({column: sample.get(column, "") for column in columns})


@pytest.fixture
def make_seqnado_project(tmp_path):
    """Build a fake SeqNado output tree and return its ``seqnado_output`` path.

    SeqNado's project API only globs paths and reads the design CSV, so empty
    files are enough. A config is written next to the output directory because
    that is the only place the reference genome is recorded, and aggregation
    refuses to merge projects whose genome it cannot read.
    """

    def _build(
        name: str,
        samples: list[dict[str, str]],
        *,
        genome: str = "hg38",
        assay: str = "ChIP",
        spikein_method: str | None = None,
        write_config: bool = True,
        root: Path | None = None,
    ) -> Path:
        root = root or tmp_path
        index_dir = root / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        for suffix in _BOWTIE_SUFFIXES:
            (index_dir / f"{genome}.{suffix}").touch()

        project = root / name
        output = project / "seqnado_output"
        bigwigs = output / "bigwigs" / "deeptools" / "unscaled"
        aligned = output / "aligned"
        bigwigs.mkdir(parents=True, exist_ok=True)
        aligned.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            (bigwigs / f"{sample['sample_id']}.bigWig").touch()
            (aligned / f"{sample['sample_id']}.bam").touch()

        _write_design(project / "design.csv", samples)

        if write_config:
            (project / f"config_{assay.lower()}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "assay": assay,
                        "project": {"name": name, "date": "2024-01-01"},
                        "genome": {
                            "name": genome,
                            "index": {"prefix": str(index_dir / genome)},
                        },
                        "metadata": "design.csv",
                        "qc": {},
                        "pcr_duplicates": {},
                    }
                )
            )

        if spikein_method:
            resources = output / "resources" / spikein_method
            resources.mkdir(parents=True, exist_ok=True)
            resources.joinpath("normalisation_factors.tsv").write_text(
                "sample\tnorm_factor\n"
                + "".join(f"{sample['sample_id']}\t1.0\n" for sample in samples)
            )

        return output

    return _build


@pytest.fixture
def project_a(make_seqnado_project):
    return make_seqnado_project(
        "expA",
        [
            {"sample_id": "CTCF_DMSO", "r1": "a_1.fq.gz", "condition": "DMSO", "ip": "CTCF"},
            {"sample_id": "CTCF_IAA", "r1": "b_1.fq.gz", "condition": "IAA", "ip": "CTCF"},
            {"sample_id": "input_1", "r1": "c_1.fq.gz", "condition": "DMSO", "ip": "input"},
        ],
        spikein_method="orlando",
    )


@pytest.fixture
def project_b(make_seqnado_project):
    return make_seqnado_project(
        "expB",
        [
            {
                "sample_id": "H3K27ac_DMSO",
                "r1": "d_1.fq.gz",
                "condition": "DMSO",
                "ip": "H3K27ac",
            },
            {"sample_id": "input_1", "r1": "e_1.fq.gz", "condition": "IAA", "ip": "input"},
        ],
    )


def _write_sheet(path: Path, columns: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


# ---------------------------------------------------------------------- #
#  Self-contained sheets                                                   #
# ---------------------------------------------------------------------- #


def test_from_csv_resolves_relative_paths_against_the_sheet(tmp_path):
    (tmp_path / "bigwigs").mkdir()
    sheet_path = _write_sheet(
        tmp_path / "sheets" / "tracks.csv",
        ["sample_id", "bigwig", "condition"],
        [
            {"sample_id": "a", "bigwig": "../bigwigs/a.bigWig", "condition": "DMSO"},
            {"sample_id": "b", "bigwig": "../bigwigs/b.bigWig", "condition": "IAA"},
        ],
    )

    sheet = TrackSheet.from_csv(sheet_path)

    assert len(sheet) == 2
    assert [record.track_name for record in sheet.records] == ["a", "b"]
    assert sheet.records[0].bigwig == (tmp_path / "bigwigs" / "a.bigWig").resolve()
    assert sheet.records[1].bigwig == (tmp_path / "bigwigs" / "b.bigWig").resolve()


def test_from_csv_rejects_a_row_with_no_identifier(tmp_path):
    sheet_path = _write_sheet(
        tmp_path / "tracks.csv",
        ["sample_id", "track_name", "bigwig", "condition"],
        [
            {"sample_id": "a", "bigwig": "a.bigWig", "condition": "DMSO"},
            {"condition": "IAA"},
        ],
    )

    with pytest.raises(ValueError, match="row 2 has none of 'sample_id'"):
        TrackSheet.from_csv(sheet_path)


@pytest.mark.parametrize("condition", ["with space", "with/slash"])
def test_from_csv_rejects_labels_outside_the_seqnado_alphabet(tmp_path, condition):
    sheet_path = _write_sheet(
        tmp_path / "tracks.csv",
        ["sample_id", "bigwig", "condition"],
        [{"sample_id": "a", "bigwig": "a.bigWig", "condition": condition}],
    )

    with pytest.raises(ValueError, match=r"a-zA-Z0-9_-"):
        TrackSheet.from_csv(sheet_path)


# ---------------------------------------------------------------------- #
#  Factorisation                                                           #
# ---------------------------------------------------------------------- #


def test_to_track_records_factorises_over_the_whole_sheet(tmp_path):
    sheet_path = _write_sheet(
        tmp_path / "tracks.csv",
        ["sample_id", "bigwig", "condition", "ip"],
        [
            {"sample_id": "a", "bigwig": "a.bigWig", "condition": "treated", "ip": "CTCF"},
            {"sample_id": "b", "bigwig": "b.bigWig", "condition": "control", "ip": ""},
            {"sample_id": "c", "bigwig": "c.bigWig", "condition": "treated", "ip": "CTCF"},
        ],
    )

    annotations, vocab = TrackSheet.from_csv(sheet_path).to_track_records()

    # Rows sharing a label share an id.
    assert annotations[0]["condition_id"] == annotations[2]["condition_id"]
    assert annotations[1]["condition_id"] != annotations[0]["condition_id"]

    # A missing label is -1, which the training code already treats as absent.
    assert annotations[1]["target_id"] == -1
    assert "ip" not in annotations[1]

    # Labels are ordered, so ids stay decodable.
    assert vocab["condition"] == ["control", "treated"]
    assert annotations[1]["condition_id"] == 0
    assert annotations[0]["condition_id"] == 1

    # Fields no row used are omitted entirely.
    assert "source" not in vocab
    assert "assay" not in vocab
    assert all(annotation["source_id"] == -1 for annotation in annotations)


# ---------------------------------------------------------------------- #
#  CSV round-trip                                                          #
# ---------------------------------------------------------------------- #


def test_csv_round_trip_preserves_annotation_and_drops_empty_columns(tmp_path):
    sheet_path = _write_sheet(
        tmp_path / "tracks.csv",
        ["sample_id", "bigwig", "condition", "source", "timepoint_minutes"],
        [
            {
                "sample_id": "a",
                "bigwig": "a.bigWig",
                "condition": "DMSO",
                "source": "K562",
                "timepoint_minutes": "0",
            },
            {
                "sample_id": "b",
                "bigwig": "b.bigWig",
                "condition": "IAA",
                "source": "K562",
                "timepoint_minutes": "120",
            },
        ],
    )
    original = TrackSheet.from_csv(sheet_path)

    written = original.to_csv(tmp_path / "out" / "tracks.csv")
    reread = TrackSheet.from_csv(written)

    assert [record.model_dump() for record in reread.records] == [
        record.model_dump() for record in original.records
    ]

    with written.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert "condition" in header and "source" in header
    # Columns empty for every row never reach the file.
    for column in ("consensus_group", "control", "group", "project", "bam"):
        assert column not in header


def test_require_resolved_names_the_unresolved_sample(tmp_path):
    sheet_path = _write_sheet(
        tmp_path / "tracks.csv",
        ["sample_id", "bigwig"],
        [
            {"sample_id": "resolved_one", "bigwig": "a.bigWig"},
            {"sample_id": "dangling_sample"},
        ],
    )
    sheet = TrackSheet.from_csv(sheet_path)

    with pytest.raises(ValueError, match="dangling_sample"):
        sheet.require_resolved()


# ---------------------------------------------------------------------- #
#  Extracted helpers (independently testable without a SeqNado install)   #
# ---------------------------------------------------------------------- #


def test_track_record_from_csv_row_derives_sample_id_from_bigwig_stem(tmp_path):
    record = _track_record_from_csv_row(
        {"bigwig": "signals/a.bigWig", "condition": "DMSO"},
        index=0,
        base=tmp_path,
        path=tmp_path / "sheet.csv",
    )

    assert record.sample_id == "a"
    assert record.track_name == "a"
    assert record.bigwig == (tmp_path / "signals" / "a.bigWig").resolve()
    assert record.condition == "DMSO"


def test_track_record_from_csv_row_raises_without_any_identifier(tmp_path):
    with pytest.raises(ValueError, match="row 1 has none of"):
        _track_record_from_csv_row(
            {"condition": "DMSO"}, index=0, base=tmp_path, path=tmp_path / "sheet.csv"
        )


def test_track_record_from_csv_row_cleans_na_like_values(tmp_path):
    record = _track_record_from_csv_row(
        {"sample_id": "a", "bigwig": "a.bigWig", "condition": "NA"},
        index=0,
        base=tmp_path,
        path=tmp_path / "sheet.csv",
    )

    assert record.condition is None


def test_unique_project_names_defaults_to_the_output_dirs_parent(tmp_path):
    entries = [
        {"path": tmp_path / "expA" / "seqnado_output"},
        {"name": "custom", "path": tmp_path / "expB" / "seqnado_output"},
    ]

    assert _unique_project_names(entries) == ["expA", "custom"]


def test_unique_project_names_raises_on_duplicates():
    entries = [{"name": "expA", "path": "x"}, {"name": "expA", "path": "y"}]

    with pytest.raises(ValueError, match="Project names must be unique"):
        _unique_project_names(entries)


def test_check_genome_consistency_allows_matching_genomes():
    _check_genome_consistency(
        {"a": "hg38", "b": "hg38"}, [], single=False, assume_same_genome=False
    )


def test_check_genome_consistency_raises_on_mismatch():
    with pytest.raises(GenomeMismatchError, match="different reference genomes"):
        _check_genome_consistency(
            {"a": "hg38", "b": "mm10"}, [], single=False, assume_same_genome=False
        )


def test_check_genome_consistency_raises_on_unknown_genome_unless_assumed():
    with pytest.raises(GenomeMismatchError, match="Could not read the reference genome"):
        _check_genome_consistency({}, ["a"], single=False, assume_same_genome=False)

    # No raise once the caller opts in.
    _check_genome_consistency({}, ["a"], single=False, assume_same_genome=True)


def test_check_genome_consistency_ignores_unknown_genome_for_a_single_project():
    _check_genome_consistency({}, ["a"], single=True, assume_same_genome=False)


def _catalogue_entry(**overrides):
    entry = {
        "bigwig": Path("/data/a.bigWig"),
        "bam": Path("/data/a.bam"),
        "assay": "ChIP",
        "method": "deeptools",
        "scale": "unscaled",
        "metadata": {"condition": "DMSO", "antibody": "CTCF", "group": None},
    }
    entry.update(overrides)
    return entry


def test_resolve_track_record_fills_blanks_from_the_catalogue():
    record = TrackRecord(sample_id="a")
    catalogue = {"expA": {"a": _catalogue_entry()}}

    _resolve_track_record(
        record,
        catalogue,
        default_project="expA",
        namespaced=False,
        method="deeptools",
        scale="unscaled",
    )

    assert record.project == "expA"
    assert record.bigwig == Path("/data/a.bigWig")
    assert record.bam == Path("/data/a.bam")
    assert record.condition == "DMSO"
    assert record.ip == "CTCF"
    assert record.track_name == "a"


def test_resolve_track_record_keeps_explicit_sheet_values():
    record = TrackRecord(sample_id="a", condition="corrected")
    catalogue = {"expA": {"a": _catalogue_entry()}}

    _resolve_track_record(
        record, catalogue, default_project="expA", namespaced=False, method=None, scale=None
    )

    assert record.condition == "corrected"


def test_resolve_track_record_namespaces_track_name_with_several_projects():
    record = TrackRecord(sample_id="a", project="expA")
    catalogue = {"expA": {"a": _catalogue_entry()}, "expB": {}}

    _resolve_track_record(
        record, catalogue, default_project=None, namespaced=True, method=None, scale=None
    )

    assert record.track_name == f"expA{PROJECT_SEPARATOR}a"


def test_resolve_track_record_raises_without_project_when_several_are_configured():
    record = TrackRecord(sample_id="a")
    catalogue = {"expA": {}, "expB": {}}

    with pytest.raises(ValueError, match="has no 'bigwig' path and no 'project'"):
        _resolve_track_record(
            record, catalogue, default_project=None, namespaced=True, method=None, scale=None
        )


def test_resolve_track_record_raises_for_an_unconfigured_project():
    record = TrackRecord(sample_id="a", project="expC")
    catalogue = {"expA": {}}

    with pytest.raises(ValueError, match="not configured"):
        _resolve_track_record(
            record, catalogue, default_project=None, namespaced=False, method=None, scale=None
        )


def test_resolve_track_record_raises_for_a_sample_missing_from_its_project():
    record = TrackRecord(sample_id="missing", project="expA")
    catalogue = {"expA": {"a": _catalogue_entry()}}

    with pytest.raises(ValueError, match="not found in project"):
        _resolve_track_record(
            record,
            catalogue,
            default_project=None,
            namespaced=False,
            method="deeptools",
            scale="unscaled",
        )


def test_filter_seqnado_index_filters_by_method_scale_and_merged():
    index = pd.DataFrame(
        {
            "sample": ["a", "b", "c"],
            "method": ["deeptools", "deeptools", "bamCoverage"],
            "scale": ["unscaled", "spikein", "unscaled"],
            "merged": [False, False, True],
            "strand": [None, None, None],
        }
    )

    filtered = _filter_seqnado_index(
        index, method="deeptools", scale="unscaled", merged=False, keep_stranded=False
    )

    assert list(filtered["sample"]) == ["a"]


def test_filter_seqnado_index_drops_stranded_tracks_unless_kept():
    index = pd.DataFrame(
        {
            "sample": ["a", "b"],
            "method": ["deeptools", "deeptools"],
            "scale": ["unscaled", "unscaled"],
            "merged": [False, False],
            "strand": [None, "+"],
        }
    )

    dropped = _filter_seqnado_index(
        index, method=None, scale=None, merged=None, keep_stranded=False
    )
    assert list(dropped["sample"]) == ["a"]

    kept = _filter_seqnado_index(index, method=None, scale=None, merged=None, keep_stranded=True)
    assert list(kept["sample"]) == ["a", "b"]


class _FakeSeqNadoProject:
    """Duck-typed stand-in for SeqNadoProject; enough for the ingestion helpers."""

    def __init__(self, *, allowed_paths=None, metadata=None):
        self._allowed = allowed_paths or []
        self._metadata = metadata or {}

    def bigwigs(self, **filters):
        return self._allowed

    def metadata_for(self, sample):
        return self._metadata.get(sample)


def test_apply_sample_filters_restricts_to_seqnados_own_selection(tmp_path):
    kept = tmp_path / "kept.bigWig"
    dropped = tmp_path / "dropped.bigWig"
    kept.touch()
    dropped.touch()
    index = pd.DataFrame({"path": [str(kept), str(dropped)]})
    project = _FakeSeqNadoProject(allowed_paths=[kept])

    filtered = _apply_sample_filters(index, project, {"condition": "DMSO"})

    assert list(filtered["path"]) == [str(kept)]


def test_apply_sample_filters_is_a_no_op_without_filters():
    index = pd.DataFrame({"path": ["a", "b"]})

    filtered = _apply_sample_filters(index, project=None, filters={})

    assert filtered is index


def test_track_records_from_seqnado_index_maps_antibody_back_to_ip():
    index = pd.DataFrame(
        {
            "sample": ["CTCF_DMSO"],
            "path": ["/out/CTCF_DMSO.bigWig"],
            "method": ["deeptools"],
            "scale": ["unscaled"],
        }
    )
    project = _FakeSeqNadoProject(
        metadata={"CTCF_DMSO": {"antibody": "CTCF", "condition": "DMSO"}}
    )

    records = _track_records_from_seqnado_index(
        index, project, bams={}, project_name=None, assay="ChIP"
    )

    assert len(records) == 1
    record = records[0]
    assert record.track_name == "CTCF_DMSO"
    assert record.ip == "CTCF"
    assert record.condition == "DMSO"
    assert record.assay == "ChIP"


def test_track_records_from_seqnado_index_namespaces_track_name_with_project():
    index = pd.DataFrame(
        {"sample": ["a"], "path": ["/out/a.bigWig"], "method": ["deeptools"], "scale": ["unscaled"]}
    )
    project = _FakeSeqNadoProject()

    records = _track_records_from_seqnado_index(
        index, project, bams={}, project_name="expA", assay="ChIP"
    )

    assert records[0].track_name == f"expA{PROJECT_SEPARATOR}a"
    assert records[0].project == "expA"


# ---------------------------------------------------------------------- #
#  SeqNado projects                                                        #
# ---------------------------------------------------------------------- #


@requires_seqnado
def test_from_seqnado_project_reads_design_annotation(project_a):
    sheet = TrackSheet.from_seqnado_project(project_a)

    assert len(sheet) == 3
    by_name = {record.track_name: record for record in sheet.records}
    assert set(by_name) == {"CTCF_DMSO", "CTCF_IAA", "input_1"}

    assert by_name["CTCF_DMSO"].condition == "DMSO"
    assert by_name["CTCF_IAA"].condition == "IAA"
    # SeqNado exposes the design's `ip` column as `antibody`; the sheet keeps the
    # design-sheet name.
    assert by_name["CTCF_DMSO"].ip == "CTCF"
    assert by_name["input_1"].ip == "input"

    assert by_name["CTCF_DMSO"].bigwig.name == "CTCF_DMSO.bigWig"
    assert by_name["CTCF_DMSO"].bam is not None
    assert by_name["CTCF_DMSO"].bam.name == "CTCF_DMSO.bam"


@requires_seqnado
def test_single_project_track_names_are_not_namespaced(project_a):
    sheet = TrackSheet.from_seqnado_projects([{"name": "expA", "path": project_a}])

    assert [record.track_name for record in sheet.records] == [
        "CTCF_DMSO",
        "CTCF_IAA",
        "input_1",
    ]
    assert all(PROJECT_SEPARATOR not in record.track_name for record in sheet.records)
    assert all(record.project == "expA" for record in sheet.records)


@requires_seqnado
def test_aggregating_projects_namespaces_colliding_sample_names(project_a, project_b):
    sheet = TrackSheet.from_seqnado_projects(
        [{"name": "expA", "path": project_a}, {"name": "expB", "path": project_b}]
    )

    names = [record.track_name for record in sheet.records]
    assert "expA__input_1" in names
    assert "expB__input_1" in names
    assert len(names) == len(set(names)) == 5


@requires_seqnado
def test_condition_ids_are_shared_across_aggregated_projects(project_a, project_b):
    """The property the aggregation design exists to guarantee.

    ``DMSO`` appears in both projects; factorising per project would give it two
    different ids and silently corrupt the merged dataset.
    """
    sheet = TrackSheet.from_seqnado_projects(
        [{"name": "expA", "path": project_a}, {"name": "expB", "path": project_b}]
    )
    annotations, vocab = sheet.to_track_records()
    by_name = {annotation["track_name"]: annotation for annotation in annotations}

    assert by_name["expA__CTCF_DMSO"]["condition"] == "DMSO"
    assert by_name["expB__H3K27ac_DMSO"]["condition"] == "DMSO"
    assert (
        by_name["expA__CTCF_DMSO"]["condition_id"]
        == by_name["expB__H3K27ac_DMSO"]["condition_id"]
    )
    assert by_name["expA__input_1"]["condition_id"] != by_name["expB__input_1"]["condition_id"]

    # One vocabulary over the union, not one per project.
    assert vocab["condition"] == ["DMSO", "IAA"]
    assert vocab["ip"] == ["CTCF", "H3K27ac", "input"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SeqNado's Metadata model defaults scaling_group to 'default', so "
        "metadata_for() never reports it as absent and the project-name fallback "
        "in TrackSheet.from_seqnado_projects (tracks.py:489) is unreachable."
    ),
)
@requires_seqnado
def test_scaling_group_defaults_to_the_project_name(project_a, project_b):
    sheet = TrackSheet.from_seqnado_projects(
        [{"name": "expA", "path": project_a}, {"name": "expB", "path": project_b}]
    )

    by_name = {record.track_name: record for record in sheet.records}
    assert by_name["expA__input_1"].scaling_group == "expA"
    assert by_name["expB__input_1"].scaling_group == "expB"


@requires_seqnado
def test_duplicate_project_names_raise(project_a, project_b):
    with pytest.raises(ValueError, match="Project names must be unique"):
        TrackSheet.from_seqnado_projects(
            [{"name": "expA", "path": project_a}, {"name": "expA", "path": project_b}]
        )


# ---------------------------------------------------------------------- #
#  Overlay sheets                                                          #
# ---------------------------------------------------------------------- #


@requires_seqnado
def test_overlay_sheet_resolves_paths_and_metadata_from_projects(
    tmp_path, project_a, project_b
):
    sheet_path = _write_sheet(
        tmp_path / "overlay.csv",
        ["project", "sample_id", "source", "timepoint_minutes"],
        [
            {
                "project": "expA",
                "sample_id": "CTCF_DMSO",
                "source": "K562",
                "timepoint_minutes": "0",
            },
            {
                "project": "expB",
                "sample_id": "input_1",
                "source": "HeLa",
                "timepoint_minutes": "120",
            },
        ],
    )

    sheet = TrackSheet.from_csv(
        sheet_path, projects={"expA": project_a, "expB": project_b}
    )
    by_name = {record.track_name: record for record in sheet.records}

    assert set(by_name) == {"expA__CTCF_DMSO", "expB__input_1"}
    sheet.require_resolved()

    ctcf = by_name["expA__CTCF_DMSO"]
    assert ctcf.bigwig is not None and ctcf.bigwig.name == "CTCF_DMSO.bigWig"
    assert ctcf.bam is not None and ctcf.bam.name == "CTCF_DMSO.bam"
    # Filled in from the SeqNado design, which the sheet never mentions.
    assert ctcf.condition == "DMSO"
    assert ctcf.ip == "CTCF"
    # Carried straight from the sheet: SeqNado has no equivalent columns.
    assert ctcf.source == "K562"
    assert ctcf.timepoint_minutes == 0.0

    assert by_name["expB__input_1"].condition == "IAA"
    assert by_name["expB__input_1"].ip == "input"


@requires_seqnado
def test_explicit_sheet_values_win_over_seqnado_metadata(tmp_path, project_a, project_b):
    sheet_path = _write_sheet(
        tmp_path / "overlay.csv",
        ["project", "sample_id", "condition", "source"],
        [
            # design.csv says DMSO; the sheet is an overlay, so it may correct it.
            {
                "project": "expA",
                "sample_id": "CTCF_DMSO",
                "condition": "corrected",
                "source": "K562",
            },
            {"project": "expA", "sample_id": "CTCF_IAA", "source": "K562"},
        ],
    )

    sheet = TrackSheet.from_csv(
        sheet_path, projects={"expA": project_a, "expB": project_b}
    )
    by_name = {record.track_name: record for record in sheet.records}

    assert by_name["expA__CTCF_DMSO"].condition == "corrected"
    # The row that said nothing still gets SeqNado's value.
    assert by_name["expA__CTCF_IAA"].condition == "IAA"


@requires_seqnado
def test_overlay_sheet_naming_an_unknown_sample_raises(tmp_path, project_a):
    sheet_path = _write_sheet(
        tmp_path / "overlay.csv",
        ["project", "sample_id", "source"],
        [{"project": "expA", "sample_id": "not_a_sample", "source": "K562"}],
    )

    with pytest.raises(ValueError, match="not_a_sample"):
        TrackSheet.from_csv(sheet_path, projects={"expA": project_a})
