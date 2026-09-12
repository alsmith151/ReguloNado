"""The pydantic config models and the workflow's JSON Schema must not drift apart."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from regulonado.config.models import (
    DatasetConfig,
    InputsConfig,
    RegulonadoConfig,
    SeqNadoProjectRef,
    TrainConfig,
    TrainPhase,
    TrainRun,
)

SCHEMA = (
    Path(__file__).parents[1]
    / "python"
    / "regulonado"
    / "workflow"
    / "schemas"
    / "config.schema.yaml"
)


def _validate_against_schema(data: dict[str, Any]) -> None:
    """Validate exactly as the Snakefile does at workflow start-up."""
    snakemake_utils = pytest.importorskip(
        "snakemake.utils", reason="Snakemake is an optional workflow dependency"
    )
    # validate() fills defaults in place; work on a copy so callers can reuse theirs.
    snakemake_utils.validate(copy.deepcopy(data), str(SCHEMA))


def _train() -> TrainConfig:
    return TrainConfig(
        phases=[TrainPhase(name="head", preset="head_only")],
        runs=[TrainRun(name="run_a", seed=0, pretrained_model="model/a")],
    )


def _config(**inputs: Any) -> RegulonadoConfig:
    return RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(intervals="intervals.bed", fasta="genome.fa", **inputs),
        train=_train(),
    )


# ---------------------------------------------------------------------- #
#  Schema round-trip                                                       #
# ---------------------------------------------------------------------- #


def test_schema_file_is_where_the_workflow_expects_it():
    assert SCHEMA.exists()


def test_bigwig_dir_config_validates_against_the_schema():
    config = _config(bigwig_dir="bigwigs")
    _validate_against_schema(config.to_dict())


def test_track_sheet_config_validates_against_the_schema():
    config = _config(track_sheet="tracks.csv")
    _validate_against_schema(config.to_dict())


def test_seqnado_projects_config_validates_against_the_schema():
    config = _config(
        seqnado_projects=[
            SeqNadoProjectRef(name="expA", path="expA/seqnado_output"),
            SeqNadoProjectRef(
                name="expB", path="expB/seqnado_output", method="bamnado", scale="csaw"
            ),
        ]
    )
    data = config.to_dict()

    assert [entry["name"] for entry in data["inputs"]["seqnado_projects"]] == ["expA", "expB"]
    _validate_against_schema(data)


def test_full_config_with_every_optional_key_validates():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(
            intervals="intervals.bed",
            fasta="genome.fa",
            bigwig_dir="bigwigs",
            bam_dir="bams",
            track_sheet="tracks.csv",
            seqnado_projects=[SeqNadoProjectRef(name="expA", path="expA/seqnado_output")],
        ),
        scaling={
            "method": "bamnado",
            "bamnado_method": "spike-in",
            "bamnado_exogenous_prefix": "dm6_",
        },
        train=TrainConfig(
            common={"trainer.max_epochs": 3},
            phases=[
                TrainPhase(name="head", preset="head_only"),
                TrainPhase(name="deep", preset="deep_finetune", settings={"seed": 1}),
            ],
            runs=[
                TrainRun(
                    name="run_a", seed=0, pretrained_model="model/a", settings={"lr": 1e-4}
                )
            ],
        ),
    )

    _validate_against_schema(config.to_dict())


# ---------------------------------------------------------------------- #
#  to_dict()                                                               #
# ---------------------------------------------------------------------- #


def test_to_dict_omits_unset_optional_keys():
    """``additionalProperties: false`` means a null for an unchosen option fails."""
    data = _config(bigwig_dir="bigwigs").to_dict()

    for key in ("track_sheet", "bam_dir", "seqnado_projects"):
        assert key not in data["inputs"]
    for key in ("bamnado_method", "bamnado_exogenous_prefix", "seqnado_project"):
        assert key not in data["scaling"]
    assert "common" not in data["train"]
    assert "settings" not in data["train"]["phases"][0]
    assert "settings" not in data["train"]["runs"][0]

    def _no_nulls(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert value is not None, f"{key} serialised as null"
                _no_nulls(value)
        elif isinstance(node, list):
            for value in node:
                _no_nulls(value)

    _no_nulls(data)


# ---------------------------------------------------------------------- #
#  Validation                                                              #
# ---------------------------------------------------------------------- #


def test_inputs_needs_a_track_source():
    with pytest.raises(
        ValueError, match="'bigwig_dir', 'track_sheet' or 'seqnado_projects'"
    ):
        InputsConfig(intervals="intervals.bed", fasta="genome.fa")


def test_shift_max_bp_must_be_a_whole_number_of_bins():
    with pytest.raises(ValueError, match="must be a multiple of"):
        DatasetConfig(bin_size=32, shift_max_bp=48)

    assert DatasetConfig(bin_size=32, shift_max_bp=64).shift_max_bp == 64


def test_duplicate_phase_names_raise():
    with pytest.raises(ValueError, match=r"train\.phases names must be unique.*head"):
        TrainConfig(
            phases=[
                TrainPhase(name="head", preset="head_only"),
                TrainPhase(name="head", preset="deep_finetune"),
            ],
            runs=[TrainRun(name="run_a", seed=0, pretrained_model="model/a")],
        )


def test_duplicate_run_names_raise():
    with pytest.raises(ValueError, match=r"train\.runs names must be unique.*run_a"):
        TrainConfig(
            phases=[TrainPhase(name="head", preset="head_only")],
            runs=[
                TrainRun(name="run_a", seed=0, pretrained_model="model/a"),
                TrainRun(name="run_a", seed=1, pretrained_model="model/b"),
            ],
        )


def test_seqnado_scaling_across_several_projects_needs_one_named():
    projects = [
        SeqNadoProjectRef(name="expA", path="expA/seqnado_output"),
        SeqNadoProjectRef(name="expB", path="expB/seqnado_output"),
    ]

    with pytest.raises(ValueError, match="only comparable within that project"):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(
                intervals="intervals.bed", fasta="genome.fa", seqnado_projects=projects
            ),
            scaling={"method": "seqnado"},
            train=_train(),
        )


def test_seqnado_scaling_with_a_named_project_is_accepted():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(
            intervals="intervals.bed",
            fasta="genome.fa",
            seqnado_projects=[
                SeqNadoProjectRef(name="expA", path="expA/seqnado_output"),
                SeqNadoProjectRef(name="expB", path="expB/seqnado_output"),
            ],
        ),
        scaling={"method": "seqnado", "seqnado_project": "expA/seqnado_output"},
        train=_train(),
    )

    _validate_against_schema(config.to_dict())


def test_seqnado_scaling_with_exactly_one_project_is_accepted():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(
            intervals="intervals.bed",
            fasta="genome.fa",
            seqnado_projects=[SeqNadoProjectRef(name="expA", path="expA/seqnado_output")],
        ),
        scaling={"method": "seqnado"},
        train=_train(),
    )

    assert config.scaling.method == "seqnado"
    _validate_against_schema(config.to_dict())


def test_bamnado_scaling_requires_a_bam_dir():
    with pytest.raises(ValueError, match="inputs.bam_dir is required"):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(
                intervals="intervals.bed", fasta="genome.fa", bigwig_dir="bigwigs"
            ),
            scaling={"method": "bamnado"},
            train=_train(),
        )


def test_bamnado_scaling_with_a_bam_dir_is_accepted():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(
            intervals="intervals.bed",
            fasta="genome.fa",
            bigwig_dir="bigwigs",
            bam_dir="bams",
        ),
        scaling={"method": "bamnado"},
        train=_train(),
    )

    _validate_against_schema(config.to_dict())
