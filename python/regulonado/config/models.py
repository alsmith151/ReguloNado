"""Pydantic mirror of the workflow's JSON Schema.

``workflow/schemas/config.schema.yaml`` remains the runtime contract enforced by
Snakemake; these models are what the interactive generator builds and serialises.
The two must stay in step — ``tests/test_config_models.py`` validates a generated
config against the schema to make sure they do.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PHASE_PRESETS = ("head_only", "unfreeze_output", "deep_finetune", "peak_finetune")
SCALING_METHODS = ("tmm", "original", "bamnado", "seqnado")
BAMNADO_METHODS = ("tmm", "csaw-background", "cpm", "median-of-ratios", "spike-in")


def _validate_name(value: str) -> str:
    if not NAME_PATTERN.match(value):
        raise ValueError(
            f"{value!r} must start with a letter or digit and contain only "
            f"letters, digits, '.', '_' or '-'"
        )
    return value


class SeqNadoProjectRef(BaseModel):
    """A SeqNado output directory contributing tracks to this dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Namespace label; prefixes track names when aggregating.")
    path: str = Field(description="Path to the SeqNado output directory (seqnado_output/).")
    method: str | None = Field(default="deeptools", description="Pileup method to take.")
    scale: str | None = Field(default="unscaled", description="Scaling variant to take.")

    _check_name = field_validator("name")(staticmethod(_validate_name))


class InputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervals: str
    fasta: str
    bigwig_dir: str | None = None
    bam_dir: str | None = Field(
        default=None,
        description="One BAM per track, named <track-stem>.bam. Needed for scaling 'bamnado'.",
    )
    track_sheet: str | None = Field(
        default=None,
        description="CSV mapping tracks to annotation; supplies the ordered track list.",
    )
    seqnado_projects: list[SeqNadoProjectRef] = Field(
        default_factory=list,
        description="SeqNado projects to draw tracks from, in order.",
    )

    @model_validator(mode="after")
    def _require_a_track_source(self) -> "InputsConfig":
        if not (self.bigwig_dir or self.track_sheet or self.seqnado_projects):
            raise ValueError(
                "inputs needs one of 'bigwig_dir', 'track_sheet' or 'seqnado_projects'"
            )
        return self


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_length: int = Field(default=524_288, ge=1)
    bin_size: int = Field(default=32, ge=1)
    n_pred_bins: int = Field(default=6_144, ge=1)
    shift_max_bp: int = Field(default=64, ge=0)
    extract_threads: int = Field(default=32, ge=1)
    arrow_write_threads: int = Field(default=4, ge=1)
    arrow_batch_size: int = Field(default=512, ge=1)
    compression: str = "lz4"
    stage_to_scratch: bool = True
    drop_missing: bool = True
    dedupe_tracks: Literal["none", "identity", "content"] = "content"

    @model_validator(mode="after")
    def _shift_is_whole_bins(self) -> "BuildConfig":
        if self.shift_max_bp % self.bin_size:
            raise ValueError(
                f"shift_max_bp ({self.shift_max_bp}) must be a multiple of "
                f"bin_size ({self.bin_size})"
            )
        return self


class RecompressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    zstd_level: int = 3
    max_batch_size: int = Field(default=4, ge=1)
    workers: int = Field(default=8, ge=1)


class ScalingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["tmm", "original", "bamnado", "seqnado"] = "tmm"
    bamnado_method: Literal[
        "tmm", "csaw-background", "cpm", "median-of-ratios", "spike-in"
    ] | None = None
    bamnado_exogenous_prefix: str | None = None
    seqnado_project: str | None = Field(
        default=None,
        description="SeqNado output dir whose normalisation factors to reuse (method 'seqnado').",
    )
    seqnado_spikein_method: str | None = Field(
        default=None,
        description="Which resources/<method>/normalisation_factors.tsv to read.",
    )


class TrainPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    preset: Literal["head_only", "unfreeze_output", "deep_finetune", "peak_finetune"]
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))


class TrainRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = Field(ge=0)
    pretrained_model: str
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nproc_per_node: int = Field(default=1, ge=1)
    common: dict[str, Any] = Field(default_factory=dict)
    phases: list[TrainPhase] = Field(min_length=1)
    runs: list[TrainRun] = Field(min_length=1)

    @model_validator(mode="after")
    def _names_are_unique(self) -> "TrainConfig":
        for label, items in (("phases", self.phases), ("runs", self.runs)):
            names = [item.name for item in items]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"train.{label} names must be unique; repeated: {', '.join(duplicates)}"
                )
        return self


class DesignTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    target: str
    group_by: str = "source"
    method: Literal["ism", "adalead"] = "ism"
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))


class DesignConfig(BaseModel):
    """Optional synthetic-enhancer-design stage: mutate candidates for cell-type specificity."""

    model_config = ConfigDict(extra="forbid")

    candidates: str = Field(min_length=1)
    shards: int = Field(default=1, ge=1)
    holdout_run: str | None = None
    design_runs: list[str] | None = None
    checkpoint_dirs: list[str] | None = None
    holdout_checkpoint: str | None = None
    common: dict[str, Any] = Field(default_factory=dict)
    targets: list[DesignTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def _target_names_are_unique(self) -> "DesignConfig":
        names = [target.name for target in self.targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"design.targets names must be unique; repeated: {', '.join(duplicates)}"
            )
        return self


class AttributionTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    track: str
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))


class AttributionConfig(BaseModel):
    """Optional ISM-attribution stage: locate each candidate's high-attribution core.

    Upstream of ``design``: point ``design.candidates`` at this stage's
    ``core_regions.bed`` to optimise only the core rather than the whole candidate.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: str = Field(min_length=1)
    shards: int = Field(default=1, ge=1)
    runs: list[str] | None = None
    checkpoint_dirs: list[str] | None = None
    common: dict[str, Any] = Field(default_factory=dict)
    targets: list[AttributionTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def _target_names_are_unique(self) -> "AttributionConfig":
        names = [target.name for target in self.targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"attribution.targets names must be unique; repeated: {', '.join(duplicates)}"
            )
        return self


class RegulonadoConfig(BaseModel):
    """Top-level workflow config."""

    model_config = ConfigDict(extra="forbid")

    results_dir: str = Field(min_length=1)
    inputs: InputsConfig
    build: BuildConfig = Field(default_factory=BuildConfig)
    recompress: RecompressConfig = Field(default_factory=RecompressConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    train: TrainConfig
    design: DesignConfig | None = None
    attribution: AttributionConfig | None = None

    @model_validator(mode="after")
    def _design_runs_are_known(self) -> "RegulonadoConfig":
        if self.design is None:
            return self
        run_names = {run.name for run in self.train.runs}
        for label, names in (
            ("holdout_run", [self.design.holdout_run] if self.design.holdout_run else []),
            ("design_runs", self.design.design_runs or []),
        ):
            unknown = sorted(name for name in names if name not in run_names)
            if unknown:
                raise ValueError(
                    f"design.{label} names train.runs entries that don't exist: "
                    f"{', '.join(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def _attribution_runs_are_known(self) -> "RegulonadoConfig":
        if self.attribution is None or not self.attribution.runs:
            return self
        run_names = {run.name for run in self.train.runs}
        unknown = sorted(name for name in self.attribution.runs if name not in run_names)
        if unknown:
            raise ValueError(
                f"attribution.runs names train.runs entries that don't exist: "
                f"{', '.join(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def _scaling_inputs_are_present(self) -> "RegulonadoConfig":
        if self.scaling.method == "bamnado" and not self.inputs.bam_dir:
            raise ValueError("inputs.bam_dir is required when scaling.method is 'bamnado'")

        if self.scaling.method == "seqnado":
            # Normalisation factors are only comparable within one SeqNado
            # project. Naming a single project does not rescue an aggregated
            # dataset either: the tracks from every other project have no entry
            # in that project's table, so scaling would fail once it ran. Reject
            # it here, where the message can name the fix.
            if len(self.inputs.seqnado_projects) > 1 and not self.scaling.seqnado_project:
                raise ValueError(
                    "scaling.method 'seqnado' reuses one project's normalisation "
                    "factors, which are only comparable within that project. With "
                    f"{len(self.inputs.seqnado_projects)} projects configured, use "
                    "scaling.method 'tmm' — it is derived from the merged dataset and "
                    "so is comparable across all of them."
                )
            if not (self.scaling.seqnado_project or self.inputs.seqnado_projects):
                raise ValueError(
                    "scaling.method 'seqnado' needs scaling.seqnado_project or "
                    "inputs.seqnado_projects"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RegulonadoConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise in the same section order as the shipped example config."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        )
        return path

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for YAML/JSON, with unset optional keys omitted.

        The workflow schema sets ``additionalProperties: false``, so emitting
        ``null`` for options that were never chosen would fail validation.
        """
        data = self.model_dump(mode="json", exclude_none=True)
        if not data.get("inputs", {}).get("seqnado_projects"):
            data.get("inputs", {}).pop("seqnado_projects", None)
        for section in ("train",):
            if not data.get(section, {}).get("common"):
                data.get(section, {}).pop("common", None)
        for item in data.get("train", {}).get("phases", []) + data.get("train", {}).get(
            "runs", []
        ):
            if not item.get("settings"):
                item.pop("settings", None)
        return data
