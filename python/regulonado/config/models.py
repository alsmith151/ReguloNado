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
SCALING_METHODS = ("tmm", "original", "bamnado", "seqnado", "anchor")
BAMNADO_METHODS = ("tmm", "csaw-background", "cpm", "median-of-ratios", "spike-in")
QC_CHECKS = ("sparsity", "interval_signal", "replicate_concordance", "anchor")


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
    exclude_tracks: list[str] = Field(
        default_factory=list,
        description=(
            "track_name(s) to force-exclude at 'tracks assemble', honoured whether or "
            "not QC is enabled."
        ),
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

    method: Literal["tmm", "original", "bamnado", "seqnado", "anchor"] = "tmm"
    anchor_regions: str | None = None
    background_regions: str | None = None
    heldout_regions: str | None = None
    window_stat_bp: int = Field(default=1000, ge=1)
    window_stat_bp_by_assay: dict[str, int] | None = None
    background_sample: int | None = Field(default=5000, ge=1)
    bamnado_method: (
        Literal["tmm", "csaw-background", "cpm", "median-of-ratios", "spike-in"] | None
    ) = None
    bamnado_exogenous_prefix: str | None = None
    seqnado_project: str | None = Field(
        default=None,
        description="SeqNado output dir whose normalisation factors to reuse (method 'seqnado').",
    )
    seqnado_spikein_method: str | None = Field(
        default=None,
        description="Which resources/<method>/normalisation_factors.tsv to read.",
    )

    @model_validator(mode="after")
    def _anchor_regions_match_method(self) -> "ScalingConfig":
        fields = ("anchor_regions", "background_regions", "heldout_regions")
        if self.method == "anchor" and (not self.anchor_regions or not self.background_regions):
            raise ValueError(
                "scaling.anchor_regions and scaling.background_regions are required "
                "for method 'anchor'"
            )
        if self.method != "anchor" and any(getattr(self, field) is not None for field in fields):
            raise ValueError("scaling anchor region options are only valid when method is 'anchor'")
        return self


class QCConfig(BaseModel):
    """Opt-in track QC, run before the Arrow build. Empty/absent ``checks`` means no QC."""

    model_config = ConfigDict(extra="forbid")

    checks: list[Literal["sparsity", "interval_signal", "replicate_concordance", "anchor"]] = Field(
        default_factory=list
    )
    rules: dict[str, dict[Literal["min", "max"], float]] = Field(default_factory=dict)
    drop_degenerate: bool = False
    sample_windows: int | None = Field(
        default=None, description="Sample this many BED rows for interval-scan checks."
    )
    max_workers: int = Field(default=16, ge=1)


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
    """Optional synthetic-enhancer-design stage: mutate candidates for cell-type specificity.

    Doubles as the single-run config consumed by ``regulonado design --params``: a workflow YAML
    sets ``targets`` to several entries (one per cell-type target, each possibly a different
    ``method``) and lets the pipeline resolve checkpoints/paths per (target, shard); a standalone
    CLI run sets ``targets`` to exactly one entry and supplies
    ``fasta``/``intervals``-or-``dataset_dir``/``out_dir``/``checkpoint_dirs`` directly (the
    pipeline instead supplies the I/O paths from ``inputs:`` and resolves checkpoints at shell
    time, so those fields are legitimately left unset in a workflow YAML).
    """

    model_config = ConfigDict(extra="forbid")

    candidates: str = Field(min_length=1)
    shards: int = Field(default=1, ge=1)
    holdout_run: str | None = None
    design_runs: list[str] | None = None
    checkpoint_dirs: list[str] | None = None
    holdout_checkpoint: str | None = None
    targets: list[DesignTarget] = Field(min_length=1)

    # Standalone-CLI I/O: unused when nested under a full workflow run, where the pipeline
    # supplies these from `inputs:` and the rule's own per-shard output path instead.
    fasta: str | None = None
    intervals: str | None = None
    dataset_dir: str | None = None
    track_sheet: str | None = None
    out_dir: str | None = None
    seed: int | None = None

    # Search tuning shared by every target unless a target's own `settings` overrides it (the
    # workflow rule flattens common+per-target settings into these fields before calling
    # `regulonado design --params`; see workflow/rules/design.smk). Defaults match the CLI's
    # previous per-flag defaults. Only set `top_k`/`ism_stride`/`ism_positions` (ISM-only) or
    # `population_size`/`model_queries_per_batch`/`mu`/`recomb_rate`/`threshold`/`rho`
    # (AdaLead-only) at this top level when every target in the file shares one `method` — for a
    # mixed-method file (like examples/enhancer_design.yaml) set them per-target instead, via
    # `targets[i].settings`.
    rounds: int = Field(default=20, ge=1)
    pad: int = Field(default=0, ge=0)
    top_k: int = Field(default=1, ge=1)
    ism_stride: int = Field(default=1, ge=1)
    ism_positions: str | None = None
    population_size: int = Field(default=20, ge=1)
    model_queries_per_batch: int | None = Field(default=None, ge=1)
    top_n: int = Field(default=10, ge=1)
    mu: float = Field(default=1.0, gt=0)
    recomb_rate: float = Field(default=0.1, ge=0)
    threshold: float = 0.1
    rho: int = Field(default=2, ge=1)
    on_missing: Literal["error", "center", "skip"] = "error"
    offtarget_reduction: Literal["logsumexp", "max", "mean"] = "logsumexp"
    objective: Literal["specificity", "selective-activation"] = "specificity"
    offtarget_boost_weight: float = Field(default=1.0, ge=0)
    offtarget_boost_tolerance: float = Field(default=0.0, ge=0)
    offtarget_temperature: float = Field(default=1.0, gt=0)
    gain_transform: Literal["raw", "log2-fold-change"] = "raw"
    gain_pseudocount: float = Field(default=1.0, gt=0)
    target_alpha: float = 1.0
    bending_factor: float = 0.0
    bin_reduction: Literal["mean", "topk"] = "mean"
    topk_bins: int = Field(default=10, ge=1)
    fold_mode: Literal["resident", "sequential"] = "resident"
    batch_size: int = Field(default=8, ge=1)
    device: str | None = None
    exclude_tracks: list[str] = Field(default_factory=list)
    wandb: bool = False
    wandb_project: str = "regulonado-design"
    wandb_group: str | None = None

    @model_validator(mode="after")
    def _target_names_are_unique(self) -> "DesignConfig":
        names = [target.name for target in self.targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"design.targets names must be unique; repeated: {', '.join(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _method_specific_settings_are_consistent(self) -> "DesignConfig":
        """Reject an ISM-only/AdaLead-only field set (top-level, or in a target's own
        `settings`) for a target whose `method` never reads it — the CLI used to accept and
        silently ignore these (docs/audit-2026-09-11.xml F03).
        """
        ism_only = {"top_k", "ism_stride", "ism_positions"}
        adalead_only = {
            "population_size",
            "model_queries_per_batch",
            "mu",
            "recomb_rate",
            "threshold",
            "rho",
        }
        set_fields = self.model_fields_set
        for target in self.targets:
            explicit = {name for name in (ism_only | adalead_only) if name in set_fields}
            explicit |= set(target.settings)
            forbidden = adalead_only if target.method == "ism" else ism_only
            bad = sorted(explicit & forbidden)
            if bad:
                other = "AdaLead" if target.method == "ism" else "ISM"
                raise ValueError(
                    f"design target {target.name!r} uses method {target.method!r}, which "
                    f"ignores {other}-only setting(s): {', '.join(bad)}"
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

    Doubles as the single-run config consumed by ``regulonado attribute --params``: a workflow
    YAML sets ``targets`` to several entries and lets the pipeline resolve checkpoints/paths per
    (target, shard); a standalone CLI run sets ``targets`` to exactly one entry and supplies
    ``fasta``/``intervals``-or-``dataset_dir``/``out_dir``/``checkpoint_dirs`` directly (the
    pipeline instead supplies the I/O paths and resolves checkpoints at shell time, so those
    fields are legitimately left unset in a workflow YAML).
    """

    model_config = ConfigDict(extra="forbid")

    candidates: str = Field(min_length=1)
    shards: int = Field(default=1, ge=1)
    runs: list[str] | None = None
    checkpoint_dirs: list[str] | None = Field(default=None, min_length=1)
    targets: list[AttributionTarget] = Field(min_length=1)

    # Standalone-CLI I/O: unused when this config is nested under a full workflow run, where the
    # pipeline supplies these from `inputs:` and the rule's own per-shard output path instead.
    fasta: str | None = None
    intervals: str | None = None
    dataset_dir: str | None = None
    out_dir: str | None = None

    # ISM-sweep tuning, shared by every target; a target's own `settings` can override any of
    # these per-target. Defaults match the sweep's previous CLI defaults.
    bin_reduction: Literal["mean", "topk", "max"] = "mean"
    topk_bins: int = Field(default=10, ge=1)
    fold_reduction: Literal["mean", "median"] = "mean"
    pad: int = Field(default=0, ge=0)
    stride: int = Field(default=1, ge=1)
    positions: str | None = None
    on_missing: Literal["error", "center", "skip"] = "error"
    smooth_bp: int = Field(default=25, ge=1)
    quantile: float = Field(default=0.90, gt=0.0, lt=1.0)
    min_width_bp: int = Field(default=50, ge=1)
    merge_gap_bp: int = Field(default=20, ge=0)
    min_zscore: float = 1.5
    max_cores_per_candidate: int = Field(default=1, ge=1)
    anchor: Literal["centroid", "peak"] = "centroid"
    fix_width: int | None = Field(default=None, ge=1)
    bigwig: bool = True
    rtol: float = Field(default=0.01, ge=0.0)
    fold_mode: Literal["resident", "sequential"] = "resident"
    batch_size: int = Field(default=8, ge=1)
    device: str | None = None

    @model_validator(mode="after")
    def _target_names_are_unique(self) -> "AttributionConfig":
        names = [target.name for target in self.targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"attribution.targets names must be unique; repeated: {', '.join(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _mode_specific_options_are_consistent(self) -> "AttributionConfig":
        set_fields = self.model_fields_set
        if "topk_bins" in set_fields and self.bin_reduction != "topk":
            raise ValueError(
                "attribution.topk_bins only applies when bin_reduction='topk'; it was set "
                f"while bin_reduction={self.bin_reduction!r}"
            )
        if "stride" in set_fields and self.stride != 1 and self.positions is not None:
            raise ValueError(
                "attribution.stride is ignored once positions restricts the sweep to explicit "
                "coordinates; set only one of them"
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
    qc: QCConfig = Field(default_factory=QCConfig)
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
    def _anchor_disables_squash(self) -> "RegulonadoConfig":
        if self.scaling.method == "anchor":
            common_squash = self.train.common.get("data.apply_squash", True)
            phase_squash = [
                phase.settings.get("data.apply_squash", common_squash)
                for phase in self.train.phases
            ]
            if common_squash or any(phase_squash):
                raise ValueError(
                    "scaling.method='anchor' requires data.apply_squash=false in "
                    "train.common or every phase settings"
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
                f"attribution.runs names train.runs entries that don't exist: {', '.join(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def _anchor_qc_requires_anchor_scaling(self) -> "RegulonadoConfig":
        if "anchor" in self.qc.checks and self.scaling.method != "anchor":
            raise ValueError(
                "qc.checks includes 'anchor', which reuses the anchor scale-factor "
                "diagnostics; requires scaling.method: anchor"
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
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))
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
        for item in data.get("train", {}).get("phases", []) + data.get("train", {}).get("runs", []):
            if not item.get("settings"):
                item.pop("settings", None)
        return data
