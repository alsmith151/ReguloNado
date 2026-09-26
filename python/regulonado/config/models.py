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

from regulonado.training.overrides import merge_training_settings

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Phase presets per trunk mode: python/configs/experiment/*.yaml compose over train.yaml
# (the trunk runs live, every step); python/configs/cached_experiment/*.yaml over
# train_cached.yaml (a head trained on cached trunk embeddings).
LIVE_PRESETS = ("head_only", "unfreeze_output", "deep_finetune", "peak_finetune", "lora_finetune")
CACHED_PRESETS = ("pretrain", "specific", "target")
PRESETS_BY_TRUNK = {"live": LIVE_PRESETS, "cached": CACHED_PRESETS}
# (trunk, target) pairs with a training implementation.
SUPPORTED_RUN_KINDS = {("live", "profile"), ("cached", "region_counts")}
# AlphaGenome takes flexible-length input; these are embed_regions' defaults.
ALPHAGENOME_CONTEXT = 1_048_576
ALPHAGENOME_STRIDE = 524_288
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
    """The genome and the tracks, shared by every run.

    Tracks are discovered once, from bigWigs when a run predicts ``profile`` targets
    (rows then also carry a ``bam`` for ``region_counts`` runs), otherwise from BAMs --
    see :meth:`RegulonadoConfig.track_format`. A track sheet can list both files.
    """

    model_config = ConfigDict(extra="forbid")

    fasta: str
    bigwig_dir: str | None = None
    bam_dir: str | None = Field(
        default=None,
        description=(
            "BAM directory. Tracks find their BAM here as <sample_id or bigWig stem>.bam, or "
            "the one file naming sample_id as a token; discovered from BAMs, every *.bam is "
            "a track unless a sheet names them. Needed for scaling 'bamnado'."
        ),
    )
    track_sheet: str | None = Field(
        default=None,
        description=(
            "CSV mapping tracks to annotation (bigwig and/or bam per row); supplies the "
            "ordered track list."
        ),
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
    track_annotations: str | None = Field(
        default=None,
        description=(
            "CSV/parquet of track_name + extra columns (e.g. 'group') merged in at 'tracks "
            "assemble' — a separate DAG input from track_sheet/bigwig_dir, so editing it "
            "re-runs only assembly, not discovery."
        ),
    )
    drop_missing: bool = True
    dedupe_tracks: Literal["none", "identity", "content"] = "content"

    def has_track_source(self, track_format: str) -> bool:
        """Whether any configured source yields *track_format* (``bigwig``/``bam``) files."""
        if self.track_sheet or self.seqnado_projects:
            return True
        return bool(self.bigwig_dir if track_format == "bigwig" else self.bam_dir)


class ProfileTargetConfig(BaseModel):
    """Binned coverage profiles over ``intervals``, from bigWigs: the ``profile`` dataset."""

    model_config = ConfigDict(extra="forbid")

    intervals: str = Field(min_length=1)
    context_length: int = Field(default=524_288, ge=1)
    bin_size: int = Field(default=32, ge=1)
    n_pred_bins: int = Field(default=6_144, ge=1)
    shift_max_bp: int = Field(default=64, ge=0)
    extract_threads: int = Field(default=32, ge=1)
    write_threads: int = Field(default=4, ge=1)
    zstd_level: int = Field(default=3, ge=1)
    rows_per_row_group: int = Field(default=1, ge=1)
    stage_to_scratch: bool = True
    # See `regulonado dataset --help`: bin mean over in-contig width vs recorded bases,
    # and the stored value for padding / all-NaN bins.
    bin_denominator: Literal["bin_width", "covered_bases"] = "bin_width"
    missing_bins: Literal["nan", "zero"] = "nan"

    @model_validator(mode="after")
    def _shift_is_whole_bins(self) -> "ProfileTargetConfig":
        if self.shift_max_bp % self.bin_size:
            raise ValueError(
                f"shift_max_bp ({self.shift_max_bp}) must be a multiple of "
                f"bin_size ({self.bin_size})"
            )
        return self


class RegionCountsTargetConfig(BaseModel):
    """One count per region per track, from BAMs: the ``region_counts`` dataset."""

    model_config = ConfigDict(extra="forbid")

    regions: str = Field(
        min_length=1,
        description=(
            "BED/parquet region set; its 'split' column, and target_start/target_end if "
            "present, are used as-is."
        ),
    )
    target_width: int = Field(default=1000, ge=1)
    threads: int = Field(default=1, ge=1, description="htslib decompression threads per BAM.")
    chrom_sizes: str | None = None
    val_chroms: list[str] = Field(
        default_factory=list,
        description="Chromosomes assigned split=val when 'regions' has no 'split' column.",
    )
    test_chroms: list[str] = Field(
        default_factory=list,
        description="Chromosomes assigned split=test when 'regions' has no 'split' column.",
    )
    anchor_regions: str | None = Field(
        default=None,
        description="High-anchor BED/parquet for size factors; defaults to scaling.anchor_regions.",
    )
    background_regions: str | None = Field(
        default=None,
        description="Background BED/parquet; defaults to scaling.background_regions.",
    )
    exclude_regions: str | None = Field(
        default=None,
        description=(
            "BED/parquet of held-out sequences (e.g. benchmark candidates); train regions "
            "overlapping one are dropped in every phase (data.exclude_regions)."
        ),
    )


class TargetsConfig(BaseModel):
    """What runs can be trained to predict; each dataset is built only if a run uses it."""

    model_config = ConfigDict(extra="forbid")

    profile: ProfileTargetConfig | None = None
    region_counts: RegionCountsTargetConfig | None = None


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
    """One step of a recipe. ``preset`` must suit the trunk mode of every run using it:
    :data:`LIVE_PRESETS` for ``trunk: live``, :data:`CACHED_PRESETS` for ``trunk: cached``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    preset: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))


class BackboneConfig(BaseModel):
    """The pretrained trunk a run starts from."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["borzoi", "enformer", "alphagenome"] = "borzoi"
    pretrained: str = Field(
        min_length=1, description="Checkpoint name or path (see python/configs/backbone/)."
    )


class TrunkCacheConfig(BaseModel):
    """How a ``trunk: cached`` run's embeddings are computed (``regulonado embed regions``)."""

    model_config = ConfigDict(extra="forbid")

    context: int | None = Field(
        default=None, ge=1, description="Input length (bp); AlphaGenome only."
    )
    stride: int | None = Field(
        default=None, ge=1, description="Central kept span per window (bp); AlphaGenome only."
    )
    pool_to: int | None = Field(
        default=None, ge=1, description="Average adjacent bins to this bp width."
    )
    rc: bool = Field(default=False, description="Also cache a reverse-complement pass.")
    batch_size: int = Field(default=1, ge=1, description="Windows per forward pass.")


class TrainRun(BaseModel):
    """One model: a backbone, how its trunk is used, what it predicts, and its recipe."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = Field(ge=0)
    recipe: str = Field(min_length=1, description="Name of one of train.recipes.")
    backbone: BackboneConfig
    trunk: Literal["live", "cached"] = Field(
        default="live",
        description=(
            "live: the trunk runs every step (and can be fine-tuned); cached: it runs once "
            "and a head trains on its stored embeddings."
        ),
    )
    target: Literal["profile", "region_counts"] = Field(
        default="profile", description="Which targets.* dataset the run predicts."
    )
    target_group: str | None = Field(
        default=None,
        description="Cell-type group stage-specific presets are relative to (data.target_group).",
    )
    cache: TrunkCacheConfig | None = Field(
        default=None, description="Embedding settings; trunk: cached only."
    )
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))

    @model_validator(mode="after")
    def _kind_is_supported(self) -> "TrainRun":
        if (self.trunk, self.target) not in SUPPORTED_RUN_KINDS:
            supported = ", ".join(f"{t}/{g}" for t, g in sorted(SUPPORTED_RUN_KINDS))
            raise ValueError(
                f"run {self.name!r}: trunk {self.trunk!r} with target {self.target!r} is not "
                f"implemented (supported trunk/target: {supported})"
            )
        if self.target_group is not None and self.trunk != "cached":
            raise ValueError(
                f"run {self.name!r}: target_group applies to trunk: cached runs "
                "(data.target_group); live runs set their target in settings"
            )
        if self.cache is not None and self.trunk != "cached":
            raise ValueError(f"run {self.name!r}: 'cache' only applies to trunk: cached")
        if (
            self.cache is not None
            and self.backbone.type != "alphagenome"
            and (self.cache.context is not None or self.cache.stride is not None)
        ):
            raise ValueError(
                f"run {self.name!r}: cache.context/stride only apply to backbone 'alphagenome' "
                "(fixed-input backbones tile by their own output span)"
            )
        return self

    def cache_name(self) -> str:
        """Directory name of this run's embedding cache: runs that would compute the same
        embeddings share one."""
        cache = self.cache or TrunkCacheConfig()
        pretrained = re.sub(r"[^A-Za-z0-9._-]+", "_", self.backbone.pretrained).strip("_.")
        parts = [self.backbone.type, pretrained]
        if self.backbone.type == "alphagenome":
            parts += [
                f"ctx{cache.context or ALPHAGENOME_CONTEXT}",
                f"stride{cache.stride or ALPHAGENOME_STRIDE}",
            ]
        if cache.pool_to:
            parts.append(f"pool{cache.pool_to}")
        if cache.rc:
            parts.append("rc")
        return "-".join(parts)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nproc_per_node: int = Field(default=1, ge=1)
    common: dict[str, Any] = Field(default_factory=dict)
    recipes: dict[str, list[TrainPhase]] = Field(
        min_length=1, description="Named phase chains; each run follows one."
    )
    runs: list[TrainRun] = Field(min_length=1)

    @model_validator(mode="after")
    def _recipes_and_runs_are_consistent(self) -> "TrainConfig":
        for name, phases in self.recipes.items():
            _validate_name(name)
            if not phases:
                raise ValueError(f"train.recipes.{name} has no phases")
            names = [phase.name for phase in phases]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(
                    f"train.recipes.{name} phase names must be unique; repeated: "
                    f"{', '.join(duplicates)}"
                )
        names = [run.name for run in self.runs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"train.runs names must be unique; repeated: {', '.join(duplicates)}")
        for run in self.runs:
            if run.recipe not in self.recipes:
                raise ValueError(
                    f"run {run.name!r} names recipe {run.recipe!r}, not one of train.recipes: "
                    f"{', '.join(self.recipes)}"
                )
            allowed = PRESETS_BY_TRUNK[run.trunk]
            wrong = [p.preset for p in self.recipes[run.recipe] if p.preset not in allowed]
            if wrong:
                raise ValueError(
                    f"run {run.name!r} (trunk: {run.trunk}) uses recipe {run.recipe!r}, whose "
                    f"preset(s) {', '.join(wrong)} are not trunk-{run.trunk} presets "
                    f"({', '.join(allowed)})"
                )
        for label, settings in [
            ("train.common", self.common),
            *(
                (f"train.recipes.{recipe}[{phase.name}].settings", phase.settings)
                for recipe, phases in self.recipes.items()
                for phase in phases
            ),
            *((f"train.runs[{run.name}].settings", run.settings) for run in self.runs),
        ]:
            try:
                merge_training_settings([settings])
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
        return self

    def run(self, name: str) -> TrainRun:
        return next(run for run in self.runs if run.name == name)

    def final_phase(self, run_name: str) -> str:
        """The last phase of *run_name*'s recipe: the checkpoint downstream stages use."""
        return self.recipes[self.run(run_name).recipe][-1].name


class ParameterSweepConfig(BaseModel):
    """Optional W&B-managed GPU parameter sweep."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sweep_config: str = Field(min_length=1)
    agents: int = Field(default=1, ge=1)
    trials_per_agent: int = Field(default=1, ge=1)
    cpus_per_agent: int = Field(default=4, ge=1)
    mem_mb_per_agent: int = Field(default=64_000, ge=1)
    runtime_minutes_per_agent: int = Field(default=240, ge=1)
    wandb_project: str = Field(default="regulonado-parameter-sweep", min_length=1)


class PredictionConfig(BaseModel):
    """Optional BigWig-prediction stage for one completed training run.

    Omitting ``tracks`` writes predictions for every track in the trained
    dataset. Set it to a non-empty list of track names (or zero-based output
    indices) to select outputs. Prediction is whole-genome by default; ``bed``
    switches it to supplied non-overlapping target regions.
    """

    model_config = ConfigDict(extra="forbid")

    run: str
    tracks: list[str] | None = Field(default=None, min_length=1)
    bed: str | None = None
    whole_genome: bool = True
    chromsizes: str | None = None
    rtol: float = Field(default=0.01, ge=0.0)
    batch_size: int = Field(default=4, ge=1)
    device: str | None = None
    inverse_squash: bool = False

    @model_validator(mode="after")
    def _prediction_extent_is_unambiguous(self) -> "PredictionConfig":
        if self.bed is not None and self.whole_genome:
            raise ValueError("prediction.bed and prediction.whole_genome cannot both be set")
        if self.bed is None and not self.whole_genome:
            raise ValueError("prediction needs either prediction.bed or whole_genome: true")
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

    candidates: str | None = Field(default=None, min_length=1)
    from_attribution: str | None = Field(
        default=None,
        description=(
            "Attribution target name to consume core_regions.bed from, instead of writing out "
            "'candidates' by hand (workflow runs only, resolved via the shared results_dir)."
        ),
    )
    shards: int = Field(default=1, ge=1)
    holdout_run: str | None = None
    design_runs: list[str] | None = None
    checkpoint_dirs: list[str] | None = None
    holdout_checkpoint: str | None = None
    targets: list[DesignTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one_candidates_source(self) -> "DesignConfig":
        if (self.candidates is None) == (self.from_attribution is None):
            raise ValueError(
                "design must set exactly one of 'candidates' (a BED path) or "
                "'from_attribution' (an attribution target name to chain from)"
            )
        return self

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
    # Widens only the scored bins (e.g. to pick up a nucleosome-free-region dip flanking a
    # narrow candidate); `pad` above widens the editable span instead, which the search may
    # actually mutate. The two are independent — set either, both, or neither.
    score_pad_bp: int = Field(default=0, ge=0)
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

    @model_validator(mode="after")
    def _adalead_budget_covers_initial_population(self) -> "DesignConfig":
        """AdaLead spends its first ``population_size`` model queries just scoring the initial
        population, before it can propose a single mutated/recombined child. A budget that
        doesn't exceed that cost runs zero search rounds and raises deep inside `search.py`
        ("No sequences generated...") only after the shard job has already spun up and paid for
        a checkpoint load — catch it here instead, at config-validation time.
        """
        for target in self.targets:
            if target.method != "adalead":
                continue
            population_size = target.settings.get("population_size", self.population_size)
            model_queries_per_batch = target.settings.get(
                "model_queries_per_batch", self.model_queries_per_batch
            )
            if model_queries_per_batch is not None and model_queries_per_batch <= population_size:
                raise ValueError(
                    f"design target {target.name!r}: model_queries_per_batch "
                    f"({model_queries_per_batch}) must exceed population_size "
                    f"({population_size}) — the first population_size queries just score the "
                    "initial population, leaving none for AdaLead to actually search with"
                )
        return self


class AttributionTarget(BaseModel):
    """One attribution readout: either an exact ``track``, or a track ``group``.

    A ``group`` readout averages every track whose ``group_by`` column (a
    ``TrackRecord``/``tracks.parquet`` column such as ``source`` or a freeform
    ``group`` label) equals ``target`` — the same grouping ``design`` targets
    already use via ``DesignTarget.group_by``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    track: str | None = None
    target: str | None = None
    group_by: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name")(staticmethod(_validate_name))

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> "AttributionTarget":
        if (self.track is None) == (self.target is None):
            raise ValueError(
                f"attribution target {self.name!r} must set exactly one of 'track' "
                f"(an exact track name/index) or 'target' (a group value, with 'group_by')"
            )
        if self.track is not None and self.group_by is not None:
            raise ValueError(
                f"attribution target {self.name!r} sets 'group_by' but not 'target'; "
                f"'group_by' only applies to a 'target'-based group readout"
            )
        if self.target is not None and self.group_by is None:
            self.group_by = "source"
        return self


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

    # Group-target resolution: a `target`-based AttributionTarget resolves against `track_sheet`
    # if set, else `dataset_dir`/tracks.parquet. `exclude_tracks` removes tracks from every
    # group (and from off-target consideration) regardless of which one they'd otherwise match.
    track_sheet: str | None = None
    exclude_tracks: list[str] = Field(default_factory=list)

    # 'gradient' trades ISM's per-position, per-alt-base re-scoring (one forward+backward pass
    # per fold total, instead of ~positions x 3 x folds forward passes) for a first-order Taylor
    # approximation of the same quantity — much cheaper, but can misjudge positions where the
    # true effect saturates or depends on other edits (see design.attribution.grad_scan).
    method: Literal["ism", "gradient"] = "ism"

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
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    train: TrainConfig | None = None
    parameter_sweep: ParameterSweepConfig | None = None
    prediction: PredictionConfig | None = None
    design: DesignConfig | None = None
    attribution: AttributionConfig | None = None

    def track_format(self) -> str:
        """``bigwig`` when a profile target is configured, else ``bam``.

        One track table serves every run: bigWig tracks for profile datasets (each row also
        carrying its ``bam`` when a sheet or ``inputs.bam_dir`` names one, so region-count
        runs count the same tracks), or BAM tracks when only region counts are needed.
        """
        return "bigwig" if self.targets.profile is not None else "bam"

    def runs_for(self, target: str) -> list[TrainRun]:
        return [run for run in (self.train.runs if self.train else []) if run.target == target]

    def _profile_run_names(self) -> set[str]:
        return {run.name for run in self.runs_for("profile")}

    @model_validator(mode="after")
    def _run_targets_are_configured(self) -> "RegulonadoConfig":
        for run in self.train.runs if self.train else []:
            if getattr(self.targets, run.target) is None:
                raise ValueError(
                    f"run {run.name!r} predicts {run.target!r}, but targets.{run.target} is "
                    "not configured"
                )
        return self

    @model_validator(mode="after")
    def _tracks_have_a_source(self) -> "RegulonadoConfig":
        if self.targets.profile is None and self.targets.region_counts is None:
            return self
        track_format = self.track_format()
        if not self.inputs.has_track_source(track_format):
            options = (
                "bigwig_dir, track_sheet or seqnado_projects"
                if track_format == "bigwig"
                else "bam_dir, track_sheet or seqnado_projects"
            )
            raise ValueError(f"inputs needs one of {options} to discover {track_format} tracks")
        return self

    @model_validator(mode="after")
    def _bigwig_stages_need_a_profile_target(self) -> "RegulonadoConfig":
        if self.targets.profile is not None:
            return self
        if self.qc.checks:
            raise ValueError("qc.checks scan bigWig tracks; configure targets.profile")
        if self.parameter_sweep is not None and self.parameter_sweep.enabled:
            raise ValueError(
                "parameter_sweep trains on the profile dataset; configure targets.profile"
            )
        return self

    @model_validator(mode="after")
    def _downstream_runs_predict_profiles(self) -> "RegulonadoConfig":
        """Prediction, attribution and design read bigWig-shaped profile models."""
        referenced: list[tuple[str, list[str]]] = []
        if self.prediction is not None:
            referenced.append(("prediction.run", [self.prediction.run]))
        if self.design is not None:
            referenced.append(
                ("design.holdout_run", [self.design.holdout_run] if self.design.holdout_run else [])
            )
            referenced.append(("design.design_runs", self.design.design_runs or []))
        if self.attribution is not None:
            referenced.append(("attribution.runs", self.attribution.runs or []))
        if not any(names for _, names in referenced):
            if self.prediction is not None and self.train is None:
                raise ValueError(
                    "prediction requires train so prediction.run can resolve a checkpoint"
                )
            return self
        if self.train is None:
            raise ValueError(f"{referenced[0][0]} names a run, but train is not configured")
        all_runs = {run.name for run in self.train.runs}
        profile_runs = self._profile_run_names()
        for label, names in referenced:
            unknown = sorted(name for name in names if name not in all_runs)
            if unknown:
                raise ValueError(
                    f"{label} names train.runs entries that don't exist: {', '.join(unknown)}"
                )
            wrong = sorted(name for name in names if name not in profile_runs)
            if wrong:
                raise ValueError(
                    f"{label} names run(s) {', '.join(wrong)} that don't predict profiles; "
                    "prediction, attribution and design need target: profile runs"
                )
        return self

    @model_validator(mode="after")
    def _anchor_disables_squash(self) -> "RegulonadoConfig":
        if self.scaling.method != "anchor" or self.train is None:
            return self
        for run in self.runs_for("profile"):
            for phase in self.train.recipes[run.recipe]:
                settings = merge_training_settings(
                    [self.train.common, phase.settings, run.settings]
                )
                # Count labels ignore apply_squash entirely.
                if settings.get("data.label_space") == "counts":
                    continue
                if settings.get("data.apply_squash", True) is not False:
                    raise ValueError(
                        "scaling.method='anchor' requires data.apply_squash=false; "
                        f"resolved true for run {run.name!r}, phase {phase.name!r}"
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
    def _region_count_anchors_are_present(self) -> "RegulonadoConfig":
        counts = self.targets.region_counts
        if counts is None:
            return self
        for field_name in ("anchor_regions", "background_regions"):
            if not (getattr(counts, field_name) or getattr(self.scaling, field_name)):
                raise ValueError(
                    f"targets.region_counts.{field_name} is required (or set "
                    f"scaling.{field_name} with scaling.method: anchor to share it with the "
                    "profile tracks)"
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
            if len(self.inputs.seqnado_projects) > 1:
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
        train = data.get("train", {})
        if not train.get("common"):
            train.pop("common", None)
        phases = [phase for recipe in train.get("recipes", {}).values() for phase in recipe]
        for item in phases + train.get("runs", []):
            if not item.get("settings"):
                item.pop("settings", None)
        return data
