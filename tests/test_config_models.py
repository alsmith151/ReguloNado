"""The pydantic config models and the workflow's JSON Schema must not drift apart."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from regulonado.config.models import (
    AttributionConfig,
    AttributionTarget,
    BackboneConfig,
    DesignConfig,
    DesignTarget,
    InputsConfig,
    ProfileTargetConfig,
    RegionCountsTargetConfig,
    RegulonadoConfig,
    SeqNadoProjectRef,
    TargetsConfig,
    TrainConfig,
    TrainPhase,
    TrainRun,
    TrunkCacheConfig,
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


PROFILE = TargetsConfig(profile=ProfileTargetConfig(intervals="intervals.bed"))


def _run(name: str = "run_a", *, seed: int = 0, pretrained: str = "model/a", **kwargs) -> TrainRun:
    return TrainRun(
        name=name,
        seed=seed,
        recipe=kwargs.pop("recipe", "finetune"),
        backbone=kwargs.pop("backbone", None)
        or BackboneConfig(
            type=kwargs.pop("type", "borzoi"),
            pretrained=pretrained,
            features=kwargs.pop("features", "trunk"),
        ),
        **kwargs,
    )


def _train() -> TrainConfig:
    return TrainConfig(
        recipes={"finetune": [TrainPhase(name="head", preset="head_only")]},
        runs=[_run()],
    )


def _config(**inputs: Any) -> RegulonadoConfig:
    return RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", **inputs),
        targets=PROFILE,
        train=_train(),
    )


def test_parameter_sweep_does_not_require_training_matrix() -> None:
    config = RegulonadoConfig(
        results_dir="results",
        targets=PROFILE,
        inputs=InputsConfig(fasta="genome.fa", bigwig_dir="bigwigs"),
        parameter_sweeps={
            "heads": {
                "enabled": True,
                "sweep_config": "sweep.yaml",
                "agents": 4,
                "trials_per_agent": 2,
                "cpus_per_agent": 4,
                "mem_mb_per_agent": 64_000,
                "runtime_minutes_per_agent": 240,
            }
        },
    )

    assert config.train is None
    _validate_against_schema(config.to_dict())


def test_region_downstream_explicit_checkpoints_need_no_train() -> None:
    """A finished live region checkpoint can run attribution/design without recreating train."""
    target = RegionCountsTargetConfig(
        regions="regions.parquet",
        anchor_regions="anchors.bed",
        background_regions="background.parquet",
    )
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
        targets=TargetsConfig(region_counts=target),
        attribution=AttributionConfig(
            candidates="candidates.bed",
            checkpoint_dirs=["locon-target"],
            model_kind="region_counts",
            targets=[AttributionTarget(name="hl60", target="HL-60")],
        ),
        design=DesignConfig(
            from_attribution="hl60",
            checkpoint_dirs=["locon-target"],
            model_kind="region_counts",
            targets=[DesignTarget(name="hl60", target="HL-60")],
        ),
    )

    assert config.train is None
    _validate_against_schema(config.to_dict())


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
        targets=PROFILE,
        inputs=InputsConfig(
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
            common={"trainer": {"max_epochs": 3}},
            recipes={
                "finetune": [
                    TrainPhase(name="head", preset="head_only"),
                    TrainPhase(name="deep", preset="deep_finetune", settings={"seed": 1}),
                ]
            },
            runs=[
                _run(
                    "run_a",
                    seed=0,
                    pretrained="model/a",
                    settings={"trainer": {"learning_rate": 1e-4}},
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
    assert "settings" not in data["train"]["recipes"]["finetune"][0]
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


def test_profile_tracks_need_a_bigwig_source():
    with pytest.raises(ValueError, match="bigwig_dir, track_sheet or seqnado_projects"):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
            targets=PROFILE,
            train=_train(),
        )


def test_runs_need_their_target_configured():
    with pytest.raises(ValueError, match=r"targets.profile is\s+not configured"):
        RegulonadoConfig(
            results_dir="results", inputs=InputsConfig(fasta="genome.fa"), train=_train()
        )


def test_shift_max_bp_must_be_a_whole_number_of_bins():
    with pytest.raises(ValueError, match="must be a multiple of"):
        ProfileTargetConfig(intervals="i.bed", bin_size=32, shift_max_bp=48)

    assert ProfileTargetConfig(intervals="i.bed", bin_size=32, shift_max_bp=64).shift_max_bp == 64


def test_duplicate_phase_names_raise():
    with pytest.raises(
        ValueError, match=r"train\.recipes\.finetune phase names must be unique.*head"
    ):
        TrainConfig(
            recipes={
                "finetune": [
                    TrainPhase(name="head", preset="head_only"),
                    TrainPhase(name="head", preset="deep_finetune"),
                ]
            },
            runs=[_run("run_a", seed=0, pretrained="model/a")],
        )


def test_duplicate_run_names_raise():
    with pytest.raises(ValueError, match=r"train\.runs names must be unique.*run_a"):
        TrainConfig(
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")]},
            runs=[
                _run("run_a", seed=0, pretrained="model/a"),
                _run("run_a", seed=1, pretrained="model/b"),
            ],
        )


def test_training_settings_require_nested_canonical_syntax():
    with pytest.raises(ValueError, match="obsolete dotted YAML syntax"):
        TrainConfig(
            common={"trainer.max_epochs": 2},
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")]},
            runs=[_run("run_a", seed=0, pretrained="model/a")],
        )


def test_anchor_scaling_checks_every_resolved_phase_and_run() -> None:
    base = {
        "results_dir": "results",
        "inputs": {"fasta": "genome.fa", "bigwig_dir": "bw"},
        "targets": {"profile": {"intervals": "intervals.bed"}},
        "scaling": {
            "method": "anchor",
            "anchor_regions": "anchors.bed",
            "background_regions": "background.bed",
        },
        "train": {
            "common": {"data": {"apply_squash": False}},
            "recipes": {"finetune": [{"name": "head", "preset": "head_only"}]},
            "runs": [
                {
                    "name": "run_a",
                    "seed": 0,
                    "recipe": "finetune",
                    "backbone": {"pretrained": "model/a"},
                }
            ],
        },
    }
    RegulonadoConfig.model_validate(base)

    conflicting = copy.deepcopy(base)
    conflicting["train"]["runs"][0]["settings"] = {"data": {"apply_squash": True}}
    with pytest.raises(ValueError, match="resolved true for run 'run_a', phase 'head'"):
        RegulonadoConfig.model_validate(conflicting)

    counts = copy.deepcopy(base)
    counts["train"]["common"] = {"data": {"label_space": "counts"}}
    RegulonadoConfig.model_validate(counts)


def test_seqnado_scaling_across_several_projects_is_rejected():
    projects = [
        SeqNadoProjectRef(name="expA", path="expA/seqnado_output"),
        SeqNadoProjectRef(name="expB", path="expB/seqnado_output"),
    ]

    with pytest.raises(ValueError, match="only comparable within that project"):
        RegulonadoConfig(
            results_dir="results",
            targets=PROFILE,
            inputs=InputsConfig(fasta="genome.fa", seqnado_projects=projects),
            scaling={"method": "seqnado"},
            train=_train(),
        )


def test_seqnado_scaling_with_a_named_project_is_still_rejected():
    with pytest.raises(ValueError, match="only comparable within that project"):
        RegulonadoConfig(
            results_dir="results",
            targets=PROFILE,
            inputs=InputsConfig(
                fasta="genome.fa",
                seqnado_projects=[
                    SeqNadoProjectRef(name="expA", path="expA/seqnado_output"),
                    SeqNadoProjectRef(name="expB", path="expB/seqnado_output"),
                ],
            ),
            scaling={"method": "seqnado", "seqnado_project": "expA/seqnado_output"},
            train=_train(),
        )


def test_seqnado_scaling_with_exactly_one_project_is_accepted():
    config = RegulonadoConfig(
        results_dir="results",
        targets=PROFILE,
        inputs=InputsConfig(
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
            targets=PROFILE,
            inputs=InputsConfig(fasta="genome.fa", bigwig_dir="bigwigs"),
            scaling={"method": "bamnado"},
            train=_train(),
        )


def test_bamnado_scaling_with_a_bam_dir_is_accepted():
    config = RegulonadoConfig(
        results_dir="results",
        targets=PROFILE,
        inputs=InputsConfig(
            fasta="genome.fa",
            bigwig_dir="bigwigs",
            bam_dir="bams",
        ),
        scaling={"method": "bamnado"},
        train=_train(),
    )

    _validate_against_schema(config.to_dict())


# ---------------------------------------------------------------------- #
#  Runs: trunk, target, recipe                                             #
# ---------------------------------------------------------------------- #

COUNTS = RegionCountsTargetConfig(
    regions="regions.parquet", anchor_regions="anchor.bed", background_regions="bg.bed"
)


def _cached(name: str = "counts_run", **kwargs) -> TrainRun:
    return _run(
        name,
        recipe=kwargs.pop("recipe", "curriculum"),
        type=kwargs.pop("type", "alphagenome"),
        pretrained=kwargs.pop("pretrained", "all_folds"),
        trunk="cached",
        target="region_counts",
        **kwargs,
    )


CURRICULUM = {"curriculum": [TrainPhase(name="pretrain", preset="pretrain")]}


def _counts_config(*runs: TrainRun, **inputs: Any) -> RegulonadoConfig:
    return RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", **(inputs or {"bam_dir": "bams"})),
        targets=TargetsConfig(region_counts=COUNTS),
        train=TrainConfig(recipes=CURRICULUM, runs=list(runs or [_cached()])),
    )


def test_region_count_runs_need_only_a_fasta_and_bams():
    config = _counts_config()
    assert config.track_format() == "bam"
    _validate_against_schema(config.to_dict())


def test_bam_tracks_need_a_bam_source():
    with pytest.raises(ValueError, match="bam_dir, track_sheet or seqnado_projects"):
        _counts_config(bigwig_dir="bigwigs")


def test_profile_and_region_count_runs_share_bigwig_tracks():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", track_sheet="tracks.csv"),
        targets=TargetsConfig(profile=PROFILE.profile, region_counts=COUNTS),
        train=TrainConfig(
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")], **CURRICULUM},
            runs=[_run(), _cached()],
        ),
    )
    assert config.track_format() == "bigwig"
    _validate_against_schema(config.to_dict())


def test_recipe_presets_must_suit_the_run_target():
    with pytest.raises(ValueError, match="not region_counts presets"):
        TrainConfig(
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")]},
            runs=[_cached(recipe="finetune")],
        )
    with pytest.raises(ValueError, match="not profile presets"):
        TrainConfig(recipes=CURRICULUM, runs=[_run(recipe="curriculum")])
    # A live-trunk region-count run follows the same region presets as a cached one.
    live = _run(recipe="curriculum", target="region_counts", target_group="HL-60")
    assert TrainConfig(recipes=CURRICULUM, runs=[live]).runs[0].trunk == "live"


def test_runs_must_name_a_known_recipe():
    with pytest.raises(ValueError, match="names recipe 'missing'"):
        TrainConfig(recipes=CURRICULUM, runs=[_cached(recipe="missing")])


def test_unimplemented_trunk_target_pairs_are_rejected():
    with pytest.raises(ValueError, match="trunk 'cached' with target 'profile' is not implemented"):
        _run(trunk="cached", target="profile")


def test_encoder_features_are_alphagenome_region_count_only():
    encoder = {"type": "alphagenome", "pretrained": "all_folds", "features": "encoder"}
    with pytest.raises(ValueError, match="only available for alphagenome"):
        BackboneConfig(type="borzoi", pretrained="model/a", features="encoder")
    with pytest.raises(ValueError, match="only implemented for target: region_counts"):
        _run(backbone=encoder)
    run = _cached(features="encoder")
    assert run.cache_name() == "alphagenome-all_folds-encoder-ctx4096-stride2048"
    assert run.tiling() == (4096, 2048)


def test_cache_settings_apply_to_cached_alphagenome_runs_only():
    with pytest.raises(ValueError, match="'cache' only applies to trunk: cached"):
        _run(cache=TrunkCacheConfig(pool_to=128))
    with pytest.raises(ValueError, match="only apply to backbone 'alphagenome'"):
        _cached(type="borzoi", pretrained="model/a", cache=TrunkCacheConfig(context=524_288))
    with pytest.raises(ValueError, match="target_group applies to target: region_counts"):
        _run(target_group="HL-60")


def test_runs_with_the_same_trunk_setup_share_a_cache():
    default = _cached("a").cache_name()
    assert default == _cached("b", cache=TrunkCacheConfig(context=1_048_576)).cache_name()
    assert default == "alphagenome-all_folds-ctx1048576-stride524288"
    pooled = _cached(
        "c",
        type="borzoi",
        pretrained="johahi/flashzoi-replicate-0",
        cache=TrunkCacheConfig(pool_to=128, rc=True),
    )
    assert pooled.cache_name() == "borzoi-johahi_flashzoi-replicate-0-pool128-rc"


def test_region_count_anchors_default_to_the_anchor_scaling_windows():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", track_sheet="tracks.csv"),
        targets=TargetsConfig(
            profile=PROFILE.profile, region_counts=RegionCountsTargetConfig(regions="r.parquet")
        ),
        scaling={"method": "anchor", "anchor_regions": "a.bed", "background_regions": "b.bed"},
    )
    assert config.targets.region_counts is not None
    with pytest.raises(ValueError, match="targets.region_counts.anchor_regions is required"):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
            targets=TargetsConfig(region_counts=RegionCountsTargetConfig(regions="r.parquet")),
        )


def test_downstream_stages_need_profile_runs():
    config = _counts_config().model_dump(mode="json", exclude_none=True)
    config["prediction"] = {"run": "counts_run", "whole_genome": True}
    with pytest.raises(ValueError, match="don't predict profiles"):
        RegulonadoConfig.model_validate(config)


def test_bigwig_stages_need_a_profile_target():
    config = _counts_config().model_dump(mode="json", exclude_none=True)
    config["qc"] = {"checks": ["sparsity"]}
    with pytest.raises(ValueError, match="qc.checks scan bigWig tracks"):
        RegulonadoConfig.model_validate(config)


# ---------------------------------------------------------------------- #
#  _downstream_runs_are_supported (batch E1): relaxed for attribution/design  #
# ---------------------------------------------------------------------- #
def _mixed_target_config() -> RegulonadoConfig:
    """A profile run ('run_a') and a region_counts run ('counts_run') in one train.runs."""
    return RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", track_sheet="tracks.csv"),
        targets=TargetsConfig(profile=PROFILE.profile, region_counts=COUNTS),
        train=TrainConfig(
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")], **CURRICULUM},
            runs=[_run(), _cached()],
        ),
    )


def test_attribution_runs_may_all_be_region_counts():
    data = _counts_config().model_dump(mode="json", exclude_none=True)
    data["attribution"] = {
        "candidates": "c.bed",
        "runs": ["counts_run"],
        "targets": [{"name": "t1", "track": "g0"}],
    }
    # Must not raise: a region_counts-only attribution stage is now supported.
    RegulonadoConfig.model_validate(data)


def test_design_runs_and_holdout_run_may_all_be_region_counts():
    data = _counts_config().model_dump(mode="json", exclude_none=True)
    data["design"] = {
        "candidates": "c.bed",
        "design_runs": ["counts_run"],
        "holdout_run": "counts_run",
        "targets": [{"name": "t1", "target": "g0"}],
    }
    RegulonadoConfig.model_validate(data)


def test_attribution_runs_must_share_one_target():
    data = _mixed_target_config().model_dump(mode="json", exclude_none=True)
    data["attribution"] = {
        "candidates": "c.bed",
        "runs": ["run_a", "counts_run"],
        "targets": [{"name": "t1", "track": "g0"}],
    }
    with pytest.raises(ValueError, match="must all predict the same target"):
        RegulonadoConfig.model_validate(data)


def test_design_runs_must_share_one_target():
    data = _mixed_target_config().model_dump(mode="json", exclude_none=True)
    data["design"] = {
        "candidates": "c.bed",
        "design_runs": ["run_a", "counts_run"],
        "targets": [{"name": "t1", "target": "g0"}],
    }
    with pytest.raises(ValueError, match="must all predict the same target"):
        RegulonadoConfig.model_validate(data)


def test_design_holdout_run_must_share_design_runs_target():
    data = _mixed_target_config().model_dump(mode="json", exclude_none=True)
    data["design"] = {
        "candidates": "c.bed",
        "design_runs": ["counts_run"],
        "holdout_run": "run_a",
        "targets": [{"name": "t1", "target": "g0"}],
    }
    with pytest.raises(ValueError, match="must all predict the same target"):
        RegulonadoConfig.model_validate(data)


def test_prediction_run_stays_profile_only_even_alongside_region_attribution():
    """prediction.run is still profile-only even when attribution is happily pointed at a
    region_counts run -- the two stages are validated independently."""
    data = _mixed_target_config().model_dump(mode="json", exclude_none=True)
    data["prediction"] = {"run": "counts_run", "whole_genome": True}
    data["attribution"] = {
        "candidates": "c.bed",
        "runs": ["counts_run"],
        "targets": [{"name": "t1", "track": "g0"}],
    }
    with pytest.raises(ValueError, match="don't predict profiles"):
        RegulonadoConfig.model_validate(data)


def test_downstream_runs_naming_a_nonexistent_run_still_raises():
    data = _counts_config().model_dump(mode="json", exclude_none=True)
    data["attribution"] = {
        "candidates": "c.bed",
        "runs": ["nonexistent_run"],
        "targets": [{"name": "t1", "track": "g0"}],
    }
    with pytest.raises(ValueError, match="don't exist"):
        RegulonadoConfig.model_validate(data)


# ---------------------------------------------------------------------- #
#  DesignConfig/AttributionConfig new fields (batch E1)                      #
# ---------------------------------------------------------------------- #
def test_design_and_attribution_default_to_genomic_flanks_and_auto_model_kind():
    design = DesignConfig(candidates="c.bed", targets=[DesignTarget(name="t1", target="a")])
    assert design.model_kind == "auto"
    assert design.flank_mode == "genomic"
    assert design.flank_keep == "scored_span"
    assert design.flank_keep_bp == 0
    assert design.energy == "contrast"

    attribution = AttributionConfig(
        candidates="c.bed", targets=[AttributionTarget(name="t1", track="a")]
    )
    assert attribution.model_kind == "auto"
    assert attribution.flank_mode == "genomic"
    assert attribution.flank_keep == "scored_span"
    assert attribution.flank_keep_bp == 0


def test_design_region_counts_requires_mean_bin_reduction():
    with pytest.raises(ValueError, match="requires bin_reduction='mean'"):
        DesignConfig(
            candidates="c.bed",
            model_kind="region_counts",
            bin_reduction="topk",
            targets=[DesignTarget(name="t1", target="g0")],
        )


def test_design_region_counts_rejects_explicit_topk_bins():
    with pytest.raises(ValueError, match="topk_bins"):
        DesignConfig(
            candidates="c.bed",
            model_kind="region_counts",
            topk_bins=5,
            targets=[DesignTarget(name="t1", target="g0")],
        )


def test_design_region_counts_requires_raw_gain_transform():
    with pytest.raises(ValueError, match="requires gain_transform='raw'"):
        DesignConfig(
            candidates="c.bed",
            model_kind="region_counts",
            gain_transform="log2-fold-change",
            targets=[DesignTarget(name="t1", target="g0")],
        )


def test_design_region_counts_defaults_are_compatible():
    # No raise: bin_reduction="mean", topk_bins unset, gain_transform="raw" are all defaults.
    DesignConfig(
        candidates="c.bed",
        model_kind="region_counts",
        targets=[DesignTarget(name="t1", target="g0")],
    )


def test_design_region_settings_check_is_skipped_under_auto_model_kind():
    # model_kind="auto" (the default): the checkpoint isn't readable at config-validation
    # time, so this convenience check can't fire yet even for settings that would be
    # rejected under an explicit model_kind="region_counts" -- the authoritative runtime
    # check (design.run._check_region_settings) catches it once the checkpoint is loaded.
    DesignConfig(
        candidates="c.bed",
        bin_reduction="topk",
        topk_bins=5,
        gain_transform="log2-fold-change",
        targets=[DesignTarget(name="t1", target="g0")],
    )


def test_attribution_region_counts_requires_mean_bin_reduction():
    with pytest.raises(ValueError, match="requires bin_reduction='mean'"):
        AttributionConfig(
            candidates="c.bed",
            model_kind="region_counts",
            bin_reduction="topk",
            topk_bins=5,
            targets=[AttributionTarget(name="t1", track="g0")],
        )


def test_attribution_region_counts_defaults_are_compatible():
    AttributionConfig(
        candidates="c.bed",
        model_kind="region_counts",
        targets=[AttributionTarget(name="t1", track="g0")],
    )


def test_final_phase_follows_each_runs_recipe():
    train = TrainConfig(
        recipes={
            "finetune": [
                TrainPhase(name="head", preset="head_only"),
                TrainPhase(name="deep", preset="deep_finetune"),
            ],
            **CURRICULUM,
        },
        runs=[_run(), _cached()],
    )
    assert train.final_phase("run_a") == "deep"
    assert train.final_phase("counts_run") == "pretrain"


def test_init_from_names_another_runs_phase_with_the_same_target():
    cached = _cached("cached")
    live = _run("live", recipe="curriculum", target="region_counts", init_from="cached/pretrain")
    assert TrainConfig(recipes=CURRICULUM, runs=[cached, live]).runs[1].init_from
    for init_from, match in (
        ("missing/pretrain", "must name another run's phase"),
        ("live/pretrain", "must name another run's phase"),
        ("cached/target", "is not a phase of run 'cached'"),
    ):
        bad = _run("live", recipe="curriculum", target="region_counts", init_from=init_from)
        with pytest.raises(ValueError, match=match):
            TrainConfig(recipes=CURRICULUM, runs=[cached, bad])
    profile = _run("profile", init_from="cached/pretrain")
    recipes = {**CURRICULUM, "finetune": [TrainPhase(name="head", preset="head_only")]}
    with pytest.raises(ValueError, match="has target region_counts, not profile"):
        TrainConfig(recipes=recipes, runs=[cached, profile])


def _counts_sweep_config(**sweep: Any) -> RegulonadoConfig:
    return RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
        targets=TargetsConfig(region_counts=COUNTS),
        train=TrainConfig(recipes=CURRICULUM, runs=[_cached()]),
        parameter_sweeps={"heads": {"enabled": True, "sweep_config": "sweep.yaml", **sweep}},
    )


def test_region_count_parameter_sweep_reads_cached_runs_embeddings():
    config = _counts_sweep_config(target="region_counts", embeddings_from=["counts_run"])
    assert config.parameter_sweeps["heads"].embeddings_from == ["counts_run"]
    _validate_against_schema(config.to_dict())


def test_parameter_sweep_needs_its_target_configured():
    with pytest.raises(ValueError, match="targets.profile is not configured"):
        _counts_sweep_config()


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        ({"target": "region_counts", "embeddings_from": ["missing"]}, "'missing' is not a train"),
        ({"embeddings_from": ["counts_run"]}, "embeddings_from applies to target: region_counts"),
    ],
)
def test_parameter_sweep_embeddings_must_come_from_cached_region_runs(sweep, message):
    with pytest.raises(ValueError, match=message):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(fasta="genome.fa", track_sheet="tracks.csv"),
            targets=TargetsConfig(profile=PROFILE.profile, region_counts=COUNTS),
            train=TrainConfig(recipes=CURRICULUM, runs=[_cached()]),
            parameter_sweeps={
                "heads": {"enabled": True, "sweep_config": "sweep.yaml", **sweep}
            },
        )


def test_parameter_sweep_embeddings_subset_needs_embeddings_from():
    with pytest.raises(ValueError, match="embeddings_subset needs embeddings_from"):
        _counts_sweep_config(target="region_counts", embeddings_subset=16384)


def test_parameter_sweep_names_are_directory_safe():
    with pytest.raises(ValueError, match="only letters, digits"):
        RegulonadoConfig(
            results_dir="results",
            inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
            targets=TargetsConfig(region_counts=COUNTS),
            parameter_sweeps={
                "a/b": {"enabled": True, "sweep_config": "s.yaml", "target": "region_counts"}
            },
        )


def test_disabled_parameter_sweeps_are_not_checked():
    config = RegulonadoConfig(
        results_dir="results",
        inputs=InputsConfig(fasta="genome.fa", bam_dir="bams"),
        targets=TargetsConfig(region_counts=COUNTS),
        parameter_sweeps={"old": {"sweep_config": "s.yaml", "embeddings_from": ["gone"]}},
    )
    assert not config.parameter_sweeps["old"].enabled
