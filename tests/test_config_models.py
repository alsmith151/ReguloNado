"""The pydantic config models and the workflow's JSON Schema must not drift apart."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from regulonado.config.models import (
    BackboneConfig,
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
        backbone=BackboneConfig(type=kwargs.pop("type", "borzoi"), pretrained=pretrained),
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
        parameter_sweep={
            "enabled": True,
            "sweep_config": "sweep.yaml",
            "agents": 4,
            "trials_per_agent": 2,
            "cpus_per_agent": 4,
            "mem_mb_per_agent": 64_000,
            "runtime_minutes_per_agent": 240,
        },
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


def test_recipe_presets_must_suit_the_run_trunk():
    with pytest.raises(ValueError, match="not trunk-cached presets"):
        TrainConfig(
            recipes={"finetune": [TrainPhase(name="head", preset="head_only")]},
            runs=[_cached(recipe="finetune")],
        )
    with pytest.raises(ValueError, match="not trunk-live presets"):
        TrainConfig(recipes=CURRICULUM, runs=[_run(recipe="curriculum")])


def test_runs_must_name_a_known_recipe():
    with pytest.raises(ValueError, match="names recipe 'missing'"):
        TrainConfig(recipes=CURRICULUM, runs=[_cached(recipe="missing")])


def test_unimplemented_trunk_target_pairs_are_rejected():
    with pytest.raises(ValueError, match="trunk 'cached' with target 'profile' is not implemented"):
        _run(trunk="cached", target="profile")
    with pytest.raises(ValueError, match="trunk 'live' with target 'region_counts'"):
        _run(target="region_counts")


def test_cache_settings_apply_to_cached_alphagenome_runs_only():
    with pytest.raises(ValueError, match="'cache' only applies to trunk: cached"):
        _run(cache=TrunkCacheConfig(pool_to=128))
    with pytest.raises(ValueError, match="only apply to backbone 'alphagenome'"):
        _cached(type="borzoi", pretrained="model/a", cache=TrunkCacheConfig(context=524_288))
    with pytest.raises(ValueError, match="target_group applies to trunk: cached"):
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
