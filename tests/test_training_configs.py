from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

hydra = pytest.importorskip("hydra")
omegaconf = pytest.importorskip("omegaconf")


CONFIG_DIR = Path(__file__).parents[1] / "python" / "configs"
CONFIG_MODULE_PATH = CONFIG_DIR.parent / "regulonado" / "training" / "config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("_training_config", CONFIG_MODULE_PATH)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG_MODULE
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)
TrainerConfig = CONFIG_MODULE.TrainerConfig
PRESETS = {
    "head_only": (0, "poisson_multinomial"),
    "unfreeze_output": (4, "poisson_multinomial"),
    "deep_finetune": (6, "poisson_multinomial"),
    "peak_finetune": (4, "topk_additive"),
}


def _compose(preset: str) -> Any:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="train", overrides=[f"+experiment={preset}"])


@pytest.mark.parametrize(("preset", "expected"), PRESETS.items())
def test_phase_preset_composes(preset: str, expected: tuple[int, str]) -> None:
    cfg = _compose(preset)
    unfreeze_stages, loss_name = expected

    assert cfg.backbone.name == "borzoi"
    assert cfg.head.type == "transfer_mlp"
    assert cfg.loss.name == loss_name
    assert cfg.trainer.unfreeze_backbone_stages_from_output_end == unfreeze_stages
    assert cfg.trainer.persistent_workers is True


@pytest.mark.parametrize(
    "loss_name",
    [
        "poisson_multinomial",
        "poisson_multinomial_binwise",
        "poisson_nll",
        "transfer_calibration",
    ],
)
def test_sweep_poisson_weight_override_composes_for_every_loss(loss_name: str) -> None:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "+experiment=head_only",
                f"loss={loss_name}",
                "++loss.poisson_weight=0.2",
            ],
        )

    assert cfg.loss.poisson_weight == 0.2


def test_only_supported_phase_presets_are_shipped() -> None:
    names = {path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml")}
    assert names == set(PRESETS)


def test_trainer_schema_covers_base_config() -> None:
    cfg = _compose("head_only")
    schema_fields = {field.name for field in fields(TrainerConfig)}

    assert set(cfg.trainer) <= schema_fields
    assert cfg.trainer.topk_bins == 256


def test_trainer_schema_rejects_unknown_keys() -> None:
    schema = omegaconf.OmegaConf.structured(TrainerConfig)

    with pytest.raises(omegaconf.errors.ConfigKeyError):
        omegaconf.OmegaConf.merge(schema, {"not_a_trainer_setting": 1})


def test_run_training_rejects_missing_sections_before_dataset_io() -> None:
    from regulonado.training.runner import run_training

    with pytest.raises(ValueError, match="required section"):
        run_training({})


def test_every_section_schema_is_omegaconf_structurable() -> None:
    """Every SECTION_SCHEMAS dataclass must survive OmegaConf.structured.

    ``compose.py::_validate_training_config`` runs this for every section on every
    ``regulonado pipeline`` invocation. A field type omegaconf 2.3.0 cannot structure
    (for example ``typing.Literal``) breaks every pipeline run before Snakemake is
    reached, not just the section that declares it.
    """
    from regulonado.training.compose import SECTION_SCHEMAS

    for name, schema in SECTION_SCHEMAS.items():
        omegaconf.OmegaConf.structured(schema)


@pytest.mark.parametrize(
    "loss_name",
    [path.stem for path in (CONFIG_DIR / "loss").glob("*.yaml")],
)
def test_contrast_weight_override_composes_for_every_loss(loss_name: str) -> None:
    """A bare ``loss.contrast_weight`` override must compose against any ``loss=<option>``.

    ``loss.contrast_weight`` is declared once on the top-level ``loss:`` block in
    train.yaml (merged over the selected group via ``_self_``), not per loss/*.yaml —
    this is what makes the override work for every loss family.
    """
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "+experiment=head_only",
                f"loss={loss_name}",
                "loss.contrast_weight=0.5",
                "loss.contrast_magnitude_weight=0.25",
            ],
        )
    assert cfg.loss.contrast_weight == 0.5
    assert cfg.loss.contrast_magnitude_weight == 0.25


def test_loss_config_covers_every_loss_cfg_key() -> None:
    """Every ``loss_cfg.get("...")`` key read in runner.py must be a LossConfig field.

    Prevents the ``learn_track_weights`` bug class: a key read at runtime but missing
    from the dataclass trains fine locally and then fails pipeline validation, because
    ``OmegaConf.merge(OmegaConf.structured(LossConfig), config["loss"])`` rejects it.
    """
    import re

    from regulonado.training.config import LossConfig

    runner_path = CONFIG_MODULE_PATH.parent / "runner.py"
    source = runner_path.read_text()
    read_keys = set(re.findall(r'loss_cfg\.get\(\s*"([a-zA-Z0-9_]+)"', source))
    schema_fields = {field.name for field in fields(LossConfig)}
    missing = read_keys - schema_fields
    assert not missing, f"loss_cfg keys read but missing from LossConfig: {missing}"


def test_run_training_rejects_unknown_trainer_setting_before_dataset_io() -> None:
    from regulonado.training.runner import run_training

    config = {
        "data": {},
        "backbone": {},
        "head": {},
        "model": {},
        "loss": {},
        "trainer": {"not_a_trainer_setting": 1},
    }
    with pytest.raises(ValueError, match="Invalid trainer configuration"):
        run_training(config)


@pytest.mark.parametrize(
    ("eval_steps", "checkpoint_steps", "valid"), [(150, 500, False), (250, 500, True)]
)
def test_checkpoint_steps_must_land_on_eval_steps(
    eval_steps: int, checkpoint_steps: int, valid: bool
) -> None:
    from regulonado.training.compose import resolved_training_config

    overrides = [
        "data.path=/dataset",
        f"trainer.eval_every_n_steps={eval_steps}",
        f"trainer.checkpoint_every_n_steps={checkpoint_steps}",
    ]
    if valid:
        resolved_training_config("head_only", overrides)
    else:
        with pytest.raises(ValueError, match="must be a multiple of"):
            resolved_training_config("head_only", overrides)


def test_selected_contrast_config_resolves_strict_production_settings() -> None:
    from regulonado.config.models import RegulonadoConfig
    from regulonado.training.compose import resolved_training_config
    from regulonado.training.overrides import hydra_override_items, merge_training_settings

    path = Path(__file__).parents[1] / "examples" / "contrast-correlation-deep-unfreeze.yaml"
    raw = yaml.safe_load(path.read_text())
    RegulonadoConfig.model_validate(raw)
    phase = raw["train"]["phases"][0]
    run = raw["train"]["runs"][0]
    settings = merge_training_settings(
        [raw["train"]["common"], phase["settings"], run["settings"]],
        seed=run["seed"],
        pretrained_model=run["pretrained_model"],
    )
    overrides = hydra_override_items(settings)
    first_dotted = next(index for index, item in enumerate(overrides) if item.startswith("++"))
    assert all("." not in item.split("=", 1)[0] for item in overrides[:first_dotted])
    resolved = yaml.safe_load(
        resolved_training_config(
            phase["preset"],
            ["data.path=/dataset", "output_dir=/output", *overrides],
        )
    )

    assert resolved["trainer"]["max_epochs"] == 2
    assert resolved["trainer"]["max_steps"] is None
    assert resolved["trainer"]["evals_per_epoch"] == 4
    assert resolved["trainer"]["eval_every_n_steps"] is None
    assert resolved["trainer"]["checkpoint_every_n_steps"] is None
    assert resolved["head"]["type"] == "film"
    assert resolved["head"]["output_init"] == "empirical_mean_constant"
    assert resolved["loss"]["name"] == "poisson_multinomial_binwise"
    assert resolved["loss"]["contrast_weight"] == pytest.approx(0.5)
    assert resolved["data"]["apply_squash"] is False
