from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

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
    "unfreeze_output": (2, "poisson_multinomial"),
    "deep_finetune": (4, "poisson_multinomial"),
    "peak_finetune": (2, "topk_additive"),
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
    assert cfg.trainer.persistent_workers is False


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
