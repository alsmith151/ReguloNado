from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from regulonado.training.config import (
    BackboneConfig,
    DataConfig,
    HeadConfig,
    LossConfig,
    ModelConfig,
    TrainerConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

SECTION_SCHEMAS = {
    "data": DataConfig,
    "model": ModelConfig,
    "backbone": BackboneConfig,
    "head": HeadConfig,
    "loss": LossConfig,
    "trainer": TrainerConfig,
}


def _validate_training_config(config: DictConfig) -> None:
    """Type-check the resolved config before any training resources are started."""
    if type(config.seed) is not int:
        raise ValueError(f"seed must be an integer, got {config.seed!r}")
    for section, schema in SECTION_SCHEMAS.items():
        # Structured OmegaConf schemas reject unknown keys and values that cannot
        # be converted to the declared type (for example batch_size=oops).
        validated = OmegaConf.merge(OmegaConf.structured(schema), config[section])
        OmegaConf.to_object(validated)


def resolved_training_config(preset: str, overrides: list[str]) -> str:
    """Compose a preset without importing the model or training stack."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"+experiment={preset}", *overrides])
    _validate_training_config(config)
    if OmegaConf.is_interpolation(config, "output_dir"):
        config.output_dir = "<automatic run directory>"
    return OmegaConf.to_yaml(config, resolve=True)
