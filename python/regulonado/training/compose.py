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
    # With evaluation and checkpoints both enabled the trainer loads the best model at the
    # end, which transformers only allows when checkpoints land on evaluation steps.
    # Otherwise this surfaces as a ValueError after a GPU job has already started.
    eval_steps = config.trainer.get("eval_every_n_steps")
    checkpoint_steps = config.trainer.get("checkpoint_every_n_steps")
    if eval_steps and checkpoint_steps and checkpoint_steps % eval_steps != 0:
        raise ValueError(
            f"trainer.checkpoint_every_n_steps ({checkpoint_steps}) must be a multiple of "
            f"trainer.eval_every_n_steps ({eval_steps}); the best model can only be "
            "restored from a checkpoint saved at an evaluation step"
        )


def resolved_training_config(preset: str, overrides: list[str]) -> str:
    """Compose a preset without importing the model or training stack."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"+experiment={preset}", *overrides])
    _validate_training_config(config)
    if OmegaConf.is_interpolation(config, "output_dir"):
        config.output_dir = "<automatic run directory>"
    return OmegaConf.to_yaml(config, resolve=True)
