"""CPU-only unit tests for the pure/module-level steps extracted from run_training.

These exercise the individual seams (config resolution, streaming guards, collate/loss
construction, history extraction, summary assembly) without loading a real dataset or
model — no GPU, no downloads.
"""
from __future__ import annotations

import json

import pytest
import torch
from regulonado.training.config import TrainerConfig
from regulonado.training.runner import (
    _build_collate_and_loss,
    _build_training_summary,
    _finalize_trainer_outputs,
    _guard_streaming_persistent_workers,
    _resolve_trainer_config,
)

MINIMAL_CFG = {
    "data": {},
    "backbone": {},
    "head": {},
    "model": {},
    "loss": {},
    "trainer": {},
}


class TestResolveTrainerConfig:
    def test_returns_trainer_config_with_defaults(self) -> None:
        trainer_cfg = _resolve_trainer_config(MINIMAL_CFG)
        assert isinstance(trainer_cfg, TrainerConfig)
        assert trainer_cfg.batch_size == 1

    def test_missing_sections_raise(self) -> None:
        with pytest.raises(ValueError, match="required section"):
            _resolve_trainer_config({})

    def test_invalid_trainer_key_raises(self) -> None:
        cfg = {**MINIMAL_CFG, "trainer": {"not_a_setting": 1}}
        with pytest.raises(ValueError, match="Invalid trainer configuration"):
            _resolve_trainer_config(cfg)

    def test_conflicting_checkpoint_options_raise(self) -> None:
        cfg = {
            **MINIMAL_CFG,
            "trainer": {
                "resume_from_checkpoint": "ckpt_a",
                "init_weights_from_checkpoint": "ckpt_b",
            },
        }
        with pytest.raises(ValueError, match="Set only one of"):
            _resolve_trainer_config(cfg)

    def test_normalises_checkpoint_mode_strings(self) -> None:
        cfg = {**MINIMAL_CFG, "trainer": {"resume_from_checkpoint": "true"}}
        trainer_cfg = _resolve_trainer_config(cfg)
        assert trainer_cfg.resume_from_checkpoint is True


class TestGuardStreamingPersistentWorkers:
    def test_disables_persistent_workers_when_streaming(self) -> None:
        trainer_cfg = TrainerConfig(persistent_workers=True)
        with pytest.warns(UserWarning, match="persistent_workers"):
            guarded = _guard_streaming_persistent_workers(trainer_cfg, streaming=True)
        assert guarded.persistent_workers is False

    def test_leaves_non_streaming_untouched(self) -> None:
        trainer_cfg = TrainerConfig(persistent_workers=True)
        guarded = _guard_streaming_persistent_workers(trainer_cfg, streaming=False)
        assert guarded.persistent_workers is True

    def test_leaves_already_disabled_untouched_when_streaming(self) -> None:
        trainer_cfg = TrainerConfig(persistent_workers=False)
        guarded = _guard_streaming_persistent_workers(trainer_cfg, streaming=True)
        assert guarded.persistent_workers is False


class TestBuildCollateAndLoss:
    RECORDS = [{}, {}]  # two tracks, all fields default via resolve_scale_and_clip fallbacks

    def test_returns_expected_shapes_and_defaults(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(
            cfg, self.RECORDS
        )
        assert scale_factors.shape == (2,)
        assert background.shape == (2,)
        assert callable(collate_fn)
        assert callable(loss_fn)

    def test_collate_fn_stacks_batch(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        collate_fn, _, _, _ = _build_collate_and_loss(cfg, self.RECORDS)
        batch = [
            {"input_ids": torch.zeros(4), "labels": torch.ones(2, 3)},
            {"input_ids": torch.zeros(4), "labels": torch.ones(2, 3)},
        ]
        collated = collate_fn(batch)
        assert collated["input_ids"].shape == (2, 4)
        assert collated["labels"].shape == (2, 2, 3)

    def test_loss_fn_computes_mse(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        _, loss_fn, _, _ = _build_collate_and_loss(cfg, self.RECORDS)
        pred = torch.zeros(2, 2, 3)
        target = torch.ones(2, 2, 3)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0)

    def test_unsupported_loss_name_raises(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "bogus"}}
        with pytest.raises(ValueError, match="Unsupported loss name"):
            _build_collate_and_loss(cfg, self.RECORDS)


class _FakeTrainerState:
    def __init__(self, log_history: list[dict]) -> None:
        self.log_history = log_history


class _FakeTrainer:
    """Stand-in for RegulonadoTrainer exposing only what _finalize_trainer_outputs uses."""

    def __init__(self, log_history: list[dict]) -> None:
        self.state = _FakeTrainerState(log_history)
        self.saved_model_dir = None
        self.state_saved = False

    def save_model(self, output_dir) -> None:
        self.saved_model_dir = output_dir

    def save_state(self) -> None:
        self.state_saved = True


class TestFinalizeTrainerOutputs:
    def test_extracts_train_and_eval_losses(self, tmp_path) -> None:
        trainer = _FakeTrainer(
            [
                {"loss": 1.0, "step": 1},
                {"eval_loss": 0.5, "step": 2},
                {"loss": 0.8, "step": 3},
            ]
        )
        history = _finalize_trainer_outputs(trainer, tmp_path)
        assert history == {"train/loss": [1.0, 0.8], "eval/loss": [0.5]}
        assert trainer.saved_model_dir == tmp_path
        assert trainer.state_saved is True

    def test_falls_back_to_train_loss_key(self, tmp_path) -> None:
        trainer = _FakeTrainer([{"train_loss": 0.42}])
        history = _finalize_trainer_outputs(trainer, tmp_path)
        assert history == {"train/loss": [0.42], "eval/loss": []}


class TestBuildTrainingSummary:
    def test_writes_and_returns_summary(self, tmp_path) -> None:
        cfg = {
            "backbone": {"name": "borzoi", "pretrained_name": None},
            "head": {"type": "transfer_mlp"},
        }
        trainer_cfg = TrainerConfig()
        history = {"train/loss": [1.0], "eval/loss": []}
        summary = _build_training_summary(
            cfg, tmp_path, 42, None, [{}, {}], trainer_cfg, history
        )
        assert summary["n_tracks"] == 2
        assert summary["seed"] == 42
        assert summary["backbone"] == "borzoi"
        assert summary["head"] == "transfer_mlp"
        assert summary["history"] == history

        written = json.loads((tmp_path / "training_summary.json").read_text())
        assert written == summary
