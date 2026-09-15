"""CPU-only unit tests for the pure/module-level steps extracted from run_training.

These exercise the individual seams (config resolution, dataset access, collate/loss
construction, history extraction, summary assembly) without loading a real dataset or
model — no GPU, no downloads.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from regulonado.training.config import TrainerConfig
from regulonado.training.data import WindowParquetDataset
from regulonado.training.runner import (
    StepBudget,
    _build_collate_and_loss,
    _build_training_arguments,
    _build_training_summary,
    _empirical_track_output_bias,
    _finalize_trainer_outputs,
    _prepare_dataset_splits,
    _resolve_empirical_output_bias,
    _resolve_step_budget,
    _resolve_trainer_config,
    _validate_dataset_schema,
)
from torch.utils.data import Subset

MINIMAL_CFG = {
    "data": {},
    "backbone": {},
    "head": {},
    "model": {},
    "loss": {},
    "trainer": {},
}


def _write_shard(
    path: Path,
    *,
    start: int,
    n_rows: int,
    context: int = 4,
    n_tracks: int = 2,
    n_bins: int = 3,
    rows_per_row_group: int = 1,
) -> None:
    """Write a tiny HF-layout Parquet shard directly with pyarrow, for dataset fixtures."""
    seq = [[(start + i) % 4] * context for i in range(n_rows)]
    sig = [
        [
            [float((start + i) * 100 + t * n_bins + b) for b in range(n_bins)]
            for t in range(n_tracks)
        ]
        for i in range(n_rows)
    ]
    table = pa.table(
        {
            "sequence_tokens": pa.array(seq, type=pa.list_(pa.uint8(), context)),
            "signal": pa.array(sig, type=pa.list_(pa.list_(pa.float32(), n_bins), n_tracks)),
            "interval": pa.array([f"chr1:{start + i}" for i in range(n_rows)]),
            "index": pa.array(list(range(start, start + n_rows)), type=pa.int64()),
            "local_index": pa.array(list(range(n_rows)), type=pa.int64()),
        }
    )
    pq.write_table(table, str(path), row_group_size=rows_per_row_group)


def _write_split(data_dir: Path, split: str, shard_row_counts: list[int]) -> None:
    (data_dir / "data").mkdir(parents=True, exist_ok=True)
    n_shards = len(shard_row_counts)
    start = 0
    for shard_idx, n_rows in enumerate(shard_row_counts):
        name = f"{split}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        _write_shard(data_dir / "data" / name, start=start, n_rows=n_rows)
        start += n_rows


class _FakeSchema:
    def __init__(self, *names: str) -> None:
        self.names = list(names)


class _FakeParquetDataset:
    def __init__(self, *names: str) -> None:
        self.schema = _FakeSchema(*names)


def test_validate_dataset_schema_accepts_model_inputs() -> None:
    _validate_dataset_schema(
        {"train": _FakeParquetDataset("sequence_tokens", "signal", "interval")}
    )


def test_validate_dataset_schema_rejects_missing_sequence_tokens() -> None:
    with pytest.raises(ValueError, match="missing required column.*sequence_tokens"):
        _validate_dataset_schema({"train": _FakeParquetDataset("signal", "interval")})


class TestWindowParquetDataset:
    def test_random_access_across_shards_first_last_and_boundary(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [3, 2])

        dataset = WindowParquetDataset(tmp_path, "train")

        assert len(dataset) == 5
        assert dataset[0]["interval"] == "chr1:0"
        # Boundary: last row of shard 0, first row of shard 1.
        assert dataset[2]["interval"] == "chr1:2"
        assert dataset[3]["interval"] == "chr1:3"
        assert dataset[4]["interval"] == "chr1:4"
        assert dataset[0]["sequence_tokens"].dtype == np.uint8
        assert dataset[0]["signal"].dtype == np.float32
        assert dataset[0]["signal"].shape == (2, 3)

    def test_negative_index_wraps_like_a_sequence(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [3, 2])
        dataset = WindowParquetDataset(tmp_path, "train")

        assert dataset[-1]["interval"] == dataset[4]["interval"]

    def test_index_error_out_of_range(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [3, 2])
        dataset = WindowParquetDataset(tmp_path, "train")

        with pytest.raises(IndexError):
            dataset[5]
        with pytest.raises(IndexError):
            dataset[-6]

    def test_missing_split_raises_file_not_found(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [3])

        with pytest.raises(FileNotFoundError, match="regulonado dataset"):
            WindowParquetDataset(tmp_path, "validation")

    def test_pickle_round_trip_with_no_open_handles(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [3, 2])
        dataset = WindowParquetDataset(tmp_path, "train")
        # Touch the dataset before pickling so any opened file handles must be dropped.
        _ = dataset[0]
        assert dataset._files

        restored = pickle.loads(pickle.dumps(dataset))

        assert restored._files == {}
        assert len(restored) == len(dataset)
        assert restored[4]["interval"] == dataset[4]["interval"]

    def test_transform_is_applied_per_example(self, tmp_path) -> None:
        _write_split(tmp_path, "train", [2])
        dataset = WindowParquetDataset(
            tmp_path, "train", transform=lambda example: {"input_ids": example["sequence_tokens"]}
        )

        assert set(dataset[0]) == {"input_ids"}

    def test_handles_multiple_rows_per_row_group(self, tmp_path) -> None:
        (tmp_path / "data").mkdir()
        _write_shard(
            tmp_path / "data" / "train-00000-of-00001.parquet",
            start=0,
            n_rows=6,
            rows_per_row_group=3,
        )
        dataset = WindowParquetDataset(tmp_path, "train")

        assert len(dataset) == 6
        assert [dataset[i]["interval"] for i in range(6)] == [f"chr1:{i}" for i in range(6)]


def test_step_budget_counts_optimizer_steps_per_epoch() -> None:
    cfg = TrainerConfig(batch_size=12, gradient_accumulation_steps=1, max_epochs=2)

    budget = _resolve_step_budget(cfg, train_rows=41_699, world_size=1)

    assert budget == StepBudget(steps_per_epoch=3_474, max_steps=6_948)


def test_step_budget_divides_by_global_batch_and_accumulation() -> None:
    cfg = TrainerConfig(batch_size=4, gradient_accumulation_steps=3, max_epochs=1)

    # 1003 rows // (4 * 2 GPUs) = 125 batches -> ceil(125 / 3) = 42 updates.
    budget = _resolve_step_budget(cfg, train_rows=1_003, world_size=2)

    assert budget == StepBudget(steps_per_epoch=42, max_steps=42)


def test_step_budget_keeps_explicit_max_steps() -> None:
    cfg = TrainerConfig(batch_size=12, max_steps=500)

    assert _resolve_step_budget(cfg, train_rows=41_699, world_size=1) == StepBudget(3_474, 500)


def test_step_budget_rejects_split_smaller_than_global_batch() -> None:
    with pytest.raises(ValueError, match="global batch"):
        _resolve_step_budget(TrainerConfig(batch_size=12), train_rows=11, world_size=1)


def test_epoch_driven_training_evaluates_once_per_epoch(tmp_path) -> None:
    cfg = TrainerConfig(mixed_precision="no", checkpoint_every_n_steps=None)

    args = _build_training_arguments(
        tmp_path, cfg, StepBudget(steps_per_epoch=3_474, max_steps=6_948), has_eval=True
    )

    assert args.max_steps == 6_948
    assert args.eval_steps == 3_474


def test_evals_per_epoch_evaluates_and_checkpoints_each_quarter_epoch(tmp_path) -> None:
    cfg = TrainerConfig(mixed_precision="no", evals_per_epoch=4, max_epochs=2)

    args = _build_training_arguments(
        tmp_path, cfg, StepBudget(steps_per_epoch=3_474, max_steps=6_948), has_eval=True
    )

    assert args.eval_steps == 868
    assert args.save_strategy == "steps"
    assert args.save_steps == 868
    assert args.load_best_model_at_end


def test_explicit_eval_interval_overrides_evals_per_epoch(tmp_path) -> None:
    cfg = TrainerConfig(
        mixed_precision="no", evals_per_epoch=4, eval_every_n_steps=250, max_steps=500
    )

    args = _build_training_arguments(
        tmp_path, cfg, StepBudget(steps_per_epoch=3_474, max_steps=500), has_eval=True
    )

    assert args.eval_steps == 250
    assert args.save_steps == 250


def test_evals_per_epoch_requires_known_epoch_size(tmp_path) -> None:
    cfg = TrainerConfig(mixed_precision="no", evals_per_epoch=4, max_steps=500)

    with pytest.raises(ValueError, match="evals_per_epoch"):
        _build_training_arguments(
            tmp_path, cfg, StepBudget(steps_per_epoch=None, max_steps=500), has_eval=True
        )


def test_seeded_eval_cap_selects_a_fixed_subset(tmp_path) -> None:
    validation = list(range(30))
    dataset_dict = {"validation": validation}

    prepared = _prepare_dataset_splits(dataset_dict, TrainerConfig(max_eval_samples=6), seed=0)

    assert isinstance(prepared["validation"], Subset)
    assert len(prepared["validation"]) == 6
    # Deterministic for a fixed seed.
    again = _prepare_dataset_splits(dataset_dict, TrainerConfig(max_eval_samples=6), seed=0)
    assert list(prepared["validation"]) == list(again["validation"])


def test_seeded_eval_cap_is_a_no_op_below_the_limit(tmp_path) -> None:
    validation = list(range(4))
    dataset_dict = {"validation": validation}

    prepared = _prepare_dataset_splits(dataset_dict, TrainerConfig(max_eval_samples=6), seed=0)

    assert prepared["validation"] is validation


def test_empirical_track_output_bias_matches_track_means_through_softplus() -> None:
    dataset = [
        {"labels": np.array([[1.0, 3.0], [2.0, 4.0]])},
        {"labels": np.array([[5.0, 7.0], [6.0, 8.0]])},
    ]

    bias = _empirical_track_output_bias(
        dataset, n_tracks=2, activation_type="softplus", max_samples=2, seed=0
    )

    torch.testing.assert_close(
        torch.nn.functional.softplus(torch.tensor(bias)), torch.tensor([4.0, 5.0])
    )


def test_empirical_track_output_bias_respects_sample_limit() -> None:
    dataset = [
        {"labels": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        {"labels": np.full((3, 2), 100.0)},
    ]

    bias = _empirical_track_output_bias(
        dataset, n_tracks=2, activation_type="exp", max_samples=1, seed=0
    )

    # With max_samples < len(dataset), only one of the two examples contributes.
    exp_bias = torch.exp(torch.tensor(bias))
    assert exp_bias.tolist() in (
        pytest.approx([2.0, 3.0]),
        pytest.approx([100.0, 100.0]),
    )


def test_empirical_track_output_bias_sampler_does_not_only_draw_the_first_rows() -> None:
    """Chromosome-major shard order means the first N rows are all chr1 — sampling must
    not systematically prefer them."""
    n_total = 200
    # Track 0 encodes the row's position so we can tell which rows were sampled.
    dataset = [{"labels": np.array([[float(i)], [1.0]])} for i in range(n_total)]

    drawn_indices: set[int] = set()
    for seed in range(20):
        bias = _empirical_track_output_bias(
            dataset, n_tracks=2, activation_type="identity", max_samples=1, seed=seed
        )
        drawn_indices.add(int(round(bias[0])))

    # Seeded random sampling across many seeds should reach well beyond the first rows.
    assert max(drawn_indices) > 20


@pytest.mark.parametrize(
    ("mode", "zero_weights"),
    [("empirical_mean_bias", False), ("empirical_mean_constant", True)],
)
def test_resolve_empirical_output_initialization(mode: str, zero_weights: bool) -> None:
    cfg = {
        "head": {"output_init": mode, "output_init_samples": 1},
        "model": {"activation_type": "softplus"},
    }
    dataset = {"train": [{"labels": np.array([[1.0, 3.0], [2.0, 4.0]])}]}

    _resolve_empirical_output_bias(cfg, dataset, n_tracks=2, seed=0)

    assert cfg["head"]["zero_output_weights"] is zero_weights
    resolved = torch.tensor(cfg["head"]["resolved_output_bias"])
    torch.testing.assert_close(torch.nn.functional.softplus(resolved), torch.tensor([2.0, 3.0]))


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


class TestBuildCollateAndLoss:
    RECORDS = [{}, {}]  # two tracks, all fields default via resolve_scale_and_clip fallbacks

    def test_returns_expected_shapes_and_defaults(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(cfg, self.RECORDS)
        assert scale_factors.shape == (2,)
        assert background.shape == (2,)
        assert callable(collate_fn)
        assert callable(loss_fn)

    def test_collate_fn_stacks_batch_and_keeps_input_ids_uint8(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        collate_fn, _, _, _ = _build_collate_and_loss(cfg, self.RECORDS)
        batch = [
            {"input_ids": np.zeros(4, dtype=np.uint8), "labels": torch.ones(2, 3)},
            {"input_ids": np.ones(4, dtype=np.uint8), "labels": torch.ones(2, 3)},
        ]
        collated = collate_fn(batch)
        assert collated["input_ids"].shape == (2, 4)
        assert collated["input_ids"].dtype == torch.uint8
        assert collated["labels"].shape == (2, 2, 3)

    def test_loss_fn_computes_mse(self) -> None:
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": {"name": "mse"}}
        _, loss_fn, _, _ = _build_collate_and_loss(cfg, self.RECORDS)
        pred = torch.zeros(2, 2, 3)
        target = torch.ones(2, 2, 3)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0)

    def test_contrast_weight_adds_cross_track_term(self) -> None:
        records = [
            {"assay_class": "ATAC", "group": "a"},
            {"assay_class": "ATAC", "group": "b"},
        ]
        loss_cfg = {"name": "mse", "contrast_weight": 1.0, "contrast_region_bins": 2}
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": loss_cfg}
        _, loss_fn, _, _ = _build_collate_and_loss(cfg, records)
        target = torch.tensor([[[4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0]]])
        shared = torch.full_like(target, 2.5)
        mse = torch.nn.functional.mse_loss(shared, target).item()
        assert loss_fn(shared, target).item() > mse + 0.01
        assert loss_fn(target, target).item() == pytest.approx(0.0, abs=1e-5)

    def test_contrast_weight_without_track_labels_raises(self) -> None:
        loss_cfg = {"name": "mse", "contrast_weight": 1.0}
        cfg = {"model": {"use_track_metadata": False}, "data": {}, "loss": loss_cfg}
        with pytest.raises(ValueError, match="contrast_weight"):
            _build_collate_and_loss(cfg, self.RECORDS)

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
        summary = _build_training_summary(cfg, tmp_path, 42, None, [{}, {}], trainer_cfg, history)
        assert summary["n_tracks"] == 2
        assert summary["seed"] == 42
        assert summary["backbone"] == "borzoi"
        assert summary["head"] == "transfer_mlp"
        assert summary["history"] == history

        written = json.loads((tmp_path / "training_summary.json").read_text())
        assert written == summary
