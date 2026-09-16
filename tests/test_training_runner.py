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
from accelerate.data_loader import BatchSamplerShard
from regulonado.training.config import TrainerConfig
from regulonado.training.data import WindowParquetDataset
from regulonado.training.runner import (
    RegulonadoTrainer,
    _build_collate_and_loss,
    _build_scheduler_for_trainer,
    _build_training_arguments,
    _build_training_summary,
    _empirical_track_output_bias,
    _finalize_trainer_outputs,
    _prepare_dataset_splits,
    _resolve_empirical_output_bias,
    _resolve_trainer_config,
    _resolve_training_schedule,
    _validate_dataset_schema,
)
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, Subset
from transformers import Trainer

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


def _schedule(cfg: TrainerConfig, *, rows: int = 41_699, world_size: int = 1):
    return _resolve_training_schedule(cfg, train_rows=rows, world_size=world_size, has_eval=True)


def test_schedule_counts_optimizer_steps_per_epoch() -> None:
    cfg = TrainerConfig(batch_size=12, gradient_accumulation_steps=1, max_epochs=2)

    schedule = _schedule(cfg)

    assert schedule.steps_per_epoch == 3_474
    assert schedule.total_steps == 6_948
    assert schedule.explicit_max_steps is None
    assert schedule.effective_global_batch == 12
    assert schedule.dropped_rows_per_epoch == 11


def test_schedule_divides_by_global_batch_and_accumulation() -> None:
    cfg = TrainerConfig(batch_size=4, gradient_accumulation_steps=3, max_epochs=1)

    # 1003 rows // (4 * 2 GPUs) = 125 batches -> ceil(125 / 3) = 42 updates.
    schedule = _schedule(cfg, rows=1_003, world_size=2)

    assert schedule.micro_batches_per_rank == 125
    assert schedule.steps_per_epoch == 42
    assert schedule.total_steps == 42
    assert schedule.effective_global_batch == 24
    assert schedule.dropped_rows_per_epoch == 3


@pytest.mark.parametrize(
    ("rows", "batch_size", "world_size", "accumulation", "max_steps"),
    [
        (96, 4, 1, 1, None),
        (99, 4, 1, 3, None),
        (96, 4, 2, 3, None),
        (101, 4, 2, 3, 5),
    ],
)
def test_schedule_matches_accelerate_and_trainer_lengths(
    tmp_path, rows, batch_size, world_size, accumulation, max_steps
) -> None:
    """Compare our arithmetic to the installed Accelerate and Trainer implementations."""
    cfg = TrainerConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_epochs=2,
        max_steps=max_steps,
        mixed_precision="no",
    )
    schedule = _schedule(cfg, rows=rows, world_size=world_size)
    sampler = SequentialSampler(range(rows))
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    if world_size > 1:
        batch_sampler = BatchSamplerShard(
            batch_sampler,
            num_processes=world_size,
            process_index=0,
            split_batches=False,
            even_batches=True,
        )
    dataloader = DataLoader(range(rows), batch_sampler=batch_sampler)
    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=True, seed=0)
    trainer = Trainer(model=torch.nn.Linear(1, 1), args=args, eval_dataset=range(rows))
    values = trainer.set_initial_training_values(args, dataloader)

    assert len(dataloader) == schedule.micro_batches_per_rank
    assert values[1] == schedule.steps_per_epoch
    assert values[-1] == schedule.total_steps


def test_schedule_keeps_explicit_max_steps() -> None:
    cfg = TrainerConfig(batch_size=12, max_steps=500)

    schedule = _schedule(cfg)

    assert schedule.steps_per_epoch == 3_474
    assert schedule.total_steps == 500
    assert schedule.explicit_max_steps == 500


def test_schedule_rejects_split_smaller_than_global_batch() -> None:
    with pytest.raises(ValueError, match="global batch"):
        _schedule(TrainerConfig(batch_size=12), rows=11)


def test_epoch_driven_training_evaluates_once_per_epoch(tmp_path) -> None:
    cfg = TrainerConfig(batch_size=12, mixed_precision="no", checkpoint_every_n_steps=None)
    schedule = _schedule(cfg)

    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=True, seed=0)

    assert args.max_steps == -1
    assert args.num_train_epochs == 1
    assert args.eval_steps == 3_474


def test_evals_per_epoch_evaluates_and_checkpoints_each_quarter_epoch(tmp_path) -> None:
    cfg = TrainerConfig(batch_size=12, mixed_precision="no", evals_per_epoch=4, max_epochs=2)
    schedule = _schedule(cfg)

    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=True, seed=0)

    assert args.eval_steps == 868
    assert args.save_strategy == "steps"
    assert args.save_steps == 868
    assert args.load_best_model_at_end
    assert schedule.evaluation_events == (
        0,
        868,
        1_736,
        2_604,
        3_472,
        4_340,
        5_208,
        6_076,
        6_944,
    )
    assert schedule.checkpoint_events == schedule.evaluation_events[1:]


def test_evaluation_keeps_the_final_partial_batch(tmp_path) -> None:
    cfg = TrainerConfig(batch_size=4, eval_batch_size=4, mixed_precision="no")
    schedule = _schedule(cfg, rows=10)
    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=True, seed=0)
    rows = [{"input_ids": torch.tensor([float(index)])} for index in range(10)]
    trainer = RegulonadoTrainer(
        model=torch.nn.Linear(1, 1),
        args=args,
        train_dataset=rows,
        eval_dataset=rows,
    )

    assert len(trainer.get_train_dataloader()) == 2
    assert len(trainer.get_eval_dataloader()) == 3


def test_two_epoch_trainer_run_matches_schedule_and_scheduler(tmp_path) -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1, 1)

        def forward(self, input_ids, labels=None, **_):
            logits = self.projection(input_ids)
            loss = torch.nn.functional.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}

    cfg = TrainerConfig(
        batch_size=4,
        num_workers=0,
        max_epochs=2,
        mixed_precision="no",
        log_every_n_steps=1,
    )
    schedule = _resolve_training_schedule(cfg, train_rows=10, world_size=1, has_eval=False)
    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=False, seed=0)
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = _build_scheduler_for_trainer(optimizer, cfg, schedule)
    dataset = [
        {"input_ids": torch.tensor([float(index)]), "labels": torch.tensor([0.0])}
        for index in range(10)
    ]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    assert schedule.steps_per_epoch == 2
    assert schedule.total_steps == 4
    assert trainer.state.global_step == schedule.total_steps
    assert trainer.state.max_steps == schedule.total_steps
    assert scheduler.last_epoch == schedule.total_steps


def test_explicit_eval_interval_overrides_evals_per_epoch(tmp_path) -> None:
    cfg = TrainerConfig(
        mixed_precision="no", evals_per_epoch=4, eval_every_n_steps=250, max_steps=500
    )
    schedule = _schedule(cfg)

    args = _build_training_arguments(tmp_path, cfg, schedule, has_eval=True, seed=0)

    assert args.eval_steps == 250
    assert args.save_steps == 250
    assert args.max_steps == 500


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("gradient_accumulation_steps", 0), ("max_epochs", 0)],
)
def test_schedule_rejects_non_positive_settings(field: str, value: int) -> None:
    cfg = TrainerConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        _schedule(cfg)


def test_schedule_rejects_warmup_longer_than_run() -> None:
    cfg = TrainerConfig(max_steps=5, warmup_steps=6)

    with pytest.raises(ValueError, match="warmup_steps"):
        _schedule(cfg)


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
        # 3 groups, not 2: with exactly 2 members the centred deviations are always
        # exactly +/- symmetric, so r is always exactly -1 regardless of what the loss
        # actually computes -- a test that would pass for the wrong reason.
        records = [
            {"assay_class": "ATAC", "group": "a"},
            {"assay_class": "ATAC", "group": "b"},
            {"assay_class": "ATAC", "group": "c"},
        ]
        loss_cfg = {"name": "mse", "contrast_weight": 1.0}
        cfg = {
            "model": {"use_track_metadata": False},
            "data": {},
            "loss": loss_cfg,
            "trainer": {"contrast_region_bins": 2, "contrast_active_fraction": 1.0},
        }
        _, loss_fn, _, _ = _build_collate_and_loss(cfg, records)
        torch.manual_seed(0)
        target = torch.rand(4, 3, 8) + 0.5
        uncorrelated_pred = torch.rand(4, 3, 8) + 0.5
        assert loss_fn(uncorrelated_pred, target).item() > loss_fn(target, target).item() + 0.1
        assert loss_fn(target, target).item() == pytest.approx(0.0, abs=1e-4)

    def test_contrast_shape_params_come_from_trainer_config(self) -> None:
        records = [
            {"assay_class": "ATAC", "group": "a"},
            {"assay_class": "ATAC", "group": "b"},
            {"assay_class": "ATAC", "group": "c"},
        ]
        loss_cfg = {"name": "mse", "contrast_weight": 1.0}
        torch.manual_seed(1)
        pred = torch.rand(4, 3, 8) + 0.5
        target = torch.rand(4, 3, 8) + 0.5
        cfg_a = {
            "model": {"use_track_metadata": False},
            "data": {},
            "loss": loss_cfg,
            "trainer": {"contrast_region_bins": 2},
        }
        cfg_b = {
            "model": {"use_track_metadata": False},
            "data": {},
            "loss": loss_cfg,
            "trainer": {"contrast_region_bins": 4},
        }
        _, loss_fn_a, _, _ = _build_collate_and_loss(cfg_a, records)
        _, loss_fn_b, _, _ = _build_collate_and_loss(cfg_b, records)
        assert loss_fn_a(pred, target).item() != pytest.approx(loss_fn_b(pred, target).item())

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
        summary = _build_training_summary(
            cfg, tmp_path, 42, None, [{}, {}], trainer_cfg, history, {"test_loss": 0.5}
        )
        assert summary["n_tracks"] == 2
        assert summary["seed"] == 42
        assert summary["backbone"] == "borzoi"
        assert summary["head"] == "transfer_mlp"
        assert summary["history"] == history
        assert summary["test_metrics"] == {"test_loss": 0.5}

        written = json.loads((tmp_path / "training_summary.json").read_text())
        assert written == summary


@pytest.mark.parametrize(("saved_log_var", "target_log_var"), [(True, False), (False, True)])
def test_warm_start_tolerates_learned_track_weight_mismatch(
    tmp_path, saved_log_var: bool, target_log_var: bool
) -> None:
    from regulonado.training.runner import load_model_weights_only
    from safetensors.torch import save_file

    source = torch.nn.Linear(2, 2)
    if saved_log_var:
        source.track_loss_log_var = torch.nn.Parameter(torch.full((2,), 0.5))
    save_file(dict(source.state_dict()), str(tmp_path / "model.safetensors"))

    target = torch.nn.Linear(2, 2)
    if target_log_var:
        target.track_loss_log_var = torch.nn.Parameter(torch.zeros(2))
    load_model_weights_only(target, tmp_path)

    torch.testing.assert_close(target.weight, source.weight)
    if target_log_var:
        torch.testing.assert_close(target.track_loss_log_var, torch.zeros(2))


def _first_epoch_order(tmp_path: Path, seed: int) -> list[int]:
    cfg = TrainerConfig(batch_size=1, num_workers=0, max_epochs=1, mixed_precision="no")
    schedule = _resolve_training_schedule(cfg, train_rows=32, world_size=1, has_eval=False)
    args = _build_training_arguments(tmp_path / str(seed), cfg, schedule, has_eval=False, seed=seed)
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        args=args,
        train_dataset=list(range(32)),
        data_collator=lambda batch: batch,
    )
    torch.manual_seed(args.seed)  # train() re-seeds from args.seed before iterating
    return [int(batch[0]) for batch in trainer.get_train_dataloader()]


def test_training_arguments_carry_run_seed_so_folds_shuffle_differently(tmp_path):
    # Trainer re-seeds from TrainingArguments.seed; defaulting it to 42 gave every fold
    # the same shuffle order, dropout masks, and augmentation draws.
    assert _first_epoch_order(tmp_path, 1) == _first_epoch_order(tmp_path, 1)
    assert _first_epoch_order(tmp_path, 1) != _first_epoch_order(tmp_path, 2)
