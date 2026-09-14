"""CPU-only unit tests for the pure/module-level steps extracted from run_training.

These exercise the individual seams (config resolution, streaming guards, collate/loss
construction, history extraction, summary assembly) without loading a real dataset or
model — no GPU, no downloads.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch
from regulonado.training.callbacks import StreamingEpochProgressCallback
from regulonado.training.config import TrainerConfig
from regulonado.training.data import count_arrow_split_rows
from regulonado.training.runner import (
    StepBudget,
    _build_collate_and_loss,
    _build_training_arguments,
    _build_training_summary,
    _empirical_track_output_bias,
    _estimate_shuffle_buffer,
    _finalize_trainer_outputs,
    _guard_streaming_persistent_workers,
    _prepare_dataset_splits,
    _resolve_empirical_output_bias,
    _resolve_step_budget,
    _resolve_trainer_config,
    _validate_dataset_schema,
)
from transformers import TrainerControl, TrainerState

MINIMAL_CFG = {
    "data": {},
    "backbone": {},
    "head": {},
    "model": {},
    "loss": {},
    "trainer": {},
}


class _DatasetColumns:
    def __init__(self, *columns: str) -> None:
        self.column_names = list(columns)


def test_validate_dataset_schema_accepts_model_inputs() -> None:
    _validate_dataset_schema(
        {"train": _DatasetColumns("input_ids", "labels", "interval")}
    )


def test_validate_dataset_schema_rejects_missing_input_ids() -> None:
    with pytest.raises(ValueError, match="missing required column.*input_ids"):
        _validate_dataset_schema({"train": _DatasetColumns("labels", "interval")})


@pytest.mark.parametrize("compression", [None, "lz4", "zstd"])
def test_count_arrow_split_rows_reads_batch_headers_across_shards(tmp_path, compression) -> None:
    shard_rows = [(1000, 300), (512, 512), (0, 10)]
    filenames = []
    for index, (n_rows, chunk) in enumerate(shard_rows):
        name = f"data-{index:05d}-of-{len(shard_rows):05d}.arrow"
        table = pa.table({"labels": pa.array([b"x" * 64] * n_rows)})
        options = ipc.IpcWriteOptions(compression=compression)
        with ipc.new_stream(str(tmp_path / name), table.schema, options=options) as writer:
            writer.write_table(table, max_chunksize=chunk)
        filenames.append(name)
    (tmp_path / "state.json").write_text(
        json.dumps({"_data_files": [{"filename": name} for name in filenames]})
    )

    assert count_arrow_split_rows(tmp_path) == 1512


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
    assert _resolve_step_budget(cfg, train_rows=None, world_size=1) == StepBudget(None, 500)


def test_step_budget_requires_max_steps_when_split_size_unknown() -> None:
    with pytest.raises(ValueError, match="trainer.max_steps"):
        _resolve_step_budget(TrainerConfig(), train_rows=None, world_size=1)


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


def test_streaming_epoch_progress_uses_steps_per_epoch() -> None:
    state = TrainerState(global_step=500, epoch=1.0)

    StreamingEpochProgressCallback(steps_per_epoch=3_474).on_step_end(None, state, TrainerControl())

    assert state.epoch == pytest.approx(500 / 3_474)


def test_streaming_eval_cap_strides_across_every_validation_shard(tmp_path) -> None:
    from datasets import Dataset, load_dataset

    Dataset.from_dict({"local_index": list(range(30))}).save_to_disk(
        str(tmp_path / "validation"), num_shards=3
    )
    (tmp_path / "dataset_dict.json").write_text(json.dumps({"splits": ["validation"]}))
    dataset_dict = dict(load_dataset(str(tmp_path), streaming=True))

    prepared = _prepare_dataset_splits(
        dataset_dict,
        {},
        TrainerConfig(max_eval_samples=6),
        {},
        data_path=tmp_path,
        streaming=True,
        seed=0,
    )

    selected = [row["local_index"] for row in prepared["validation"]]
    assert selected == [0, 5, 10, 15, 20, 25]


def test_streaming_train_examples_own_their_arrays_before_shuffle(tmp_path) -> None:
    from datasets import Array2D, Dataset, Features, Value, load_dataset

    features = Features(
        {
            "input_ids": Array2D(dtype="int8", shape=(4, 8)),
            "labels": Array2D(dtype="float32", shape=(2, 3)),
            "interval": Value("string"),
        }
    )
    Dataset.from_dict(
        {
            "input_ids": [np.eye(4, 8, dtype=np.int8)] * 6,
            "labels": [np.ones((2, 3), dtype=np.float32)] * 6,
            "interval": [f"chr1:{i}" for i in range(6)],
        },
        features=features,
    ).save_to_disk(str(tmp_path / "train"), num_shards=2)
    (tmp_path / "dataset_dict.json").write_text(json.dumps({"splits": ["train"]}))
    dataset_dict = dict(load_dataset(str(tmp_path), streaming=True))
    data_cfg = {"shuffle_buffer_ram_gb": 1.0, "context_length": 8, "n_pred_bins": 3}

    prepared = _prepare_dataset_splits(
        dataset_dict, data_cfg, TrainerConfig(), {}, data_path=tmp_path, streaming=True, seed=0
    )

    examples = list(prepared["train"])
    assert len(examples) == 6
    for example in examples:
        assert example["input_ids"].dtype == np.int8
        assert example["input_ids"].flags.owndata
        assert example["labels"].shape == (2, 3)
        assert example["labels"].flags.owndata


def test_shuffle_buffer_sizes_examples_by_stored_dtypes() -> None:
    data_cfg = {"shuffle_buffer_ram_gb": 8.0, "context_length": 100, "n_pred_bins": 10}

    # int8 one-hot sequence (4 x 100 bytes) + float32 labels (2 x 10 x 4 bytes) = 480 bytes.
    assert _estimate_shuffle_buffer(data_cfg, {"n_tracks": 2}) == int(8e9 / 480)


def test_streaming_shuffle_ram_budget_is_divided_across_workers() -> None:
    data_cfg = {"shuffle_buffer_ram_gb": 8.0, "context_length": 100, "n_pred_bins": 10}
    metadata = {"n_tracks": 2}

    single_process = _estimate_shuffle_buffer(data_cfg, metadata, num_workers=0)
    four_workers = _estimate_shuffle_buffer(data_cfg, metadata, num_workers=4)

    assert four_workers == single_process // 4


def test_empirical_track_output_bias_matches_track_means_through_softplus() -> None:
    dataset = [
        {"labels": np.array([[1.0, 3.0], [2.0, 4.0]])},
        {"labels": np.array([[5.0, 7.0], [6.0, 8.0]])},
    ]

    bias = _empirical_track_output_bias(
        dataset, n_tracks=2, activation_type="softplus", max_samples=2
    )

    torch.testing.assert_close(
        torch.nn.functional.softplus(torch.tensor(bias)), torch.tensor([4.0, 5.0])
    )


def test_empirical_track_output_bias_respects_sample_limit_and_transposed_labels() -> None:
    dataset = [
        {"labels": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        {"labels": np.full((3, 2), 100.0)},
    ]

    bias = _empirical_track_output_bias(dataset, n_tracks=2, activation_type="exp", max_samples=1)

    torch.testing.assert_close(torch.exp(torch.tensor(bias)), torch.tensor([3.0, 4.0]))


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

    _resolve_empirical_output_bias(cfg, dataset, n_tracks=2)

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
        collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(cfg, self.RECORDS)
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
        summary = _build_training_summary(cfg, tmp_path, 42, None, [{}, {}], trainer_cfg, history)
        assert summary["n_tracks"] == 2
        assert summary["seed"] == 42
        assert summary["backbone"] == "borzoi"
        assert summary["head"] == "transfer_mlp"
        assert summary["history"] == history

        written = json.loads((tmp_path / "training_summary.json").read_text())
        assert written == summary
