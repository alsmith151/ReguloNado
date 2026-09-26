"""``regulonado.training.cached.runner``: optimizer weight-decay exclusion, warm start, and a CPU
end-to-end cached-trunk training run on a tiny synthetic cache."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from regulonado.counts.dataset import RegionCountData
from regulonado.embeddings.cache import region_table_hash
from regulonado.training.cached.model import RegionCountConfig, RegionCountModel
from regulonado.training.cached.runner import _build_optimizer, _load_warm_start, run_training
from regulonado.training.config import TrainerConfig


def _toy_region_data(n_per_split: int = 8) -> RegionCountData:
    splits = ["train"] * n_per_split + ["val"] * max(4, n_per_split // 2)
    n = len(splits)
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"] * n,
            "start": [i * 2000 for i in range(n)],
            "end": [i * 2000 + 2000 for i in range(n)],
            "target_start": [i * 2000 + 500 for i in range(n)],
            "target_end": [i * 2000 + 1500 for i in range(n)],
            "split": splits,
        }
    )
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=5.0, size=(n, 3)).astype(np.float32)
    tracks = pl.DataFrame(
        {
            "track_name": ["hl60_a", "hl60_b", "k562_a"],
            "group": ["hl60", "hl60", "k562"],
            "log_size_factor": [0.0, 0.1, -0.1],
        }
    )
    return RegionCountData.from_arrays(regions, counts, ["hl60_a", "hl60_b", "k562_a"], tracks)


def _write_cache(out_dir, regions_df: pl.DataFrame, *, k: int, d: int, seed: int = 0) -> None:
    n = regions_df.height
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, k, d)).astype(np.float16)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.DataFrame(
        {
            "backbone": ["stub"],
            "checkpoint": ["ckpt"],
            "bin_size": [32],
            "k": [k],
            "d": [d],
            "context": [1024],
            "stride": [512],
            "pool_to": [None],
            "rc": [False],
            "region_hash": [region_table_hash(regions_df)],
            "n_regions": [n],
        },
        schema={
            "backbone": pl.Utf8,
            "checkpoint": pl.Utf8,
            "bin_size": pl.Int64,
            "k": pl.Int64,
            "d": pl.Int64,
            "context": pl.Int64,
            "stride": pl.Int64,
            "pool_to": pl.Int64,
            "rc": pl.Boolean,
            "region_hash": pl.Utf8,
            "n_regions": pl.Int64,
        },
    )
    manifest.write_parquet(out_dir / "manifest.parquet")
    table = pa.table(
        {
            "region_row": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "features": pa.FixedSizeListArray.from_arrays(
                pa.array(features.reshape(-1), type=pa.float16()), k * d
            ),
        }
    )
    pq.write_table(table, out_dir / "chr1.parquet")


def _base_cfg(dataset_dir, embeddings_dir, output_dir) -> dict:
    return {
        "seed": 0,
        "output_dir": str(output_dir),
        "data": {
            "path": str(dataset_dir),
            "embeddings_dir": str(embeddings_dir),
            "target_group": None,
            "specific_only": False,
            "gini_std_threshold": 1.0,
            "contrast_weighting": False,
            "contrast_weight_floor": 0.25,
            "contrast_weight_cap": 4.0,
            "count_mask_quantile": 0.999,
            "count_mask_factor": 0.0,
            "enable_rc_aug": False,
            "preload": False,
            "drop_missing_from_cache": False,
        },
        "model": {
            "hidden": 8,
            "dropout": 0.0,
            "eta_max": None,
            "count_noise": "nb",
            "count_replicate_offsets": True,
            "count_freeze_noise": False,
            "count_noise_shrinkage": 0.0,
        },
        "loss": {
            "contrast_weight": 0.0,
            "contrast_multiplier": None,
            "contrast_task_weight": 1.0,
        },
        "trainer": {
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "learning_rate": 1.0e-2,
            "weight_decay": 1.0e-2,
            "scheduler": "constant",
            "warmup_steps": 0,
            "max_epochs": 1,
            "max_steps": None,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "no",
            "gradient_clip_norm": 1.0,
            "log_every_n_steps": 1,
            "eval_every_n_steps": None,
            "evals_per_epoch": 1,
            "checkpoint_every_n_steps": None,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "early_stopping_patience": None,
            "eval_accumulation_steps": None,
            "max_eval_samples": None,
            "eval_on_start": False,
            "report_to": [],
            "resume_from_checkpoint": None,
            "init_weights_from_checkpoint": None,
            "provenance": {"enabled": False},
        },
    }


# --------------------------------------------------------------------------- #
# _build_optimizer
# --------------------------------------------------------------------------- #


def test_build_optimizer_excludes_count_head_from_weight_decay():
    config = RegionCountConfig(
        k=2,
        d=4,
        track_groups=[0, 0, 1],
        log_size_factors=[0.0, 0.1, -0.1],
        hidden=8,
    )
    model = RegionCountModel(config)
    trainer_cfg = TrainerConfig(learning_rate=1e-2, weight_decay=1e-2)
    optimizer = _build_optimizer(model, trainer_cfg)

    count_head_names = set(model.count_head_parameter_names())
    count_head_params = {id(p) for name, p in model.named_parameters() if name in count_head_names}

    # Every count_head parameter must appear in a weight_decay=0.0 group.
    zero_decay_ids: set[int] = set()
    for group in optimizer.param_groups:
        if group["weight_decay"] == 0.0:
            zero_decay_ids |= {id(p) for p in group["params"]}
    assert count_head_params <= zero_decay_ids


def test_build_optimizer_still_decays_multi_dim_head_weights():
    config = RegionCountConfig(k=2, d=4, track_groups=[0, 1], log_size_factors=[0.0, 0.0], hidden=8)
    model = RegionCountModel(config)
    optimizer = _build_optimizer(model, TrainerConfig(learning_rate=1e-2, weight_decay=1e-2))
    decayed_names = set()
    for name, parameter in model.named_parameters():
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0.0 and any(p is parameter for p in group["params"]):
                decayed_names.add(name)
    assert any(name.startswith("mlp.") and "weight" in name for name in decayed_names)


# --------------------------------------------------------------------------- #
# warm start
# --------------------------------------------------------------------------- #


def test_load_warm_start_loads_matching_shapes(tmp_path):
    config = RegionCountConfig(k=2, d=4, track_groups=[0, 1], log_size_factors=[0.0, 0.0], hidden=8)
    source = RegionCountModel(config)
    source.save_pretrained(tmp_path / "checkpoint")

    target = RegionCountModel(
        RegionCountConfig(k=2, d=4, track_groups=[0, 1], log_size_factors=[0.0, 0.0], hidden=8)
    )
    _load_warm_start(target, tmp_path / "checkpoint")
    params = zip(source.named_parameters(), target.named_parameters(), strict=True)
    for (name, a), (_, b) in params:
        assert torch.allclose(a, b), name


def test_load_warm_start_skips_shape_mismatched_count_head(tmp_path, caplog):
    source = RegionCountModel(
        RegionCountConfig(k=2, d=4, track_groups=[0, 1], log_size_factors=[0.0, 0.0], hidden=8)
    )
    source.save_pretrained(tmp_path / "checkpoint")

    # Three tracks instead of two: CountHead's buffers/parameters change shape.
    target = RegionCountModel(
        RegionCountConfig(
            k=2, d=4, track_groups=[0, 0, 1], log_size_factors=[0.0, 0.0, 0.0], hidden=8
        )
    )
    before = target.count_head.log_noise.clone()
    _load_warm_start(target, tmp_path / "checkpoint")
    # Shape-mismatched CountHead params are skipped, not raised on.
    assert torch.equal(target.count_head.log_noise, before)
    # The pooling/MLP trunk (unaffected by track count) is warm-started.
    assert torch.allclose(target.pool.score.weight, source.pool.score.weight)


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


def test_run_training_end_to_end_logs_contrast_pearson(tmp_path):
    data = _toy_region_data(n_per_split=8)
    dataset_dir = tmp_path / "dataset"
    data.write(dataset_dir)
    embeddings_dir = tmp_path / "embeddings"
    _write_cache(embeddings_dir, data.regions, k=2, d=4)

    output_dir = tmp_path / "out"
    cfg = _base_cfg(dataset_dir, embeddings_dir, output_dir)

    summary = run_training(cfg)

    assert summary["n_tracks"] == 3
    assert summary["n_groups"] == 2
    assert set(summary["group_names"]) == {"hl60", "k562"}

    state = json.loads((output_dir / "trainer_state.json").read_text())
    eval_entries = [
        entry for entry in state["log_history"] if "eval_contrast_pearson_mean" in entry
    ]
    assert eval_entries, f"no eval_contrast_pearson_mean in log history: {state['log_history']}"
    assert np.isfinite(eval_entries[0]["eval_contrast_pearson_mean"])

    assert (output_dir / "training_summary.json").exists()
    assert (output_dir / "provenance.json").exists() is False  # provenance disabled in this test


def test_run_training_with_target_group_and_contrast_weighting(tmp_path):
    data = _toy_region_data(n_per_split=8)
    dataset_dir = tmp_path / "dataset"
    data.write(dataset_dir)
    embeddings_dir = tmp_path / "embeddings"
    _write_cache(embeddings_dir, data.regions, k=2, d=4)

    output_dir = tmp_path / "out"
    cfg = _base_cfg(dataset_dir, embeddings_dir, output_dir)
    cfg["data"]["target_group"] = "hl60"
    cfg["data"]["contrast_weighting"] = True
    cfg["loss"]["contrast_weight"] = 1.0
    cfg["model"]["count_freeze_noise"] = True

    summary = run_training(cfg)
    assert summary["n_groups"] == 2

    state = json.loads((output_dir / "trainer_state.json").read_text())
    eval_entries = [
        entry for entry in state["log_history"] if "eval_contrast_pearson_mean" in entry
    ]
    assert eval_entries
