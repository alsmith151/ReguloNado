from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
from regulonado.metrics import (
    finalize_validation_metric_state,
    init_validation_metric_state,
    update_validation_metric_state,
)
from regulonado.model import (
    BorzoiBackboneAdapter,
    EnformerBackboneAdapter,
    FiLMPerturbHead,
    FreezePolicy,
    RegulonadoConfig,
    RegulonadoModel,
    ResidualFiLMPerturbHead,
    TransferMLPPerturbHead,
    build_condition_shared_track_index,
)
from regulonado.tracks_table import write_track_table
from regulonado.training.config import TrainerConfig
from regulonado.training.data import stack_batch_tensors
from regulonado.training.losses import scaled_poisson_multinomial_loss
from regulonado.training.runner import (
    _build_optimizer,
    _normalise_checkpoint_mode,
    constant_track_metadata_values,
    resolved_condition_ids,
    run_training,
)
from regulonado.training.transforms import get_transform


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(4, 8, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Conv1d(8, 8, kernel_size=1) for _ in range(3)])
        self.feature_dim = 8

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.stem(input_ids)
        for block in self.blocks:
            x = block(x)
        return x

    def iter_named_blocks(self):
        for idx, block in enumerate(self.blocks):
            yield f"blocks.{idx}", block


class DummyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = DummyBackbone()
        self.feature_dim = 8

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_features(input_ids)

    def iter_named_blocks(self):
        return self.inner.iter_named_blocks()


class DummyBorzoiModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleList([nn.Conv1d(8, 8, 1) for _ in range(2)])
        self.final_joined_convs = nn.Conv1d(8, 8, 1)

    def get_embs_after_crop(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids


class DummyEnformerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"dim": 8})()
        self.transformer = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
        self.final_pointwise = nn.Linear(8, 8)

    def forward(self, input_ids: torch.Tensor, return_only_embeddings: bool = False):
        assert return_only_embeddings
        return torch.cat([input_ids, input_ids], dim=-1)


def test_condition_shared_track_index_collapses_non_condition_axes():
    records = [
        {"track_index": 0, "condition_id": 0, "assay_type": "ATAC", "target": "X"},
        {"track_index": 1, "condition_id": 1, "assay_type": "ATAC", "target": "X"},
        {"track_index": 2, "condition_id": 0, "assay_type": "ChIP", "target": "Y"},
        {"track_index": 3, "condition_id": 1, "assay_type": "ChIP", "target": "Y"},
    ]

    assert build_condition_shared_track_index(records) == [0, 0, 1, 1]


def test_group_condition_source_is_stably_encoded_and_persistable():
    records = [
        {"track_index": 0, "condition_id": 8, "group": " treatment "},
        {"track_index": 1, "condition_id": 3, "group": "cell_type"},
        {"track_index": 2, "condition_id": 1, "group": "treatment"},
        {"track_index": 3, "condition_id": 2, "group": ""},
    ]

    # Sorted opaque labels are stable even if track rows arrive in a different order.
    assert resolved_condition_ids(records, "group") == [1, 0, 1, -1]
    assert constant_track_metadata_values(records, condition_source="group")[
        "track_condition_ids"
    ] == [1, 0, 1, -1]
    # Existing condition_id behavior remains unchanged by default.
    assert resolved_condition_ids(records) == [8, 3, 1, 2]


def test_group_condition_source_requires_usable_label():
    with pytest.raises(ValueError, match="non-empty 'group' value"):
        resolved_condition_ids([{"group": None}, {"group": "  "}], "group")


def test_group_is_excluded_from_shared_channel_identity():
    records = [
        {"track_index": 0, "group": "A", "project": "shared"},
        {"track_index": 1, "group": "B", "project": "shared"},
    ]
    assert build_condition_shared_track_index(records, condition_source="group") == [0, 0]


def test_shared_channel_identity_ignores_per_track_technical_fields():
    # tracks.parquet-derived records carry per-file fields (path, resolved_path,
    # background) alongside assay_class/group. None of the "preferred" condition-collapse
    # fields (assay_type, source, cell_line, ...) exist under those exact names here, so
    # this exercises the fallback branch: it must skip path/resolved_path/background
    # (near-unique per track) rather than let them defeat the collapse entirely.
    records = [
        {
            "track_index": 0,
            "group": "NK_cell",
            "assay_class": "ATAC",
            "path": "/data/a.bigWig",
            "resolved_path": "/ceph/data/a.bigWig",
            "background": 0.165,
        },
        {
            "track_index": 1,
            "group": "B_cell",
            "assay_class": "ATAC",
            "path": "/data/b.bigWig",
            "resolved_path": "/ceph/data/b.bigWig",
            "background": 0.412,
        },
        {
            "track_index": 2,
            "group": "K-562",
            "assay_class": "ChIP",
            "path": "/data/c.bigWig",
            "resolved_path": "/ceph/data/c.bigWig",
            "background": 0.775,
        },
    ]
    assert build_condition_shared_track_index(records, condition_source="group") == [0, 0, 1]


def test_film_head_accepts_optional_metadata():
    head = FiLMPerturbHead(
        in_ch=8,
        hidden=4,
        n_tracks=3,
        use_track_metadata=True,
        num_conditions=4,
    )
    inputs = torch.randn(2, 8, 16)
    outputs = head(inputs, track_condition_ids=torch.tensor([0, 1, 2]))
    assert outputs.shape == (2, 3, 16)
    assert torch.all(outputs >= 0)


def test_residual_and_transfer_heads_produce_expected_shapes():
    residual_head = ResidualFiLMPerturbHead(in_ch=8, hidden=8, n_tracks=2)
    transfer_head = TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2, mlp_hidden=6)
    inputs = torch.randn(2, 8, 12)
    assert residual_head(inputs).shape == (2, 2, 12)
    assert transfer_head(inputs).shape == (2, 2, 12)


def test_transfer_head_accepts_per_track_output_bias_initialization():
    head = TransferMLPPerturbHead(
        in_ch=8,
        hidden=4,
        n_tracks=2,
        output_bias_init=[-1.5, 0.75],
    )

    torch.testing.assert_close(head.proj[-1].bias, torch.tensor([-1.5, 0.75]))


def test_transfer_head_can_start_as_empirical_mean_constant():
    means = torch.tensor([0.25, 1.5])
    bias = torch.log(torch.expm1(means))
    head = TransferMLPPerturbHead(
        in_ch=8,
        hidden=4,
        n_tracks=2,
        output_bias_init=bias.tolist(),
        zero_output_weights=True,
    )

    output = head(torch.randn(3, 8, 7))
    torch.testing.assert_close(output, means.view(1, 2, 1).expand(3, 2, 7))


def test_freeze_policy_unfreezes_last_block_only():
    backbone = DummyBackbone()
    model = RegulonadoModel(
        backbone=backbone, head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2)
    )
    model.apply_freeze_policy(
        FreezePolicy(freeze_backbone=True, unfreeze_backbone_stages_from_output_end=1)
    )

    trainable = {
        name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("blocks.2") for name in trainable)


def test_borzoi_adapter_normalizes_backbone_interface():
    adapter = BorzoiBackboneAdapter(DummyBorzoiModule())
    features = adapter.forward_features(torch.randn(2, 8, 12))
    assert features.shape == (2, 8, 12)
    assert [name for name, _ in adapter.iter_named_blocks()] == [
        "transformer.0",
        "transformer.1",
        "final_joined_convs",
    ]


class DummyBorzoiUNetModule(DummyBorzoiModule):
    def __init__(self):
        super().__init__()
        for level in (1, 0):
            setattr(self, f"upsampling_unet{level}", nn.Conv1d(8, 8, 1))
            setattr(self, f"horizontal_conv{level}", nn.Conv1d(8, 8, 1))
            setattr(self, f"separable{level}", nn.Conv1d(8, 8, 1))


def test_borzoi_adapter_keeps_float32_master_weights_for_half_checkpoints():
    # A bf16 backbone would make AdamW round small updates to zero; the adapter must
    # restore float32 weights regardless of the dtype the checkpoint was loaded in.
    adapter = BorzoiBackboneAdapter(DummyBorzoiModule().to(torch.bfloat16))

    assert {parameter.dtype for parameter in adapter.parameters()} == {torch.float32}
    features = adapter.forward_features(torch.randn(2, 8, 12, dtype=torch.bfloat16))
    assert features.dtype == torch.float32


def test_borzoi_adapter_stages_include_unet_upsampling_path():
    adapter = BorzoiBackboneAdapter(DummyBorzoiUNetModule())

    assert [name for name, _ in adapter.iter_named_blocks()] == [
        "transformer.0",
        "transformer.1",
        "unet1",
        "unet0",
        "final_joined_convs",
    ]


def test_freeze_policy_unfreezes_unet_stages_between_transformer_and_output():
    model = RegulonadoModel(
        backbone=BorzoiBackboneAdapter(DummyBorzoiUNetModule()),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    model.apply_freeze_policy(
        FreezePolicy(freeze_backbone=True, unfreeze_backbone_stages_from_output_end=4)
    )

    trainable = {
        name.split(".")[1]
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "transformer",
        "upsampling_unet1",
        "horizontal_conv1",
        "separable1",
        "upsampling_unet0",
        "horizontal_conv0",
        "separable0",
        "final_joined_convs",
    }
    assert not model.backbone.model.transformer[0].weight.requires_grad


def test_train_mode_keeps_frozen_backbone_batchnorm_statistics_fixed():
    module = DummyBorzoiUNetModule()
    module.transformer[0] = nn.BatchNorm1d(8)
    module.final_joined_convs = nn.Sequential(nn.Conv1d(8, 8, 1), nn.BatchNorm1d(8))
    model = RegulonadoModel(
        backbone=BorzoiBackboneAdapter(module),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    model.apply_freeze_policy(
        FreezePolicy(freeze_backbone=True, unfreeze_backbone_stages_from_output_end=1)
    )
    frozen_norm = module.transformer[0]
    trainable_norm = module.final_joined_convs[1]
    running_mean = frozen_norm.running_mean.clone()

    model.train()
    frozen_norm(torch.randn(4, 8, 12) + 5.0)

    assert not frozen_norm.training
    assert trainable_norm.training
    assert model.head.training
    torch.testing.assert_close(frozen_norm.running_mean, running_mean)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash_attn needs CUDA")
def test_flashed_borzoi_trains_in_float32_under_bf16_autocast():
    pytest.importorskip("flash_attn")
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    from regulonado.model.adapters import Borzoi

    torch.manual_seed(0)
    adapter = BorzoiBackboneAdapter(
        Borzoi(BorzoiConfig(depth=1, flashed=True, bins_to_return=1024))
    ).cuda()
    layer_norm = adapter.model.transformer[0][0].fn[0]
    before = layer_norm.weight.detach().clone()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0)

    one_hot = torch.nn.functional.one_hot(
        torch.randint(0, 4, (1, 65536), device="cuda"), num_classes=4
    ).permute(0, 2, 1)
    features = adapter.forward_features(one_hot)
    assert torch.isfinite(features).all()
    features.float().square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in adapter.parameters()
    )
    optimizer.step()

    assert layer_norm.weight.dtype == torch.float32
    # LayerNorm gains sit near 1.0, where a 1e-4 step is below bf16 resolution and
    # would round away without float32 master weights.
    assert not torch.equal(layer_norm.weight.detach(), before)


def test_enformer_adapter_transposes_sequence_axes():
    adapter = EnformerBackboneAdapter(DummyEnformerModule())
    features = adapter.forward_features(torch.randn(2, 4, 16))
    assert features.shape == (2, 8, 16)
    assert [name for name, _ in adapter.iter_named_blocks()] == [
        "transformer.0",
        "transformer.1",
        "final_pointwise",
    ]


def test_stack_batch_tensors_and_transform_roundtrip():
    batch = [
        {"sequence": torch.tensor([1.0, 2.0]), "target": [3.0, 4.0]},
        {"sequence": torch.tensor([5.0, 6.0]), "target": [7.0, 8.0]},
    ]
    stacked = stack_batch_tensors(batch, tensor_keys=("sequence", "target"))
    assert stacked["sequence"].shape == (2, 2)
    transform = get_transform("log1p")
    restored = transform.inverse(transform.forward(stacked["target"]))
    assert torch.allclose(restored, stacked["target"].float())


def test_scaled_poisson_multinomial_loss_is_finite():
    pred = torch.rand(2, 3, 10) + 0.1
    target = torch.rand(2, 3, 10) + 0.1
    loss = scaled_poisson_multinomial_loss(
        pred, target, scale_factors=torch.tensor([1.0, 2.0, 0.5])
    )
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_optimizer_excludes_bias_and_norm_from_weight_decay():
    model = RegulonadoModel(
        backbone=DummyBackbone(),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    cfg = TrainerConfig(learning_rate=1e-3, backbone_learning_rate=1e-4, weight_decay=0.1)
    optimizer = _build_optimizer(model, cfg)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}
    assert {group["lr"] for group in optimizer.param_groups} == {1e-3, 1e-4}


def test_regulonado_model_save_pretrained_roundtrip(tmp_path, monkeypatch):
    from regulonado.model import adapters

    config = RegulonadoConfig(
        backbone_type="dummy",
        head_type="transfer_mlp",
        head_hidden=4,
        mlp_hidden=4,
        feature_dim=8,
        n_tracks=2,
        context_length=12,
        n_pred_bins=12,
        bin_size=1,
        track_names=["a", "b"],
    )
    model = RegulonadoModel(
        config,
        backbone=DummyBackbone(),
        head=TransferMLPPerturbHead(in_ch=8, hidden=4, n_tracks=2),
    )
    model.save_pretrained(tmp_path, safe_serialization=True)

    monkeypatch.setattr(adapters, "build_backbone_architecture", lambda *_: DummyBackbone())
    loaded = RegulonadoModel.from_pretrained(tmp_path)

    assert isinstance(loaded, RegulonadoModel)
    assert loaded.config.context_length == 12
    assert loaded.config.track_names == ["a", "b"]
    assert loaded(torch.randn(1, 4, 12)).shape == (1, 2, 12)


def test_checkpoint_mode_normalisation():
    assert _normalise_checkpoint_mode("true") is True
    assert _normalise_checkpoint_mode("false") is None
    assert _normalise_checkpoint_mode("") is None
    assert _normalise_checkpoint_mode("checkpoint-3000") == "checkpoint-3000"


def test_metric_state_accumulates_and_finalizes():
    state = init_validation_metric_state()
    update_validation_metric_state(state, pred_lfc=[0.0, 1.0, 2.0], meas_lfc=[0.1, 0.9, 2.1])
    metrics = finalize_validation_metric_state(state)
    assert "delta_lfc/pearson" in metrics


def test_metric_state_without_conditions_skips_delta_lfc_metrics():
    state = init_validation_metric_state()
    metrics = finalize_validation_metric_state(state)
    assert "delta_lfc/pearson" not in metrics
    assert "reconstruction/raw_pearson" in metrics


def _write_parquet_split(data_dir, split: str, *, n_rows: int, context: int, n_tracks: int) -> None:
    """Write a tiny HF-layout Parquet shard directly with pyarrow, for a dataset fixture."""
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=(n_rows, context), dtype=np.uint8).tolist()
    sig = rng.random((n_rows, n_tracks, context)).astype(np.float32).tolist()
    table = pa.table(
        {
            "sequence_tokens": pa.array(seq, type=pa.list_(pa.uint8(), context)),
            "signal": pa.array(sig, type=pa.list_(pa.list_(pa.float32(), context), n_tracks)),
            "interval": pa.array([f"chr1:{i}" for i in range(n_rows)]),
            "index": pa.array(list(range(n_rows)), type=pa.int64()),
            "local_index": pa.array(list(range(n_rows)), type=pa.int64()),
        }
    )
    (data_dir / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(data_dir / "data" / f"{split}-00000-of-00001.parquet"))


def test_run_training_entrypoint_with_dummy_adapter(tmp_path):
    data_dir = tmp_path / "dataset"
    (data_dir / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text("# fixture dataset\n")
    _write_parquet_split(data_dir, "train", n_rows=4, context=12, n_tracks=2)
    _write_parquet_split(data_dir, "validation", n_rows=2, context=12, n_tracks=2)

    # 'condition' is a label, not an id — categorical ids are derived by sorted
    # factorisation at load time (see tracks_table.to_track_records), which is
    # why the two tables below use different label sets rather than literal ids.
    def _write_tracks(path, conditions, scale_factors, groups):
        write_track_table(
            pd.DataFrame(
                {
                    "track_name": ["t0", "t1"],
                    "status": ["included", "included"],
                    "track_index": [0, 1],
                    "condition": conditions,
                    "group": groups,
                    "assay": ["atac", "atac"],
                    "scale_factor": scale_factors,
                }
            ),
            path,
            context_length=12,
            bin_size=1,
            n_pred_bins=12,
            shift_max_bp=0,
        )

    _write_tracks(data_dir / "tracks.parquet", ["a", "b"], [1.0, 1.0], ["x", "y"])
    enriched_metadata = tmp_path / "tracks.enriched.parquet"
    _write_tracks(enriched_metadata, ["z", "a"], [2.0, 3.0], ["beta", "alpha"])

    summary = run_training(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "run"),
            "data": {
                "path": str(data_dir),
                "metadata_path": str(enriched_metadata),
                "apply_scale": False,
                "apply_squash": False,
                "apply_clip": False,
                "enable_rc_aug": False,
                "context_length": 12,
                "n_pred_bins": 12,
            },
            "backbone": {"name": "borzoi", "config_overrides": {}},
            "model": {
                "use_track_metadata": True,
                "condition_source": "group",
                "share_condition_base_channels": True,
                "metadata_hidden": 8,
                "activation_type": "softplus",
            },
            "head": {
                "type": "film",
                "hidden": 8,
                "dropout": 0.0,
                "refinement_kernel": 9,
                "mlp_hidden": None,
            },
            "loss": {"name": "mse", "poisson_weight": 0.0, "delta": 1.0},
            "trainer": {
                "batch_size": 2,
                "eval_batch_size": 2,
                "num_workers": 0,
                "learning_rate": 1e-3,
                "backbone_learning_rate": 1e-3,
                "weight_decay": 0.0,
                "scheduler": "linear",
                "warmup_steps": 0,
                "max_epochs": 1,
                "max_steps": 2,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "no",
                "gradient_clip_norm": 1.0,
                "eval_every_n_steps": 1,
                "checkpoint_every_n_steps": None,
                "freeze_backbone": True,
                "unfreeze_backbone_stages_from_output_end": 1,
                "unfreeze_module_names": [],
                "provenance": {"enabled": True, "save_git_diff": False},
            },
        },
        adapter_builder=lambda _: DummyAdapter(),
    )

    assert summary["history"]["train/loss"]
    assert summary["metadata_path"] == str(enriched_metadata)
    assert (tmp_path / "run" / "training_summary.json").exists()
    assert (tmp_path / "run" / "provenance.json").exists()
    assert (tmp_path / "run" / "resolved_config.json").exists()
    assert (tmp_path / "run" / "config.json").exists()
    assert (tmp_path / "run" / "model.safetensors").exists()

    saved_config = RegulonadoConfig.from_pretrained(tmp_path / "run")
    assert saved_config.track_names == ["t0", "t1"]
    # Opaque group labels ["beta", "alpha"] sort to ["alpha", "beta"], so t0=1 and t1=0.
    # This proves the enriched metadata path drives the persisted FiLM IDs.
    assert saved_config.track_metadata["track_condition_ids"] == [1, 0]
    assert saved_config.condition_source == "group"


@pytest.mark.parametrize("head_type", ["transfer_mlp", "film", "log_film", "bias"])
def test_model_construction_keeps_head_output_initialisation(head_type: str) -> None:
    config = RegulonadoConfig(
        feature_dim=8,
        n_tracks=3,
        head_type=head_type,
        head_hidden=4,
        output_bias_init=[0.5, 1.0, 1.5],
        zero_output_weights=True,
    )
    model = RegulonadoModel(config, backbone=DummyAdapter())

    output = model(torch.randn(2, 4, 6))

    expected = torch.nn.functional.softplus(torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(output, expected.view(1, 3, 1).expand(2, 3, 6))


def test_shared_base_channels_average_member_track_output_bias() -> None:
    head = FiLMPerturbHead(
        in_ch=8,
        hidden=4,
        n_tracks=3,
        use_track_metadata=True,
        num_conditions=2,
        condition_shared_track_index=[0, 0, 1],
        output_bias_init=[1.0, 3.0, 5.0],
        zero_output_weights=True,
    )

    torch.testing.assert_close(head.proj[3].bias, torch.tensor([2.0, 5.0]))
    assert not head.proj[3].weight.any()


def test_model_construction_keeps_film_modulation_identity_at_init() -> None:
    config = RegulonadoConfig(
        feature_dim=8, n_tracks=2, head_type="film", head_hidden=4, use_track_metadata=True,
        num_conditions=2,
    )
    model = RegulonadoModel(config, backbone=DummyAdapter())

    assert not model.head.metadata_to_scale.weight.any()
    assert not model.head.metadata_to_shift.weight.any()
