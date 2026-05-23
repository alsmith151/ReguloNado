from __future__ import annotations

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
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
    HeadedSequenceModel,
    ResidualFiLMPerturbHead,
    TransferMLPPerturbHead,
    build_condition_shared_track_index,
)
from regulonado.train import run_training
from regulonado.training import (
    TrainerConfig,
    get_transform,
    scaled_poisson_multinomial_loss,
    stack_batch_tensors,
)


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


def test_freeze_policy_unfreezes_last_block_only():
    backbone = DummyBackbone()
    model = HeadedSequenceModel(
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


def test_metric_state_accumulates_and_finalizes():
    state = init_validation_metric_state()
    update_validation_metric_state(state, pred_lfc=[0.0, 1.0, 2.0], meas_lfc=[0.1, 0.9, 2.1])
    metrics = finalize_validation_metric_state(state)
    assert "delta_lfc/pearson" in metrics


def test_run_training_entrypoint_with_dummy_adapter(tmp_path):
    data_dir = tmp_path / "dataset"
    ds = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [torch.randn(4, 12).tolist() for _ in range(4)],
                    "labels": [torch.rand(2, 12).tolist() for _ in range(4)],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "input_ids": [torch.randn(4, 12).tolist() for _ in range(2)],
                    "labels": [torch.rand(2, 12).tolist() for _ in range(2)],
                }
            ),
        }
    )
    ds.save_to_disk(str(data_dir))
    (data_dir / "regulonado_metadata.json").write_text(
        """
{
  "context_length": 12,
  "n_pred_bins": 12,
  "bin_size": 1,
  "shift_max_bp": 0,
  "final_track_records": [
    {"track_index": 0, "condition_id": 0, "assay_type_id": 0, "scale_factor": 1.0},
    {"track_index": 1, "condition_id": 1, "assay_type_id": 0, "scale_factor": 1.0}
  ]
}
""".strip()
    )

    summary = run_training(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "run"),
            "data": {
                "path": str(data_dir),
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
                "checkpoint_every_n_steps": None,
                "freeze_backbone": True,
                "unfreeze_backbone_stages_from_output_end": 1,
                "unfreeze_module_names": [],
            },
        },
        adapter_builder=lambda _: DummyAdapter(),
    )

    assert summary["history"]["train/loss"]
    assert (tmp_path / "run" / "training_summary.json").exists()
