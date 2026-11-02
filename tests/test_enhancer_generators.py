from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from regulonado.cli.app import app
from regulonado.design.diffusion import (
    PAD,
    Denoiser,
    DiffusionConfig,
    condition_vector,
    load_checkpoint,
    mdlm_loss,
    sample,
    train_diffusion,
)
from regulonado.design.generators import diffusion_search, diverse_sequences, gradient_search
from regulonado.genomics import one_hot
from typer.testing import CliRunner


def config(**kwargs):
    return DiffusionConfig(
        activity_columns=["target", "other"],
        width=16,
        heads=2,
        layers=1,
        max_length=16,
        device="cpu",
        epochs=1,
        **kwargs,
    )


def test_numeric_conditioning_and_null_are_distinct():
    torch.manual_seed(1)
    model = Denoiser(config()).eval()
    tokens = torch.full((1, 8), 4)
    time = torch.ones(1)
    values = torch.tensor([[2.0, -1.0]])
    known = torch.ones(1, 2, dtype=torch.bool)
    assert not torch.allclose(
        model(tokens, time, values, known), model(tokens, time, values, ~known)
    )
    assert torch.allclose(
        model(tokens, time, values, ~known), model(tokens, time, values * 10, ~known)
    )


def test_sampling_complete_and_deterministic():
    model = Denoiser(config())
    first = sample(model, [0.0, 1.0], [True, True], length=8, count=3, steps=4, seed=5)
    assert first == sample(model, [0.0, 1.0], [True, True], length=8, count=3, steps=4, seed=5)
    assert all(len(s) == 8 and set(s) <= set("ACGT") for s in first)
    assert all(
        len(s) == 8
        for s in sample(model, [0.0, 0.0], [False, False], length=8, count=2, steps=1, guidance=2)
    )
    with pytest.raises(ValueError):
        sample(model, [0.0, 0.0], [True, True], length=17)


def test_loss_ignores_padding_and_unknown_targets():
    model = Denoiser(config())
    clean = torch.tensor([[0, 1, 2, 3, 6, PAD]])
    loss = mdlm_loss(model, clean, torch.zeros(1, 2), torch.tensor([[True, False]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output.weight.grad is not None


def table(tmp_path):
    path = tmp_path / "training.tsv"
    pd.DataFrame(
        dict(
            chrom=["chr1", "chr1", "chr8"],
            sequence=["AAAACCCC", "ACGTACGA", "TGCATTGC"],
            target=[1.0, 3.0, 1000.0],
            other=[2.0, np.nan, 5.0],
        )
    ).to_csv(path, sep="\t", index=False)
    return path


def test_training_checkpoint_and_split_normalisation(tmp_path):
    checkpoint_path = train_diffusion(table(tmp_path), tmp_path / "model", config())
    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    assert checkpoint["mean"] == [2.0, 2.0]
    assert checkpoint["training_rows"] == 2
    values, observed = condition_vector(checkpoint, {"target": 3.0})
    assert values.tolist() == [1.0, 0.0]
    assert observed.tolist() == [True, False]
    with pytest.raises(ValueError, match="Unknown"):
        condition_vector(checkpoint, {"typo": 1})
    assert len(sample(model, values, observed, length=8, count=1, steps=2)) == 1
    runner = CliRunner()
    output = tmp_path / "generated"
    result = runner.invoke(
        app,
        [
            "generate",
            "sample",
            "--checkpoint",
            str(checkpoint_path),
            "--activities",
            '{"target": 3}',
            "--length",
            "8",
            "--count",
            "3",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "sequences.fa").exists()
    assert (output / "run.json").exists()


def test_split_leakage_rejected(tmp_path):
    path = table(tmp_path)
    data = pd.read_csv(path, sep="\t")
    data.loc[2, "sequence"] = "GGGGTTTT"  # reverse complement of training row 0
    data.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="reverse-complement"):
        train_diffusion(path, tmp_path / "bad", config())


def toy_energy(batch):
    x = torch.as_tensor(batch)
    return SimpleNamespace(energy=-x[:, 0, 2:6].sum(1))


def test_gradient_improves_discrete_sequence_preserves_flanks():
    context = one_hot("CCCCCCCC")
    seed = SimpleNamespace(editable=slice(2, 6))
    state = gradient_search(toy_energy, seed, context, rounds=20, learning_rate=0.5)
    assert state.energy == -4
    np.testing.assert_array_equal(state.context[:, :2], context[:, :2])
    np.testing.assert_array_equal(state.context[:, 6:], context[:, 6:])
    np.testing.assert_array_equal(context, one_hot("CCCCCCCC"))


def test_diffusion_ranks_completed_samples_and_preserves_flanks(tmp_path, monkeypatch):
    import regulonado.design.diffusion as module

    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (None, {}))
    monkeypatch.setattr(module, "condition_vector", lambda *args: ([0.0], [True]))
    monkeypatch.setattr(module, "sample", lambda *args, **kwargs: ["CCCC", "AAAA", "AACC"])
    seed = SimpleNamespace(editable=slice(2, 6))
    context = one_hot("CCCCCCCC")
    state = diffusion_search(toy_energy, seed, context, checkpoint="unused", activities={})
    assert state.energy == -4
    np.testing.assert_array_equal(state.context, one_hot("CCAAAACC"))
    assert len(state.history) == 4


def test_diversity():
    assert diverse_sequences(["AAAA", "AAAA", "AAAC", "CCCC"], count=3, min_distance=2) == [
        "AAAA",
        "CCCC",
    ]


def test_mdlm_weight_and_denominator(monkeypatch):
    class Uniform(torch.nn.Module):
        def forward(self, tokens, time, values, observed):
            return torch.zeros(*tokens.shape, 4, requires_grad=True)

    # t=0.5, every valid base masked. Padding/unknowns do not enter denominator.
    calls = iter([torch.tensor([0.5]), torch.zeros(1, 6), torch.ones(1)])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(calls))
    loss = mdlm_loss(
        Uniform(),
        torch.tensor([[0, 1, 2, 3, 6, PAD]]),
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert float(loss.detach()) == pytest.approx(2 * np.log(4))


def test_revealed_tokens_never_change():
    class Recording(Denoiser):
        def forward(self, tokens, *args):
            if hasattr(self, "previous"):
                revealed = self.previous.ne(4)
                assert torch.equal(tokens[revealed], self.previous[revealed])
            self.previous = tokens.clone()
            return super().forward(tokens, *args)

    model = Recording(config())
    sample(model, [0.0, 0.0], [True, True], length=16, count=2, steps=8)


def test_shared_panel_selection(tmp_path):
    path = tmp_path / "pool.tsv"
    pd.DataFrame(dict(sequence=["AAAA", "AAAC", "CCCC"], energy=[-3, -2, -1])).to_csv(
        path, sep="\t", index=False
    )
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "select",
            "--table",
            str(path),
            "--out",
            str(tmp_path / "panel"),
            "--count",
            "2",
            "--min-distance",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    selected = pd.read_csv(tmp_path / "panel" / "selected.tsv", sep="\t")
    assert selected.sequence.tolist() == ["AAAA", "CCCC"]


@pytest.mark.parametrize("region", [False, True])
def test_real_predictor_interfaces_preserve_input_gradients(region):
    from regulonado.design.predictor import SequencePredictor
    from regulonado.design.region_predictor import RegionSequencePredictor

    class Toy(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x[:, :2].sum(-1, keepdim=True)

        def group_log_rates(self, sequence):
            return sequence[:, :2].sum(-1)

    predictor = object.__new__(RegionSequencePredictor if region else SequencePredictor)
    predictor.model = Toy()
    predictor.device, predictor.dtype = "cpu", torch.float32
    predictor.track_names, predictor.track_metadata = ["a", "b"], {}
    x = torch.rand(1, 4, 8, requires_grad=True)
    output = predictor(x)
    (gradient,) = torch.autograd.grad(output.sum(), x)
    assert gradient[:, :2].abs().sum() > 0
    assert predictor.model.training  # no mode mutation inside prediction


def test_new_method_configuration():
    from regulonado.config.models import DesignConfig

    base = {"candidates": "cores.bed"}
    with pytest.raises(ValueError, match="diffusion_checkpoint"):
        DesignConfig(**base, targets=[dict(name="d", target="HL60", method="diffusion")])
    with pytest.raises(ValueError, match="resident"):
        DesignConfig(
            **base,
            fold_mode="sequential",
            targets=[dict(name="g", target="HL60", method="gradient")],
        )
    parsed = DesignConfig(
        **base,
        targets=[
            dict(
                name="g", target="HL60", method="gradient", settings={"gradient_learning_rate": 0.2}
            ),
            dict(
                name="d",
                target="HL60",
                method="diffusion",
                settings={
                    "diffusion_checkpoint": "best.pt",
                    "diffusion_activities": {"HL60_activity": 2.1},
                },
            ),
        ],
    )
    assert len(parsed.targets) == 2
    with pytest.raises(ValueError, match="gradient-only"):
        DesignConfig(**base, gradient_temperature=1.0, targets=[dict(name="i", target="HL60")])


@pytest.mark.parametrize("min_epochs,expected", [(0, 3), (5, 5)])
def test_early_stopping_and_observed_condition_sampling(
    tmp_path, monkeypatch, min_epochs, expected
):
    import json

    import regulonado.design.diffusion as module
    import torch.utils.data as data

    original = data.WeightedRandomSampler
    captured = []

    def sampler(weights, *args, **kwargs):
        captured.append(weights.tolist())
        return original(weights, *args, **kwargs)

    monkeypatch.setattr(data, "WeightedRandomSampler", sampler)
    monkeypatch.setattr(
        module, "mdlm_loss", lambda model, *args, **kwargs: model.output.weight.sum() * 0 + 1.0
    )
    settings = config().model_copy(
        update=dict(
            epochs=10,
            early_stopping_patience=2,
            early_stopping_min_epochs=min_epochs,
            condition_sampling_weights={"other": 16.0},
        )
    )
    out = tmp_path / "model"
    train_diffusion(table(tmp_path), out, settings)
    assert captured == [[16.0, 1.0]]
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["epochs_completed"] == expected
    assert summary["best_epoch"] == 1
    assert summary["early_stopped"]
    with pytest.raises(ValueError, match="Sampling weights"):
        DiffusionConfig(activity_columns=["target"], condition_sampling_weights={"typo": 2})


def test_fixed_mask_diagnostics_are_per_masked_base_cross_entropy():
    from regulonado.design.diffusion import denoising_diagnostics

    class Uniform(torch.nn.Module):
        def forward(self, tokens, *args):
            return torch.zeros(*tokens.shape, 4)

    clean = torch.tensor([[0, 1, 2, 3] * 8] * 4)
    metrics = denoising_diagnostics(
        Uniform(),
        clean,
        torch.zeros(4, 2),
        torch.ones(4, 2, dtype=torch.bool),
        prefix="validation",
        seed=17,
    )
    for rate in ["15pct", "50pct", "90pct"]:
        assert metrics[f"validation_masked_nll_{rate}"] == pytest.approx(np.log(4), abs=1e-6)
        assert 0 <= metrics[f"validation_masked_accuracy_{rate}"] <= 1
