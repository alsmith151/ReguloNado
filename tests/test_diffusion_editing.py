"""Edit invariants and recoverable sequence provenance, without remote services."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from regulonado.design.diffusion import sample
from regulonado.design.generators import diffusion_search
from regulonado.design.provenance import (
    ProposalJournal,
    merge_provenance,
    read_template,
    sequence_identity,
)
from regulonado.genomics import one_hot


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_length=16)
        self.calls = []

    def forward(self, tokens, times, *conditions):
        self.calls.append((tokens.clone(), times.clone()))
        logits = torch.zeros((*tokens.shape, 4), device=tokens.device)
        logits[:, :, 0] = 100
        return logits


def test_edit_defaults_preserve_unmasked_bases_and_use_partial_schedule():
    model = RecordingModel()
    records = sample(
        model,
        [1.0],
        [True],
        length=8,
        template="CCCCCCCC",
        count=3,
        steps=4,
        seed=12,
        return_records=True,
    )
    for row in records:
        assert len(row["masked_positions"]) == 2
        assert row["actual_mask_fraction"] == 0.25
        assert row["sequence"].count("A") == 2
        for i, base in enumerate(row["sequence"]):
            assert base == ("A" if i in row["masked_positions"] else "C")
    assert [float(times[0]) for _, times in model.calls] == [0.25, 0.1875, 0.125, 0.0625]
    assert records == sample(
        RecordingModel(),
        [1.0],
        [True],
        length=8,
        template="CCCCCCCC",
        count=3,
        steps=4,
        seed=12,
        return_records=True,
    )
    for initial, _ in model.calls:
        assert initial.eq(4).sum(1).max() <= 2


def test_short_sequences_round_up_to_one_mask():
    row = sample(
        RecordingModel(), [1.0], [True], length=1, template="C", count=1, return_records=True
    )[0]
    assert row["masked_positions"] == [0]
    assert row["actual_mask_fraction"] == 1
    assert row["sequence"] == "A"


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_edit_fraction(fraction):
    with pytest.raises(ValueError, match="mask_fraction"):
        sample(RecordingModel(), [1.0], [True], length=4, template="CCCC", mask_fraction=fraction)


def test_edit_requires_matching_template():
    for template in [None, "CCC", "CCCN"]:
        with pytest.raises(ValueError, match="template"):
            sample(RecordingModel(), [1.0], [True], length=4, template=template)


def test_explicit_de_novo_does_not_require_a_template():
    assert sample(RecordingModel(), [1.0], [True], mode="de_novo", length=4, count=1) == ["AAAA"]


def mock_proposals(monkeypatch, sequences):
    import regulonado.design.diffusion as module

    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (None, {}))
    monkeypatch.setattr(module, "condition_vector", lambda *args: ([0.0], [True]))
    monkeypatch.setattr(
        module,
        "sample",
        lambda *args, **kwargs: [
            dict(
                sequence=s,
                actual_mask_fraction=0.25,
                requested_mask_fraction=0.25,
                mode=kwargs["mode"],
                masked_positions=[0],
                random_seed=kwargs["seed"],
                sample_index=i,
            )
            for i, s in enumerate(sequences)
        ],
    )


def test_score_cache_event_identity_selection_and_flanks(tmp_path, monkeypatch):
    mock_proposals(monkeypatch, ["ACCC", "ACCC", "CCCC"])
    calls = []

    def oracle(batch):
        calls.append(batch.copy())
        return SimpleNamespace(energy=-batch[:, 0, 2:6].sum(1))

    journal = ProposalJournal(tmp_path, parent="CCCC", metadata={"conditions": {"HL60": 2}})
    state = diffusion_search(
        oracle,
        SimpleNamespace(editable=slice(2, 6)),
        one_hot("CCCCCCCC"),
        checkpoint="unused",
        activities={},
        journal=journal,
        count=3,
    )
    assert len(calls) == 2
    assert state.energy == -1
    assert len(journal.records) == 4  # baseline + all 3 events
    rows = list(journal.records.values())
    assert rows[1]["sequence_id"] == rows[2]["sequence_id"] == sequence_identity("accc")
    assert rows[1]["proposal_id"] != rows[2]["proposal_id"]
    np.testing.assert_array_equal(state.context[:, :2], one_hot("CC"))
    np.testing.assert_array_equal(state.context[:, 6:], one_hot("CC"))
    selected = json.loads((journal.root / "selection.json").read_text())
    assert selected["oracle_evaluations"] == 2
    assert selected["proposal_id"] == rows[1]["proposal_id"]
    table = pd.read_csv(journal.root / "proposals.tsv", sep="\t")
    assert len(table) == 4
    assert set(table.proposal_id) == {r["proposal_id"] for r in rows}
    assert rows[1]["proposal_id"] in (journal.root / "proposals.fa").read_text()


@pytest.mark.parametrize("mode,energy", [("edit", -4), ("de_novo", 0)])
def test_seed_fallback_only_in_edit_mode(tmp_path, monkeypatch, mode, energy):
    mock_proposals(monkeypatch, ["CCCC"])
    journal = ProposalJournal(tmp_path, parent="AAAA", metadata={})
    state = diffusion_search(
        lambda batch: SimpleNamespace(energy=-batch[:, 0, 2:6].sum(1)),
        SimpleNamespace(editable=slice(2, 6)),
        one_hot("AAAAAAAA"),
        checkpoint="unused",
        activities={},
        mode=mode,
        journal=journal,
    )
    assert state.energy == energy
    selected = json.loads((journal.root / "selection.json").read_text())
    assert selected["baseline"] == (mode == "edit")


def test_completed_proposals_survive_oracle_failure(tmp_path, monkeypatch):
    mock_proposals(monkeypatch, ["ACCC", "ACCC"])
    journal = ProposalJournal(tmp_path, parent="CCCC", metadata={})

    def oracle(batch):
        raise RuntimeError("oracle interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        diffusion_search(
            oracle,
            SimpleNamespace(editable=slice(2, 6)),
            one_hot("CCCCCCCC"),
            checkpoint="unused",
            activities={},
            journal=journal,
        )
    snapshots = [
        json.loads(line) for line in (journal.root / "proposals.jsonl").read_text().splitlines()
    ]
    assert len(snapshots) == 3
    assert [row["sequence"] for row in snapshots] == ["CCCC", "ACCC", "ACCC"]
    assert all(row["status"] == "generated" for row in snapshots)
    assert not (journal.root / "selection.json").exists()


def test_full_wandb_records_and_artifact_and_failure_recovery(tmp_path):
    journal = ProposalJournal(tmp_path, parent="CCCC", metadata={"conditions": {"HL60": 2}})
    event_id = journal.add(dict(sequence="ACCC", masked_positions=[0], random_seed=3))
    journal.update(event_id, energy=-1.0, status="scored")
    journal.select(event_id)
    uploads = []

    class Artifact:
        def __init__(self, name, **kwargs):
            self.name = name

        def add_dir(self, directory):
            assert (Path(directory) / "selection.fa").exists()

    run = SimpleNamespace(
        log=lambda table: uploads.append(table),
        log_artifact=lambda artifact: uploads.append(artifact),
    )
    module = SimpleNamespace(Table=lambda **kwargs: kwargs, Artifact=Artifact)
    journal.upload(module, run, final=True)
    table = uploads[0]["sequence_proposals"]
    assert table["data"][0][table["columns"].index("sequence")] == "ACCC"
    assert table["data"][0][table["columns"].index("proposal_id")] == event_id
    assert len(uploads) == 2

    def broken(*args):
        raise RuntimeError("network unavailable")

    journal.upload(module, SimpleNamespace(log=broken), final=True)
    assert "network unavailable" in (journal.root / "upload_errors.jsonl").read_text()
    assert (journal.root / "selection.fa").exists()


def test_shard_merge_keeps_ids_and_records(tmp_path):
    journal = ProposalJournal(tmp_path / "shard", parent="CCCC", metadata={})
    event_id = journal.add(dict(sequence="ACCC"))
    merge_provenance([tmp_path / "shard"], tmp_path / "merged")
    copied = tmp_path / "merged" / journal.run_id / "proposals.jsonl"
    assert event_id in copied.read_text()
    assert copied.read_bytes() == (journal.root / "proposals.jsonl").read_bytes()


@pytest.mark.parametrize("contents", ["CCCC", ">one\nCCCC\n>two\nAAAA", ">one\n", ">one\nCCCN"])
def test_template_validation(tmp_path, contents):
    path = tmp_path / "template.fa"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_template(path)


def test_de_novo_matches_pre_edit_sampler_golden():
    class Fixed(RecordingModel):
        def forward(self, tokens, *args):
            return torch.tensor([0.0, 0.2, 0.5, 1.0]).expand(*tokens.shape, 4)

    # Recorded using the pre-edit sampler, not the implementation under test.
    assert sample(
        Fixed(), [1.0], [True], mode="de_novo", length=8, count=4, steps=4, guidance=1.5, seed=7
    ) == ["TCTCTTGT", "TACCGTTG", "TCCTGATG", "TGCTTTTT"]


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from regulonado.design.diffusion import Denoiser, DiffusionConfig

    config = DiffusionConfig(activity_columns=["target"], width=8, heads=2, layers=1, max_length=8)
    model = Denoiser(config)
    path = tmp_path / "model.pt"
    torch.save(
        dict(
            format_version=1,
            config=config.model_dump(),
            model=model.state_dict(),
            mean=[0.0],
            scale=[1.0],
        ),
        path,
    )
    return path


def test_cli_defaults_to_edit_and_keeps_matching_proposal_ids(tmp_path, tiny_checkpoint):
    from regulonado.cli.app import app
    from typer.testing import CliRunner

    template = tmp_path / "template.fa"
    template.write_text(">endogenous\nCCCCCCCC\n")
    out = tmp_path / "outputs"
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "sample",
            "--checkpoint",
            str(tiny_checkpoint),
            "--activities",
            '{"target": 2}',
            "--template-fasta",
            str(template),
            "--out",
            str(out),
            "--count",
            "4",
            "--steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((out / "run.json").read_text())["mode"] == "edit"
    records = pd.read_csv(next((out / "provenance").glob("*/proposals.tsv")), sep="\t")
    assert len(records) == 5
    selected = pd.read_csv(out / "sequences.tsv", sep="\t")
    assert set(selected.proposal_id) <= set(records.proposal_id)
    assert all(
        sum(a != b for a, b in zip(sequence, "CCCCCCCC")) <= 2 for sequence in selected.sequence
    )
    assert all(
        proposal_id in (out / "sequences.fa").read_text() for proposal_id in selected.proposal_id
    )


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "requires --template-fasta"),
        (["--template-fasta", "TEMPLATE", "--length", "7"], "conflicts"),
        (["--mode", "de_novo"], "requires --length"),
    ],
)
def test_cli_invalid_template_arguments(tmp_path, tiny_checkpoint, extra, message):
    from regulonado.cli.app import app
    from typer.testing import CliRunner

    template = tmp_path / "template.fa"
    template.write_text(">one\nCCCCCCCC\n")
    extra = [str(template) if argument == "TEMPLATE" else argument for argument in extra]
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "sample",
            "--checkpoint",
            str(tiny_checkpoint),
            "--activities",
            "{}",
            "--out",
            str(tmp_path / "outputs"),
            *extra,
        ],
    )
    assert result.exit_code != 0
    assert message in result.output


def test_design_example_defaults_and_per_target_override():
    import yaml
    from regulonado.config.models import DesignConfig

    payload = yaml.safe_load(Path("examples/enhancer_diffusion_edit.yaml").read_text())
    assert DesignConfig.model_validate(payload).diffusion_mode == "edit"
    payload.pop("diffusion_mode")
    assert DesignConfig.model_validate(payload).diffusion_mode == "edit"
    for key, value in [
        ("diffusion_mode", "compare"),
        ("diffusion_mask_fraction", 1.0),
        ("diffusion_mask_fraction", float("nan")),
    ]:
        with pytest.raises(ValueError):
            DesignConfig.model_validate({**payload, key: value})
    payload["targets"][0]["settings"] = {"diffusion_mode": "de_novo"}
    config = DesignConfig.model_validate(payload)
    assert config.targets[0].settings["diffusion_mode"] == "de_novo"


def test_encoder_checkpoint_supports_edit_without_retraining(tmp_path, monkeypatch):
    from regulonado.design import encoder_diffusion as encoder
    from regulonado.design.diffusion import Denoiser, DiffusionConfig
    from test_encoder_diffusion import ToyEncoder

    monkeypatch.setattr(encoder, "load_encoder", lambda *args: ToyEncoder())
    config = DiffusionConfig(
        architecture="alphagenome_encoder",
        encoder_checkpoint="unused",
        activity_columns=["target"],
        width=8,
        heads=2,
        layers=1,
        max_length=8,
    )
    model = Denoiser(config)
    records = sample(
        model, [1.0], [True], length=8, template="CCCCCCCC", count=2, steps=2, return_records=True
    )
    assert len(records) == 2
    for record in records:
        assert all(
            base == "C"
            for i, base in enumerate(record["sequence"])
            if i not in record["masked_positions"]
        )


def test_design_entrypoint_and_output_identity_survive_wandb_init_failure(
    tmp_path, tiny_checkpoint, monkeypatch
):
    import regulonado.genomics as genomics
    import yaml
    from regulonado.config.models import DesignConfig
    from regulonado.design import run
    from regulonado.design.report import write_designs

    payload = yaml.safe_load(Path("examples/enhancer_diffusion_edit.yaml").read_text())
    payload.update(
        out_dir=str(tmp_path / "design"),
        diffusion_checkpoint=str(tiny_checkpoint),
        objective="specificity",
        diffusion_samples=3,
    )
    config = DesignConfig.model_validate(payload)
    mock_proposals(monkeypatch, ["ACCC", "ACCC", "CCCC"])
    calls = []

    def oracle(batch):
        calls.append(batch.copy())
        target = batch[:, 0, 2:6].sum(1)
        return SimpleNamespace(energy=-target, target=target)

    monkeypatch.setattr(run, "_build_energy_fn", lambda *args: oracle)
    monkeypatch.setattr(genomics, "one_hot_context", lambda *args: one_hot("CCCCCCCC"))

    def unavailable(**kwargs):
        raise RuntimeError("W&B unavailable")

    wandb = SimpleNamespace(init=unavailable)
    seed = SimpleNamespace(
        name="endogenous",
        chrom="chr1",
        cand_start=2,
        cand_end=6,
        editable=slice(2, 6),
        bins=slice(0, 1),
        fold_label=None,
        window=SimpleNamespace(chrom="chr1", ctx_start=0, ctx_end=8, pred_start=0, pred_end=8),
    )
    record, _ = run._score_one_candidate(
        config,
        1,
        1,
        seed,
        "diffusion",
        SimpleNamespace(context_length=8),
        None,
        None,
        None,
        {"chr1": 8},
        None,
        17,
        wandb,
        config.targets[0],
    )
    assert len(calls) == 2  # repeated generation and final selection rescore are cached
    assert record.state.energy == -1
    # Test sequence/proposal links independently of predictor-specific result columns.
    record.result = None
    write_designs(tmp_path / "design", [record], run_info={})
    chosen = record.state.provenance["proposal_id"]
    selected = pd.read_csv(tmp_path / "design/designs.tsv", sep="\t")
    assert selected.proposal_id.tolist() == [chosen]
    assert chosen in (tmp_path / "design/designs.fa").read_text()
    root = Path(record.state.provenance["provenance_dir"])
    assert chosen in (root / "selection.fa").read_text()
    assert "W&B unavailable" in (root / "upload_errors.jsonl").read_text()
    trajectory = pd.read_csv(tmp_path / "design/trajectory.tsv", sep="\t")
    assert chosen in trajectory.proposal_id.tolist()
