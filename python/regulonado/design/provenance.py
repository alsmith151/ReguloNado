"""Durable sequence records shared by standalone sampling and oracle search."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def sequence_identity(sequence):
    return hashlib.sha256(sequence.upper().encode("ascii")).hexdigest()


def checkpoint_identity(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ProposalJournal:
    """One invocation's append-only snapshots; JSONL is authoritative after interruption."""

    def __init__(self, root, *, parent, metadata):
        self.run_id = uuid4().hex
        self.root = Path(root) / self.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.parent = parent.upper() if parent is not None else None
        self.metadata = metadata
        self.records = {}
        self._json(
            "provenance.json",
            dict(
                run_id=self.run_id,
                parent_sequence=self.parent,
                parent_sequence_id=sequence_identity(self.parent) if self.parent else None,
                **metadata,
            ),
        )

    def _json(self, name, value):
        temporary = self.root / (name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(self.root / name)

    def _append(self, name, value):
        with (self.root / name).open("a") as handle:
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def add(self, proposal, *, baseline=False):
        sequence = proposal["sequence"].upper()
        event_id = f"{self.run_id}-{len(self.records):06d}"
        row = {**self.metadata, **proposal}
        row.update(
            proposal_id=event_id,
            sequence=sequence,
            sequence_id=sequence_identity(sequence),
            parent_sequence=self.parent,
            parent_sequence_id=sequence_identity(self.parent) if self.parent else None,
            status="generated",
            baseline=baseline,
            substitutions=(
                [i for i, (a, b) in enumerate(zip(self.parent, sequence)) if a != b]
                if self.parent is not None
                else []
            ),
        )
        self.records[event_id] = row
        self._append("proposals.jsonl", row)
        with (self.root / "proposals.fa").open("a") as handle:
            handle.write(f">{event_id} sequence_id={row['sequence_id']}\n{sequence}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._table()
        return event_id

    def update(self, event_id, **values):
        self.records[event_id].update(values)
        self._append("proposals.jsonl", self.records[event_id])
        self._table()

    def _table(self):
        rows = list(self.records.values())
        keys = list(dict.fromkeys(key for row in rows for key in row))
        temporary = self.root / "proposals.tsv.tmp"
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in rows
            )
        temporary.replace(self.root / "proposals.tsv")

    def select(self, event_id, **budgets):
        for candidate_id in self.records:
            self.update(candidate_id, selected=candidate_id == event_id)
        row = dict(self.records[event_id], **budgets)
        self._json("selection.json", row)
        (self.root / "selection.fa").write_text(
            f">{event_id} sequence_id={row['sequence_id']}\n{row['sequence']}\n"
        )
        return row

    def upload(self, wandb_module, run, *, final=False):
        if run is None:
            return
        try:
            rows = list(self.records.values())
            columns = list(dict.fromkeys(k for row in rows for k in row))
            table = wandb_module.Table(
                columns=columns,
                data=[
                    [
                        json.dumps(row.get(k))
                        if isinstance(row.get(k), (list, dict))
                        else row.get(k)
                        for k in columns
                    ]
                    for row in rows
                ],
            )
            run.log({"sequence_proposals": table})
            if final:
                artifact = wandb_module.Artifact(
                    f"diffusion-sequences-{self.run_id}",
                    type="sequence-design",
                    metadata={"run_id": self.run_id},
                )
                artifact.add_dir(str(self.root))
                run.log_artifact(artifact)
        except Exception as exc:
            logger.warning("W&B sequence upload failed; records remain at %s: %s", self.root, exc)
            self._append("upload_errors.jsonl", {"error": str(exc), "final": final})


def read_template(path):
    """Read exactly one nonempty FASTA record without creating an index file."""
    records, current = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records.append("".join(current))
            current = []
        elif current is None:
            raise ValueError("Template must be FASTA with a record header")
        else:
            current.append(line)
    if current is not None:
        records.append("".join(current))
    if len(records) != 1 or not records[0] or set(records[0].upper()) - set("ACGT"):
        raise ValueError("Template FASTA must contain exactly one nonempty A/C/G/T sequence")
    return records[0].upper()


def merge_provenance(sources, destination):
    """Copy each shard journal verbatim; IDs and sequence identities never change."""
    import shutil

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for journal in Path(source).iterdir():
            if not journal.is_dir():
                continue
            target = destination / journal.name
            if target.exists():
                raise ValueError(f"Duplicate provenance journal ID: {journal.name}")
            shutil.copytree(journal, target)
