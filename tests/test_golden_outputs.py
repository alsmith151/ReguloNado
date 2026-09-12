"""Golden-output regression tests (audit phase 0, "Pin behaviour").

These tests pin the *current* byte-for-byte output of two pipeline stages so
that later refactors (moving the window-derivation rule into a new module,
swapping BED parsing to bioframe, decomposing the dataset builder) can be
verified to be behaviour-preserving:

1. A tiny deterministic ``build_dataset`` (in_memory strategy) run —
   pinned as a sha256 content hash per split, computed over the Arrow row
   data itself (not file bytes, which can carry nondeterministic metadata
   such as timestamps or thread-order-dependent compression artifacts).
2. A tiny deterministic attribution run (dummy predictor) — pinned as the
   exact ``core_regions.bed`` text it produces.

If a change legitimately alters output, regenerate the fixtures with::

    REGULONADO_REGEN_GOLDEN=1 uv run --no-sync pytest tests/test_golden_outputs.py

This file is intentionally self-contained (no imports from other test
modules) so it can run independently of concurrent test-suite edits. The
tiny-genome/tiny-bigwig construction mirrors the approach used in
``tests/test_chrom_pass.py``; the tiny-predictor/attribution construction
mirrors ``tests/test_attribution.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools
import pytest
import torch

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
REGEN_ENV_VAR = "REGULONADO_REGEN_GOLDEN"


def _regen_enabled() -> bool:
    return os.environ.get(REGEN_ENV_VAR) == "1"


def _check_or_regen_text(golden_path: Path, actual: str, label: str) -> None:
    if _regen_enabled():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual)
        return
    expected = golden_path.read_text()
    assert actual == expected, (
        f"{label}: golden output changed — if intentional, regenerate with "
        f"{REGEN_ENV_VAR}=1"
    )


def _check_or_regen_json(golden_path: Path, actual: dict, label: str) -> None:
    if _regen_enabled():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        return
    expected = json.loads(golden_path.read_text())
    assert actual == expected, (
        f"{label}: golden output changed — if intentional, regenerate with "
        f"{REGEN_ENV_VAR}=1"
    )


# --------------------------------------------------------------------------- #
# 1. Dataset build golden (in_memory strategy)                                #
# --------------------------------------------------------------------------- #
# Deliberately tiny so the whole build (2 chroms x 2 tracks x 4 intervals)
# stays well under a second.
_N_TRACKS = 2
_CHROM_LEN = 1024
_BIN_SIZE = 4
_N_PRED_BINS = 16
_INTERVAL_BP = _N_PRED_BINS * _BIN_SIZE  # 64
_CONTEXT_LEN = _INTERVAL_BP  # no shift augmentation


def _build_tiny_dataset_inputs(root: Path) -> dict:
    """Deterministically construct a tiny FASTA + BigWigs + BED + tracks.parquet."""
    root.mkdir(parents=True, exist_ok=True)
    chroms = [("chrA", _CHROM_LEN), ("chrB", _CHROM_LEN)]

    fasta_path = root / "ref.fa"
    with fasta_path.open("w") as fh:
        for name, n in chroms:
            fh.write(f">{name}\n")
            seq = ("ACGT" * ((n + 3) // 4))[:n]
            for i in range(0, n, 80):
                fh.write(seq[i : i + 80] + "\n")
    import pyfaidx

    pyfaidx.Fasta(str(fasta_path))  # writes the .fai companion

    bw_paths: list[str] = []
    rng = np.random.default_rng(0)
    for t in range(_N_TRACKS):
        path = root / f"track_{t}.bw"
        chromsize_map = {name: n for name, n in chroms}
        entries: list[tuple[str, int, int, float]] = []
        for chrom_idx, (name, n) in enumerate(chroms):
            xs = np.arange(n)
            vals = (
                np.sin(2 * np.pi * (xs + 17 * t) / 128.0) * 5.0
                + 10.0
                + t * 2.0
                + chrom_idx * 0.5
                + rng.normal(0, 0.1, size=n)
            ).astype(np.float32)
            entries.extend((name, int(i), int(i + 1), float(v)) for i, v in enumerate(vals))
        w = pybigtools.open(str(path), "w")
        w.write(chromsize_map, iter(entries))
        bw_paths.append(str(path))

    # Two intervals per chromosome, interleaved, alternating fold assignment.
    starts = [0, 512]
    bed_path = root / "intervals.bed"
    folds = ["fold0", "fold1"]
    with bed_path.open("w") as fh:
        for i, s in enumerate(starts):
            for c_idx, (cname, _) in enumerate(chroms):
                e = s + _INTERVAL_BP
                fold = folds[(i + c_idx) % 2]
                fh.write(f"{cname}\t{s}\t{e}\t{fold}\n")

    track_names = [f"track_{t}" for t in range(_N_TRACKS)]
    tracks_parquet = root / "tracks.parquet"
    from regulonado.tracks_table import write_track_table

    write_track_table(
        pd.DataFrame(
            {
                "track_name": track_names,
                "status": ["included"] * _N_TRACKS,
                "track_index": list(range(_N_TRACKS)),
                "path": bw_paths,
                "resolved_path": bw_paths,
            }
        ),
        tracks_parquet,
    )

    return {
        "fasta": str(fasta_path),
        "bed_path": str(bed_path),
        "tracks_parquet": str(tracks_parquet),
    }


def _hash_arrow_split_dir(split_dir: Path) -> str:
    """sha256 over a split's Arrow row data, in on-disk row order.

    Reads shard files directly (sorted by filename, matching the order the
    in_memory writer produces them in) and hashes each record batch's
    columns, sorted by column name so key order can't introduce noise.
    Uses ``repr()`` of the decoded Python values rather than raw dtype bytes
    so the hash is insensitive to incidental numpy dtype choices while still
    being exact and order-sensitive over the actual row content.
    """
    import pyarrow.ipc as ipc

    hasher = hashlib.sha256()
    shards = sorted(split_dir.glob("data-*-of-*.arrow"))
    assert shards, f"no Arrow shards found in {split_dir}"
    for shard in shards:
        with shard.open("rb") as fh:
            reader = ipc.open_stream(fh)
            for batch in reader:
                tbl = batch.to_pydict()
                for col in sorted(tbl.keys()):
                    hasher.update(col.encode())
                    hasher.update(repr(tbl[col]).encode())
    return hasher.hexdigest()


def _run_tiny_build(inputs: dict, output_dir: Path) -> dict[str, str]:
    from regulonado.dataset import build_dataset

    build_dataset(
        inputs["bed_path"],
        inputs["fasta"],
        inputs["tracks_parquet"],
        output_dir,
        splits={"train": ["fold0"], "validation": ["fold1"]},
        context_length=_CONTEXT_LEN,
        bin_size=_BIN_SIZE,
        n_pred_bins=_N_PRED_BINS,
        shift_max_bp=0,
        n_extract_threads=2,
        arrow_batch_size=2,
        arrow_write_threads=1,
        strategy="in_memory",
        stage_to_scratch=False,
        return_dataset=False,
    )
    return {
        split: _hash_arrow_split_dir(output_dir / split)
        for split in sorted(("train", "validation"))
    }


def test_dataset_build_golden_output(tmp_path):
    """in_memory dataset build output must match the pinned content hash."""
    inputs = _build_tiny_dataset_inputs(tmp_path / "src")
    hashes = _run_tiny_build(inputs, tmp_path / "out")
    _check_or_regen_json(
        GOLDEN_DIR / "dataset_build_golden.json",
        hashes,
        "dataset build (in_memory)",
    )


def test_dataset_build_is_deterministic_across_runs(tmp_path):
    """The same tiny build run three times must produce identical content hashes."""
    inputs = _build_tiny_dataset_inputs(tmp_path / "src")
    runs = [
        _run_tiny_build(inputs, tmp_path / f"out{i}")
        for i in range(3)
    ]
    assert runs[0] == runs[1] == runs[2], (
        "dataset build (in_memory) is not deterministic across repeated runs "
        f"with identical inputs: {runs}"
    )


# --------------------------------------------------------------------------- #
# 2. Attribution golden (core_regions.bed)                                    #
# --------------------------------------------------------------------------- #
# Tiny geometry, matching tests/test_attribution.py's convention.
_ATTR_N_PRED_BINS = 8
_ATTR_BIN_SIZE = 10
_ATTR_PRED_BP = _ATTR_N_PRED_BINS * _ATTR_BIN_SIZE  # 80
_ATTR_CONTEXT = 400
_ATTR_MOTIF = slice(200, 240)


class _MotifEnsemble:
    """2 folds x 3 tracks. Only track 1 responds, and only to 'A' content inside the motif."""

    track_names = ["t0", "t1", "t2"]

    def __init__(self, n_folds: int = 2, fold_scale: list[float] | None = None) -> None:
        self.n_folds = n_folds
        self.fold_scale = fold_scale or [1.0] * n_folds

    def predict(self, one_hot_batch):
        x = torch.as_tensor(np.asarray(one_hot_batch)).float()
        batch = x.shape[0]
        signal = x[:, 0, _ATTR_MOTIF].sum(dim=1)  # A-count inside the motif
        out = torch.zeros(self.n_folds, batch, 3, _ATTR_N_PRED_BINS)
        for fold, scale in enumerate(self.fold_scale):
            out[fold, :, 1, :] = signal.view(batch, 1) * scale
        return out


def _attr_window():
    from regulonado.genomics import Window

    return Window(
        chrom="chr1",
        pred_start=_ATTR_CONTEXT // 2 - _ATTR_PRED_BP // 2,
        pred_end=_ATTR_CONTEXT // 2 + _ATTR_PRED_BP // 2,
        ctx_start=0,
        ctx_end=_ATTR_CONTEXT,
    )


def _attr_seed(name: str = "c0", start: int = 180, end: int = 280):
    from regulonado.design.sequence import Seed

    return Seed(
        name=name,
        chrom="chr1",
        cand_start=start,
        cand_end=end,
        window=_attr_window(),
        fold_label="test",
        editable=slice(start, end),
        bins=slice(0, _ATTR_N_PRED_BINS),
    )


def _attr_motif_context(seed_value: int = 0) -> np.ndarray:
    from regulonado.genomics import one_hot

    rng = np.random.default_rng(seed_value)
    context = one_hot("".join(rng.choice(list("ACGT"), _ATTR_CONTEXT)))
    context[:, _ATTR_MOTIF] = 0
    context[0, _ATTR_MOTIF] = 1  # a pure poly-A block
    return context


def _run_tiny_attribution(out_dir: Path) -> str:
    from regulonado.design.attribution import (
        AttributionRecord,
        TrackReadout,
        _smooth,
        call_cores,
        ism_scan,
        write_attributions,
    )

    seed = _attr_seed()
    context = _attr_motif_context()
    readout = TrackReadout(_MotifEnsemble(), track_index=1, bins=seed.bins)
    result = ism_scan(readout, seed, context, batch_size=32)
    cores, diagnostics = call_cores(
        result.importance,
        editable=seed.editable,
        smooth_bp=11,
        quantile=0.75,
        min_width_bp=20,
        merge_gap_bp=5,
        min_zscore=1.0,
        bounds=(seed.window.pred_start, seed.window.pred_end),
    )
    record = AttributionRecord(
        seed=seed,
        ism=result,
        smoothed=_smooth(result.importance, 11),
        cores=cores,
        diagnostics=diagnostics,
    )
    write_attributions(
        out_dir,
        [record],
        run_info={"status": "ok"},
        chrom_sizes={"chr1": _ATTR_CONTEXT},
        bigwig=False,
    )
    return (out_dir / "core_regions.bed").read_text()


def test_attribution_core_regions_golden_output(tmp_path):
    """A deterministic dummy-predictor attribution run must match the pinned BED text."""
    actual = _run_tiny_attribution(tmp_path)
    _check_or_regen_text(
        GOLDEN_DIR / "attribution_core_regions.bed",
        actual,
        "attribution core_regions.bed",
    )


def test_attribution_core_regions_is_deterministic_across_runs(tmp_path):
    """The same tiny attribution run three times must produce byte-identical BED text."""
    runs = [
        _run_tiny_attribution(tmp_path / f"run{i}")
        for i in range(3)
    ]
    assert runs[0] == runs[1] == runs[2], "attribution core_regions.bed is not deterministic"


@pytest.mark.skipif(
    not GOLDEN_DIR.exists(), reason="golden fixtures not yet generated"
)
def test_golden_fixtures_exist():
    """Guard against an accidental empty-fixture regen wiping out the pinned outputs."""
    assert (GOLDEN_DIR / "dataset_build_golden.json").exists()
    assert (GOLDEN_DIR / "attribution_core_regions.bed").exists()
