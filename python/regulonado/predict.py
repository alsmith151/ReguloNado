"""Generate BigWig tracks from trained Borzoi/Enformer model predictions.

The model emits a dense ``(n_tracks, n_pred_bins)`` array per window, covering only the
central ``n_pred_bins * bin_size`` bp of the input context (the backbone crops the flanks).
Writing that naively is one ``bin_size`` interval per bin and produces huge files, so adjacent
near-equal bins are run-length collapsed (relative tolerance) into single wide intervals before
writing with ``pybigtools``.

Two prediction modes are supported:

* **targeted** — predict the central window centred on each BED row.
* **whole-genome** — tile each chromosome into adjacent, non-overlapping prediction windows.

Because only the central region is predicted and windows are stepped by the prediction width,
windows never overlap on the genome ("center crop only").

Pure Python; ``torch``/``pyfaidx``/``pybigtools`` are imported lazily so the lightweight helpers
(``iter_windows``, ``collapse_bins``) import without a GPU stack.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Base -> one-hot row. Unknown bases (incl. N) map to an all-zero column.
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for _base, _row in {"A": 0, "C": 1, "G": 2, "T": 3}.items():
    _BASE_LUT[ord(_base)] = _row
    _BASE_LUT[ord(_base.lower())] = _row


@dataclass(slots=True)
class Window:
    """A single prediction window: predicted region + surrounding model context."""

    chrom: str
    pred_start: int  # start of the predicted (central) region, bp
    pred_end: int  # end of the predicted region, bp (== pred_start + n_pred_bins*bin_size)
    ctx_start: int  # start of the model input context, bp (may be < 0 near chrom start)
    ctx_end: int  # end of the model input context, bp (may be > chrom length near chrom end)


@dataclass(slots=True)
class RegionPredictionConfig:
    """Configuration for reusing a trained model for coordinate-based predictions."""

    checkpoint_dir: str | Path
    fasta_path: str | Path
    dataset_dir: str | Path | None = None
    chromsizes_path: str | Path | None = None
    tracks: Sequence[str] | None = None
    device: str | None = None
    inverse_squash: bool = False


@dataclass(slots=True)
class RegionPrediction:
    """Prediction values and genomic bin coordinates for one requested interval."""

    chrom: str
    query_start: int
    query_end: int
    pred_start: int
    pred_end: int
    bin_size: int
    track_names: list[str]
    values: np.ndarray

    @property
    def bin_starts(self) -> np.ndarray:
        return self.pred_start + np.arange(self.values.shape[-1]) * self.bin_size

    @property
    def bin_ends(self) -> np.ndarray:
        return self.bin_starts + self.bin_size

    def as_records(self) -> list[dict[str, str | int | float]]:
        """Return long-form records: one row per track/bin."""
        records: list[dict[str, str | int | float]] = []
        starts = self.bin_starts
        ends = self.bin_ends
        for track_index, track_name in enumerate(self.track_names):
            for bin_index, value in enumerate(self.values[track_index]):
                records.append(
                    {
                        "chrom": self.chrom,
                        "start": int(starts[bin_index]),
                        "end": int(ends[bin_index]),
                        "track": track_name,
                        "value": float(value),
                    }
                )
        return records


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def _unique_track_names(records: Sequence[dict]) -> list[str]:
    """Derive filesystem-safe, unique track names from BigWig basenames."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        path = record.get("path") or record.get("resolved_path") or f"track{index}"
        stem = Path(str(path)).name
        for suffix in (".bigWig", ".bigwig", ".bw", ".bedGraph", ".bedgraph"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem) or f"track{index}"
        if stem in seen:
            seen[stem] += 1
            stem = f"{stem}_{seen[stem]}"
        else:
            seen[stem] = 0
        names.append(stem)
    return names


def _find_config(checkpoint_dir: Path) -> Path:
    """Locate resolved_config.json in checkpoint_dir or its parent (HF Trainer layout)."""
    local = checkpoint_dir / "resolved_config.json"
    if local.exists():
        return local
    parent = checkpoint_dir.parent / "resolved_config.json"
    if parent.exists():
        return parent
    raise FileNotFoundError(
        f"resolved_config.json not found in {checkpoint_dir} or {checkpoint_dir.parent}"
    )


def _find_weights(checkpoint_dir: Path) -> Path:
    """Locate model weights in checkpoint_dir or its checkpoint-* subdirectories.

    When checkpoint_dir is a run root (no weights directly), reads trainer_state.json from
    the latest checkpoint to find the best checkpoint, then falls back to the latest if
    trainer_state.json is absent or its best_model_checkpoint path has no weights.
    """
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = checkpoint_dir / name
        if p.exists():
            return p

    subdirs = sorted(
        checkpoint_dir.glob("checkpoint-*/"),
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir} or its checkpoint-* subdirectories"
        )

    # Read trainer_state.json from the latest checkpoint to find the best checkpoint.
    latest = subdirs[-1]
    state_path = latest / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        best_ckpt = state.get("best_model_checkpoint")
        if best_ckpt:
            best_dir = Path(best_ckpt)
            for name in ("model.safetensors", "pytorch_model.bin"):
                p = best_dir / name
                if p.exists():
                    return p

    # Fall back to latest checkpoint.
    for subdir in reversed(subdirs):
        for name in ("model.safetensors", "pytorch_model.bin"):
            p = subdir / name
            if p.exists():
                return p

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_dir} or its checkpoint-* subdirectories"
    )


def load_model_for_inference(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path | None = None,
    device: str | None = None,
) -> object:
    """Load a ``RegulonadoModel`` for prediction.

    HF model directories with ``config.json`` are loaded directly via
    ``RegulonadoModel.from_pretrained``. Legacy run roots without ``config.json`` fall back to
    rebuilding from ``resolved_config.json`` and dataset metadata.

    ``dataset_dir`` is only needed for legacy checkpoints whose resolved config does not contain
    ``data.path``.
    """
    import torch

    from regulonado.model import RegulonadoModel, build_backbone_adapter
    from regulonado.train import (
        build_model,
        load_dataset_metadata,
        load_model_weights_only,
    )

    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "config.json").exists():
        model = RegulonadoModel.from_pretrained(checkpoint_dir)
        model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)

    config_path = _find_config(checkpoint_dir)
    cfg = json.loads(config_path.read_text())

    if dataset_dir is None:
        try:
            dataset_dir = Path(cfg["data"]["path"])
        except (KeyError, TypeError):
            raise ValueError(
                "dataset_dir was not provided and could not be inferred from "
                f"resolved_config.json (no 'data.path' key) at {config_path}"
            )
    else:
        dataset_dir = Path(dataset_dir)

    metadata = load_dataset_metadata(dataset_dir)
    records = list(metadata.get("final_track_records") or [])
    if not records:
        raise ValueError(f"No 'final_track_records' in dataset metadata under {dataset_dir}")

    # Keep pretrained_name so from_pretrained builds the exact architecture used during training
    # (e.g. flash-attention key layout for flashzoi checkpoints). The pretrained weights are
    # overwritten by load_model_weights_only below; HF caches them locally after first download.
    # Only set allow_random_init if there is no pretrained_name, to bypass the training guard.
    backbone_cfg = dict(cfg.get("backbone") or {})
    if not backbone_cfg.get("pretrained_name"):
        backbone_cfg["allow_random_init"] = True
        cfg = {**cfg, "backbone": backbone_cfg}

    weight_path = _find_weights(checkpoint_dir)
    model = build_model(cfg, metadata, records, build_backbone_adapter)
    load_model_weights_only(model, weight_path)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def _model_track_metadata(model: object, device: str) -> dict[str, object]:
    """Convert config-stored constant track metadata to tensors for metadata-conditioned heads."""
    import torch

    config = getattr(model, "config", None)
    metadata = getattr(config, "track_metadata", {}) or {}
    tensors: dict[str, object] = {}
    for key, values in metadata.items():
        if key == "track_timepoint_minutes":
            values = [float("nan") if value is None else value for value in values]
            tensors[key] = torch.as_tensor(values, dtype=torch.float32, device=device)
        else:
            tensors[key] = torch.as_tensor(values, dtype=torch.long, device=device)
    return tensors


# --------------------------------------------------------------------------- #
# Window enumeration                                                          #
# --------------------------------------------------------------------------- #
def _parse_bed(bed_path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for line in bed_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "track", "browser")):
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < 3:
            continue
        rows.append((fields[0], int(fields[1]), int(fields[2])))
    return rows


def read_chrom_sizes(path: str | Path) -> dict[str, int]:
    """Parse a two-column ``chrom<TAB>size`` chrom.sizes file, preserving order."""
    sizes: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < 2:
            continue
        sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError(f"No chromosome sizes parsed from {path}")
    return sizes


def iter_windows(
    *,
    chrom_sizes: dict[str, int],
    context_length: int,
    n_pred_bins: int,
    bin_size: int,
    bed_path: str | Path | None = None,
    whole_genome: bool = False,
) -> list[Window]:
    """Enumerate prediction windows for targeted (BED) or whole-genome tiling.

    Whole-genome mode tiles every chromosome in ``chrom_sizes`` (i.e. the chrom.sizes file
    decides which chromosomes are tiled and how long they are). Windows are returned sorted by
    ``(chromosome order, pred_start)``. In targeted mode an overlap between two predicted regions
    raises (BigWig requires non-overlapping intervals).
    """
    if (bed_path is None) == (not whole_genome):
        raise ValueError("Provide exactly one of bed_path (targeted) or whole_genome=True")

    pred_bp = n_pred_bins * bin_size
    chrom_rank = {name: i for i, name in enumerate(chrom_sizes)}
    windows: list[Window] = []

    if bed_path is not None:
        for chrom, start, end in _parse_bed(Path(bed_path)):
            if chrom not in chrom_sizes:
                raise ValueError(f"BED chromosome {chrom!r} not present in chrom sizes")
            center = (start + end) // 2
            pred_start = center - pred_bp // 2
            ctx_start = center - context_length // 2
            windows.append(
                Window(
                    chrom,
                    pred_start,
                    pred_start + pred_bp,
                    ctx_start,
                    ctx_start + context_length,
                )
            )
    else:
        for chrom, length in chrom_sizes.items():
            for pred_start in range(0, length, pred_bp):
                center = pred_start + pred_bp // 2
                ctx_start = center - context_length // 2
                windows.append(
                    Window(
                        chrom,
                        pred_start,
                        pred_start + pred_bp,
                        ctx_start,
                        ctx_start + context_length,
                    )
                )

    windows.sort(key=lambda w: (chrom_rank[w.chrom], w.pred_start))

    # Reject overlapping predicted regions (would corrupt the BigWig).
    for prev, cur in zip(windows, windows[1:]):
        if cur.chrom == prev.chrom and cur.pred_start < prev.pred_end:
            raise ValueError(
                f"Overlapping prediction windows on {cur.chrom}: "
                f"[{prev.pred_start},{prev.pred_end}) and [{cur.pred_start},{cur.pred_end}). "
                "Targeted regions must be further apart than the prediction window."
            )
    return windows


# --------------------------------------------------------------------------- #
# Sequence one-hot                                                            #
# --------------------------------------------------------------------------- #
def one_hot_context(fasta, window: Window, context_length: int, chrom_length: int) -> np.ndarray:
    """One-hot encode the model context for ``window`` as an ``int8`` ``(4, context_length)`` array.

    Positions running off either chromosome end are zero-padded (treated as N).
    """
    out = np.zeros((4, context_length), dtype=np.int8)
    fetch_start = max(0, window.ctx_start)
    fetch_end = min(chrom_length, window.ctx_end)
    if fetch_end <= fetch_start:
        return out

    seq = str(fasta[window.chrom][fetch_start:fetch_end])
    codes = _BASE_LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    valid = codes >= 0
    offset = fetch_start - window.ctx_start
    cols = np.arange(offset, offset + codes.shape[0])[valid]
    out[codes[valid], cols] = 1
    return out


# --------------------------------------------------------------------------- #
# Bin collapsing                                                              #
# --------------------------------------------------------------------------- #
def collapse_bins(
    values: np.ndarray,
    chrom: str,
    pred_start: int,
    bin_size: int,
    rtol: float,
    chrom_length: int,
) -> list[tuple[str, int, int, float]]:
    """Run-length collapse per-bin values into ``(chrom, start, end, value)`` intervals.

    Adjacent bins are merged while the next bin is within a relative tolerance of the current
    run's mean: ``abs(v - mean) <= rtol * max(abs(v), abs(mean))`` (so flat zero regions, where
    both are 0, always merge). Bins are clamped to ``[0, chrom_length)``; empty after clamping
    are skipped.
    """
    intervals: list[tuple[str, int, int, float]] = []
    run_start = run_end = -1
    run_sum = 0.0
    run_count = 0

    def flush() -> None:
        if run_count:
            intervals.append((chrom, run_start, run_end, run_sum / run_count))

    for bin_index, raw in enumerate(values):
        start = pred_start + bin_index * bin_size
        end = start + bin_size
        start = max(start, 0)
        end = min(end, chrom_length)
        if end <= start:
            continue
        v = float(raw)
        if run_count and start == run_end:
            mean = run_sum / run_count
            if abs(v - mean) <= rtol * max(abs(v), abs(mean)):
                run_end = end
                run_sum += v
                run_count += 1
                continue
        flush()
        run_start, run_end, run_sum, run_count = start, end, v, 1
    flush()
    return intervals


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _resolve_tracks(tracks: Sequence[str] | None, track_names: Sequence[str]) -> list[int]:
    if not tracks:
        return list(range(len(track_names)))
    name_to_index = {name: i for i, name in enumerate(track_names)}
    selected: list[int] = []
    for token in tracks:
        token = str(token).strip()
        if token in name_to_index:
            selected.append(name_to_index[token])
        elif token.lstrip("-").isdigit() and 0 <= int(token) < len(track_names):
            selected.append(int(token))
        else:
            raise ValueError(
                f"Unknown track {token!r}; expected a track name or an index in "
                f"[0, {len(track_names)})"
            )
    # de-duplicate, preserve order
    return list(dict.fromkeys(selected))


def _chunks(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class RegionPredictor:
    """Reusable coordinate-based predictor for quick notebook or script inference.

    Example:
        ``pred = RegionPredictor(RegionPredictionConfig(checkpoint_dir, fasta_path))(\
        "chr1", 1_000_000, 1_001_000)``
    """

    def __init__(self, config: RegionPredictionConfig):
        import pyfaidx

        self.config = config
        self.model = load_model_for_inference(
            config.checkpoint_dir,
            config.dataset_dir,
            config.device,
        )
        model_config = self.model.config
        self.context_length = int(model_config.context_length)
        self.n_pred_bins = int(model_config.n_pred_bins)
        self.bin_size = int(model_config.bin_size)
        self.track_names = list(
            model_config.track_names or [f"track{i}" for i in range(int(model_config.n_tracks))]
        )
        self.selected_tracks = _resolve_tracks(config.tracks, self.track_names)
        self.model_device = str(next(self.model.parameters()).device)
        self.track_metadata = _model_track_metadata(self.model, self.model_device)
        self.fasta = pyfaidx.Fasta(
            str(config.fasta_path),
            as_raw=True,
            sequence_always_upper=False,
        )
        if config.chromsizes_path is not None:
            self.chrom_sizes = read_chrom_sizes(config.chromsizes_path)
        else:
            self.chrom_sizes = {name: len(self.fasta[name]) for name in self.fasta.keys()}

    @classmethod
    def from_paths(
        cls,
        checkpoint_dir: str | Path,
        fasta_path: str | Path,
        **kwargs: object,
    ) -> "RegionPredictor":
        """Construct from paths while passing optional ``RegionPredictionConfig`` fields."""
        return cls(RegionPredictionConfig(checkpoint_dir, fasta_path, **kwargs))

    def __call__(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        tracks: Sequence[str] | None = None,
    ) -> RegionPrediction:
        return self.predict(chrom, start, end, tracks=tracks)

    def predict(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        tracks: Sequence[str] | None = None,
    ) -> RegionPrediction:
        """Predict the model's central output window centred on ``chrom:start-end``.

        ``start`` and ``end`` are zero-based half-open genomic coordinates. Passing the same
        value for both makes a point prediction centred on that coordinate.
        """
        import torch

        if chrom not in self.chrom_sizes:
            raise ValueError(f"Chromosome {chrom!r} is not present in the configured FASTA/sizes")
        if start < 0 or end < start:
            raise ValueError("Coordinates must satisfy 0 <= start <= end")

        center = (start + end) // 2
        pred_bp = self.n_pred_bins * self.bin_size
        pred_start = center - pred_bp // 2
        window = Window(
            chrom=chrom,
            pred_start=pred_start,
            pred_end=pred_start + pred_bp,
            ctx_start=center - self.context_length // 2,
            ctx_end=center - self.context_length // 2 + self.context_length,
        )
        seq = one_hot_context(
            self.fasta,
            window,
            self.context_length,
            self.chrom_sizes[chrom],
        )
        x = torch.from_numpy(seq[None]).to(dtype=torch.float32, device=self.model_device)
        with torch.no_grad():
            preds = self.model(x, **self.track_metadata)
        values = preds[0].float().cpu().numpy()
        if self.config.inverse_squash:
            from regulonado.dataset import inverse_transform_signal

            values = inverse_transform_signal(values, apply_squash=True, apply_scale=False)

        selected = (
            _resolve_tracks(tracks, self.track_names)
            if tracks is not None
            else self.selected_tracks
        )
        selected_values = values[selected]
        selected_names = [self.track_names[index] for index in selected]
        pred_end = pred_start + selected_values.shape[-1] * self.bin_size
        return RegionPrediction(
            chrom=chrom,
            query_start=start,
            query_end=end,
            pred_start=pred_start,
            pred_end=pred_end,
            bin_size=self.bin_size,
            track_names=selected_names,
            values=selected_values,
        )

    def predict_many(
        self,
        coordinates: Sequence[tuple[str, int, int]],
        *,
        tracks: Sequence[str] | None = None,
    ) -> list[RegionPrediction]:
        """Predict multiple coordinate intervals with the loaded model and FASTA."""
        return [self.predict(chrom, start, end, tracks=tracks) for chrom, start, end in coordinates]


def predict_to_bigwig(
    checkpoint_dir: str | Path,
    fasta_path: str | Path,
    out_dir: str | Path,
    *,
    dataset_dir: str | Path,
    bed_path: str | Path | None = None,
    whole_genome: bool = False,
    chromsizes_path: str | Path | None = None,
    tracks: Sequence[str] | None = None,
    rtol: float = 0.01,
    batch_size: int = 4,
    device: str | None = None,
    inverse_squash: bool = False,
) -> list[Path]:
    """Run inference over windows and write one collapsed BigWig per selected track.

    ``chromsizes_path`` (a ``chrom<TAB>size`` file) is the authoritative set of chromosomes and
    lengths used to tile (whole-genome mode), clamp intervals, and write the BigWig header. When
    omitted, sizes fall back to the FASTA ``.fai`` index (every contig in the FASTA). Returns the
    list of written BigWig paths.
    """
    import pyfaidx
    import torch

    log.info("Loading model from %s", checkpoint_dir)
    model = load_model_for_inference(checkpoint_dir, dataset_dir, device)
    config = model.config
    context_length = int(config.context_length)
    n_pred_bins = int(config.n_pred_bins)
    bin_size = int(config.bin_size)
    track_names = list(config.track_names or [f"track{i}" for i in range(int(config.n_tracks))])
    model_device = str(next(model.parameters()).device)
    track_metadata = _model_track_metadata(model, model_device)
    log.info(
        "Model ready — %d tracks, %d bins × %d bp, device=%s",
        len(track_names),
        n_pred_bins,
        bin_size,
        model_device,
    )

    fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=False)
    if chromsizes_path is not None:
        chrom_sizes = read_chrom_sizes(chromsizes_path)
    else:
        chrom_sizes = {name: len(fasta[name]) for name in fasta.keys()}

    windows = iter_windows(
        chrom_sizes=chrom_sizes,
        context_length=context_length,
        n_pred_bins=n_pred_bins,
        bin_size=bin_size,
        bed_path=bed_path,
        whole_genome=whole_genome,
    )
    if not windows:
        raise ValueError("No prediction windows were produced (empty BED or chromosome list)")

    selected = _resolve_tracks(tracks, track_names)
    n_batches = (len(windows) + batch_size - 1) // batch_size
    log.info(
        "%d windows across %d chromosome(s), %d selected track(s), %d batch(es)",
        len(windows),
        len({w.chrom for w in windows}),
        len(selected),
        n_batches,
    )

    accum: dict[int, list[tuple[str, int, int, float]]] = {t: [] for t in selected}

    if inverse_squash:
        from regulonado.dataset import inverse_transform_signal

    from tqdm import tqdm

    current_chrom = ""
    bar = tqdm(total=len(windows), unit="win", desc="predicting")
    for batch in _chunks(windows, batch_size):
        seqs = np.stack(
            [one_hot_context(fasta, w, context_length, chrom_sizes[w.chrom]) for w in batch]
        )
        x = torch.from_numpy(seqs).to(dtype=torch.float32, device=model_device)
        with torch.no_grad():
            preds = model(x, **track_metadata)
        preds = preds.float().cpu().numpy()  # (B, n_tracks, n_pred_bins)
        if inverse_squash:
            preds = inverse_transform_signal(preds, apply_squash=True, apply_scale=False)
        for batch_index, window in enumerate(batch):
            if window.chrom != current_chrom:
                current_chrom = window.chrom
                bar.set_postfix(chrom=current_chrom)
            chrom_length = chrom_sizes[window.chrom]
            for track in selected:
                accum[track].extend(
                    collapse_bins(
                        preds[batch_index, track],
                        window.chrom,
                        window.pred_start,
                        bin_size,
                        rtol,
                        chrom_length,
                    )
                )
        bar.update(len(batch))
    bar.close()

    log.info("Writing %d BigWig(s) to %s", len(selected), out_dir)
    written = _write_bigwigs(out_dir, track_names, selected, accum, chrom_sizes)
    log.info("Done — wrote %s", ", ".join(str(p.name) for p in written))
    return written


def _write_bigwigs(
    out_dir: str | Path,
    track_names: Sequence[str],
    selected: Sequence[int],
    accum: dict[int, list[tuple[str, int, int, float]]],
    chrom_sizes: dict[str, int],
) -> list[Path]:
    import pybigtools

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_rank = {name: i for i, name in enumerate(chrom_sizes)}
    written: list[Path] = []
    for track in selected:
        intervals = sorted(accum[track], key=lambda r: (chrom_rank[r[0]], r[1]))
        path = out_dir / f"{track_names[track]}.bw"
        writer = pybigtools.open(str(path), "w")
        writer.write(chrom_sizes, iter(intervals))
        written.append(path)
    return written


__all__ = [
    "RegionPrediction",
    "RegionPredictionConfig",
    "RegionPredictor",
    "Window",
    "collapse_bins",
    "iter_windows",
    "load_model_for_inference",
    "one_hot_context",
    "predict_to_bigwig",
    "read_chrom_sizes",
]
