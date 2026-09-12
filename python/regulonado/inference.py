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

Pure Python; ``torch``/``pyfaidx``/``pybigtools`` are imported lazily so the lightweight helper
(``iter_windows``) imports without a GPU stack. Sequence/window primitives (``Window``,
``one_hot_context``, ``collapse_bins``, ``write_bigwigs``) live in :mod:`regulonado.genomics`.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from regulonado.genomics import (
    Window,
    collapse_bins,
    one_hot_context,
    read_chrom_sizes,
    read_intervals,
    safe_track_filename,
    window_for_interval,
    write_bigwigs,
)

log = logging.getLogger(__name__)


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
    include_input: bool = False  # include the one-hot input in RegionPrediction.input


@dataclass(slots=True)
class RegionPrediction:
    """Prediction values and genomic bin coordinates for one requested interval."""

    chrom: str
    query_start: int
    query_end: int
    input_start: int  # start of model input context (ctx_start; may be < 0 near chrom start)
    input_end: int  # end of model input context (ctx_end; may be > chrom length near chrom end)
    pred_start: int
    pred_end: int
    bin_size: int
    track_names: list[str]
    values: np.ndarray
    input: np.ndarray | None = None  # optional (4, context_length) one-hot input

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
    from regulonado.training.runner import (
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


def model_track_metadata(model: object, device: str) -> dict[str, object]:
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
        _bed_frame = read_intervals(Path(bed_path))
        for chrom, start, end in _bed_frame[["chrom", "start", "end"]].itertuples(
            index=False, name=None
        ):
            chrom = str(chrom)
            start, end = int(start), int(end)
            if chrom not in chrom_sizes:
                raise ValueError(f"BED chromosome {chrom!r} not present in chrom sizes")
            windows.append(
                window_for_interval(
                    chrom,
                    start,
                    end,
                    context_length=context_length,
                    n_pred_bins=n_pred_bins,
                    bin_size=bin_size,
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
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def resolve_tracks(tracks: Sequence[str] | None, track_names: Sequence[str]) -> list[int]:
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
        self.selected_tracks = resolve_tracks(config.tracks, self.track_names)
        first_param = next(self.model.parameters())
        self.model_device = str(first_param.device)
        self.model_dtype = first_param.dtype
        self.track_metadata = model_track_metadata(self.model, self.model_device)
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

        window = window_for_interval(
            chrom,
            start,
            end,
            context_length=self.context_length,
            n_pred_bins=self.n_pred_bins,
            bin_size=self.bin_size,
        )
        seq = one_hot_context(
            self.fasta,
            window,
            self.context_length,
            self.chrom_sizes[chrom],
        )
        x = torch.from_numpy(seq[None]).to(dtype=self.model_dtype, device=self.model_device)
        with torch.no_grad():
            preds = self.model(x, **self.track_metadata)
        values = preds[0].float().cpu().numpy()
        if self.config.inverse_squash:
            from regulonado.dataset.build import inverse_transform_signal

            values = inverse_transform_signal(values, apply_squash=True, apply_scale=False)

        selected = (
            resolve_tracks(tracks, self.track_names)
            if tracks is not None
            else self.selected_tracks
        )
        selected_values = values[selected]
        selected_names = [self.track_names[index] for index in selected]
        pred_end = window.pred_start + selected_values.shape[-1] * self.bin_size
        return RegionPrediction(
            chrom=chrom,
            query_start=start,
            query_end=end,
            input_start=window.ctx_start,
            input_end=window.ctx_end,
            pred_start=window.pred_start,
            pred_end=pred_end,
            bin_size=self.bin_size,
            track_names=selected_names,
            values=selected_values,
            input=seq if self.config.include_input else None,
        )

    def predict_many(
        self,
        coordinates: Sequence[tuple[str, int, int]],
        *,
        tracks: Sequence[str] | None = None,
        batch_size: int = 32,
    ) -> list[RegionPrediction]:
        """Predict multiple coordinate intervals with batched model inference."""
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        selected = (
            resolve_tracks(tracks, self.track_names)
            if tracks is not None
            else self.selected_tracks
        )
        selected_names = [self.track_names[index] for index in selected]
        predictions: list[RegionPrediction] = []

        for coordinate_batch in _chunks(list(coordinates), batch_size):
            windows: list[Window] = []
            sequences: list[np.ndarray] = []
            for chrom, start, end in coordinate_batch:
                if chrom not in self.chrom_sizes:
                    raise ValueError(
                        f"Chromosome {chrom!r} is not present in the configured FASTA/sizes"
                    )
                if start < 0 or end < start:
                    raise ValueError("Coordinates must satisfy 0 <= start <= end")
                window = window_for_interval(
                    chrom,
                    start,
                    end,
                    context_length=self.context_length,
                    n_pred_bins=self.n_pred_bins,
                    bin_size=self.bin_size,
                )
                windows.append(window)
                sequences.append(
                    one_hot_context(
                        self.fasta, window, self.context_length, self.chrom_sizes[chrom]
                    )
                )

            x = torch.from_numpy(np.stack(sequences)).to(
                dtype=self.model_dtype, device=self.model_device
            )
            with torch.no_grad():
                batch_values = self.model(x, **self.track_metadata).float().cpu().numpy()
            for (chrom, start, end), window, seq, values in zip(
                coordinate_batch, windows, sequences, batch_values, strict=True
            ):
                if self.config.inverse_squash:
                    from regulonado.dataset.build import inverse_transform_signal

                    values = inverse_transform_signal(values, apply_squash=True, apply_scale=False)
                selected_values = values[selected]
                predictions.append(
                    RegionPrediction(
                        chrom=chrom,
                        query_start=start,
                        query_end=end,
                        input_start=window.ctx_start,
                        input_end=window.ctx_end,
                        pred_start=window.pred_start,
                        pred_end=window.pred_start + selected_values.shape[-1] * self.bin_size,
                        bin_size=self.bin_size,
                        track_names=selected_names,
                        values=selected_values,
                        input=seq if self.config.include_input else None,
                    )
                )
        return predictions


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
    first_param = next(model.parameters())
    model_device = str(first_param.device)
    model_dtype = first_param.dtype
    track_metadata = model_track_metadata(model, model_device)
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

    selected = resolve_tracks(tracks, track_names)
    n_batches = (len(windows) + batch_size - 1) // batch_size
    log.info(
        "%d windows across %d chromosome(s), %d selected track(s), %d batch(es)",
        len(windows),
        len({w.chrom for w in windows}),
        len(selected),
        n_batches,
    )

    accum: dict[int, list[tuple[str, int, int, float]]] = {t: [] for t in selected}
    spool_root: Path | None = None
    spool_handles: dict[int, object] = {}
    chrom_rank = {name: index for index, name in enumerate(chrom_sizes)}
    if whole_genome:
        spool_root = Path(tempfile.mkdtemp(prefix="regulonado-predict-"))

    def close_spool_handles() -> None:
        for handle in spool_handles.values():
            handle.close()
        spool_handles.clear()

    # Bound to a local rather than left as a bare conditional import: the call site below
    # sits under a separate `if inverse_squash` guard, so the name being defined depended
    # on two guards staying in sync. Importing lazily still avoids pulling in
    # regulonado.dataset (and its Arrow/datasets deps) when no inverse transform is needed.
    inverse_signal = None
    if inverse_squash:
        from regulonado.dataset.build import inverse_transform_signal as inverse_signal

    from tqdm import tqdm

    current_chrom = ""
    bar = tqdm(total=len(windows), unit="win", desc="predicting")
    for batch in _chunks(windows, batch_size):
        seqs = np.stack(
            [one_hot_context(fasta, w, context_length, chrom_sizes[w.chrom]) for w in batch]
        )
        x = torch.from_numpy(seqs).to(dtype=model_dtype, device=model_device)
        with torch.no_grad():
            preds = model(x, **track_metadata)
        preds = preds.float().cpu().numpy()  # (B, n_tracks, n_pred_bins)
        if inverse_signal is not None:
            preds = inverse_signal(preds, apply_squash=True, apply_scale=False)
        for batch_index, window in enumerate(batch):
            if window.chrom != current_chrom:
                if spool_root is not None:
                    close_spool_handles()
                current_chrom = window.chrom
                bar.set_postfix(chrom=current_chrom)
            chrom_length = chrom_sizes[window.chrom]
            for track in selected:
                intervals = collapse_bins(
                    preds[batch_index, track],
                    window.chrom,
                    window.pred_start,
                    bin_size,
                    rtol,
                    chrom_length,
                )
                if spool_root is None:
                    accum[track].extend(intervals)
                    continue
                if track not in spool_handles:
                    track_dir = spool_root / str(track)
                    track_dir.mkdir(parents=True, exist_ok=True)
                    spool_handles[track] = (track_dir / f"{chrom_rank[window.chrom]:06d}.tsv").open(
                        "a", encoding="utf-8"
                    )
                handle = spool_handles[track]
                for _, start, end, value in intervals:
                    handle.write(f"{start}\t{end}\t{value!r}\n")
        bar.update(len(batch))
    bar.close()
    if spool_root is not None:
        close_spool_handles()

    log.info("Writing %d BigWig(s) to %s", len(selected), out_dir)
    if spool_root is None:
        written = write_bigwigs(out_dir, track_names, selected, accum, chrom_sizes)
    else:
        written = _write_spooled_bigwigs(
            out_dir, track_names, selected, spool_root, chrom_sizes
        )
        import shutil

        shutil.rmtree(spool_root)
    log.info("Done — wrote %s", ", ".join(str(p.name) for p in written))
    return written


def _write_spooled_bigwigs(
    out_dir: str | Path,
    track_names: Sequence[str],
    selected: Sequence[int],
    spool_root: Path,
    chrom_sizes: dict[str, int],
) -> list[Path]:
    import pybigtools

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    output_names: set[str] = set()
    chrom_names = list(chrom_sizes)
    for track in selected:
        track_name = safe_track_filename(track_names[track])
        if track_name in output_names:
            raise ValueError(f"Track names produce duplicate output filename: {track_name!r}")
        output_names.add(track_name)
        track_dir = spool_root / str(track)

        def intervals() -> Iterator[tuple[str, int, int, float]]:
            for chrom_index, chrom in enumerate(chrom_names):
                path = track_dir / f"{chrom_index:06d}.tsv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        start, end, value = line.rstrip("\n").split("\t")
                        yield chrom, int(start), int(end), float(value)

        path = out_dir / f"{track_name}.bw"
        writer = pybigtools.open(str(path), "w")
        try:
            writer.write(chrom_sizes, intervals())
        finally:
            close = getattr(writer, "close", None)
            if close is not None:
                close()
        written.append(path)
    return written


__all__ = [
    "RegionPrediction",
    "RegionPredictionConfig",
    "RegionPredictor",
    "iter_windows",
    "load_model_for_inference",
    "predict_to_bigwig",
]
