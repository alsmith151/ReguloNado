"""Array-in prediction: a coordinate-free counterpart to ``inference.RegionPredictor``.

``RegionPredictor`` only accepts genomic coordinates and fetches sequence from a FASTA itself.
The design search needs to score *mutated* one-hot arrays that no longer correspond to any
genomic position, so it needs a predictor that takes one-hot in directly. ``FoldEnsemble`` adds
the second gap: nothing else in the training/inference code ensembles the independently trained
folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

__all__ = ["FoldEnsemble", "FoldSpec", "SequencePredictor"]


@dataclass(slots=True)
class FoldSpec:
    checkpoint_dir: Path
    dataset_dir: Path | None = None
    name: str | None = None


class SequencePredictor:
    """Array-in predictor for one fold: ``(B, 4, context_length) -> (B, n_tracks, n_pred_bins)``."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        dataset_dir: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        from regulonado.inference import _model_track_metadata, load_model_for_inference

        self.model = load_model_for_inference(checkpoint_dir, dataset_dir, device)
        self.model.eval()

        config = self.model.config
        self.context_length = int(config.context_length)
        self.n_pred_bins = int(config.n_pred_bins)
        self.bin_size = int(config.bin_size)
        self.track_names = list(
            config.track_names or [f"track{i}" for i in range(int(config.n_tracks))]
        )
        first_param = next(self.model.parameters())
        self.device = str(first_param.device)
        self.dtype = first_param.dtype
        self.track_metadata = _model_track_metadata(self.model, self.device)
        self.batch_size = batch_size

    def __call__(self, one_hot_batch):
        import numpy as np
        import torch

        if isinstance(one_hot_batch, np.ndarray):
            one_hot_batch = torch.from_numpy(one_hot_batch)

        outputs = []
        with torch.inference_mode():
            for start in range(0, one_hot_batch.shape[0], self.batch_size):
                chunk = one_hot_batch[start : start + self.batch_size].to(
                    device=self.device, dtype=self.dtype
                )
                outputs.append(self.model(chunk, **self.track_metadata))
        return torch.cat(outputs, dim=0)


class FoldEnsemble:
    """Runs the same one-hot batch through several independently trained folds."""

    def __init__(
        self,
        folds: Sequence[FoldSpec],
        device: str | None = None,
        batch_size: int = 1,
        mode: Literal["resident", "sequential"] = "resident",
    ) -> None:
        self._specs = list(folds)
        if not self._specs:
            raise ValueError("FoldEnsemble needs at least one fold")
        if mode not in ("resident", "sequential"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'resident' or 'sequential'")

        self.mode = mode
        self.device = device
        self.batch_size = batch_size

        if mode == "resident":
            self._resident: list[SequencePredictor] | None = [
                self._load(spec) for spec in self._specs
            ]
            geometry_source = self._resident
        else:
            self._resident = None
            geometry_source = [self._load(spec) for spec in self._specs]

        self._assert_consistent(geometry_source)
        first = geometry_source[0]
        self.context_length = first.context_length
        self.n_pred_bins = first.n_pred_bins
        self.bin_size = first.bin_size
        self._track_names = first.track_names

        if mode == "sequential":
            del geometry_source  # transient load; freed so folds don't co-reside

    def _load(self, spec: FoldSpec) -> SequencePredictor:
        return SequencePredictor(
            spec.checkpoint_dir, spec.dataset_dir, self.device, self.batch_size
        )

    def _assert_consistent(self, predictors: list[SequencePredictor]) -> None:
        first_spec, first = self._specs[0], predictors[0]
        first_label = first_spec.name or str(first_spec.checkpoint_dir)
        for spec, predictor in zip(self._specs[1:], predictors[1:]):
            label = spec.name or str(spec.checkpoint_dir)
            if predictor.track_names != first.track_names:
                raise ValueError(
                    f"Fold {label!r} has different track_names than {first_label!r}"
                )
            geometry = (predictor.context_length, predictor.n_pred_bins, predictor.bin_size)
            first_geometry = (first.context_length, first.n_pred_bins, first.bin_size)
            if geometry != first_geometry:
                raise ValueError(
                    f"Fold {label!r} has geometry {geometry} but {first_label!r} has "
                    f"{first_geometry}"
                )

    @property
    def track_names(self) -> list[str]:
        return self._track_names

    def predict(self, one_hot_batch):
        """Predict all folds: returns ``(n_folds, B, n_tracks, n_bins)``."""
        import torch

        outputs = []
        if self.mode == "resident":
            for predictor in self._resident:
                outputs.append(predictor(one_hot_batch))
        else:
            for spec in self._specs:
                predictor = self._load(spec)
                outputs.append(predictor(one_hot_batch))
                del predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.stack(outputs, dim=0)
