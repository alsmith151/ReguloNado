"""Parquet-backed count dataset: the handoff from BAM counting to region-head training.

Ported from ``unique_enhancer_finding.modelling.dataset.TrainingData``, with
counts (raw event counts, one task per track/replicate) as the only target
kind. One run is one directory, all parquet, region-major throughout::

    <run>/regions.parquet   one row per region: chrom, start, end,
                            target_start, target_end, split, extras
    <run>/counts.parquet    one row per region, one float32 column per
                            track, row-aligned to regions.parquet
    <run>/tracks.parquet    one row per track: track_name, assay_class,
                            group, count_unit, n_events, count_scale_low,
                            count_scale_high, log_size_factor

In memory the three files are one :class:`RegionCountData`.

Coordinates are 0-based half-open, as everywhere in ReguloNado.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

__all__ = [
    "COUNTS_FILENAME",
    "REGIONS_FILENAME",
    "TRACKS_FILENAME",
    "RegionCountData",
    "group_count_rates",
]

REGIONS_FILENAME = "regions.parquet"
COUNTS_FILENAME = "counts.parquet"
TRACKS_FILENAME = "tracks.parquet"

_REQUIRED_REGION_COLUMNS = ("chrom", "start", "end")


def group_count_rates(
    counts: np.ndarray, track_groups: np.ndarray, log_size_factors: np.ndarray, n_groups: int
) -> np.ndarray:
    """Pool per-track counts into per-group normalised rates, ``[n_regions, n_groups]``.

    ``rate_g = sum_{t in g} y_t / sum_{t in g} exp(o_t)``: the maximum-likelihood
    shared rate of a group's replicates under a Poisson model with per-track
    size factors ``exp(o_t)``.

    ``NaN`` counts (masked, see :meth:`RegionCountData.mask_counts`) are left
    out of both sums, so a group's rate comes from its remaining replicates;
    it is ``NaN`` only where every replicate of the group is masked.
    """
    counts = np.asarray(counts, dtype=np.float64)
    groups = np.asarray(track_groups, dtype=np.int64)
    size = np.exp(np.asarray(log_size_factors, dtype=np.float64))
    observed = np.isfinite(counts)
    pooled = np.zeros((counts.shape[0], n_groups))
    np.add.at(pooled.T, groups, np.where(observed, counts, 0.0).T)
    denominator = np.zeros((counts.shape[0], n_groups))
    np.add.at(denominator.T, groups, (observed * size).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(denominator > 0, pooled / denominator, np.nan)
    return rates.astype(np.float32)


@dataclass(frozen=True)
class RegionCountData:
    """One run's region table, count matrix and track table, kept row-aligned.

    Attributes:
        regions: one row per region, with at least ``chrom``/``start``/``end``
            and normally ``target_start``/``target_end``, ``split`` and any
            provenance columns.
        counts: ``[n_regions, n_tracks]`` float32 raw event counts, row-aligned
            to *regions*. float32 because torch accelerators reject float64.
        track_names: the ``n_tracks`` count column names, in column order.
        tracks: per-track metadata, one row per track in *track_names* order.
    """

    regions: pl.DataFrame
    counts: np.ndarray
    track_names: list[str]
    tracks: pl.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in _REQUIRED_REGION_COLUMNS if c not in self.regions.columns]
        if missing:
            raise ValueError(f"regions is missing columns: {missing}")

        matrix = np.ascontiguousarray(self.counts, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"counts must be 2-D [n_regions, n_tracks]; got shape {matrix.shape}")
        if matrix.shape[0] != self.regions.height:
            raise ValueError(
                f"counts has {matrix.shape[0]} rows but regions has {self.regions.height} — "
                "they must be row-aligned"
            )
        if matrix.shape[1] != len(self.track_names):
            raise ValueError(
                f"counts has {matrix.shape[1]} columns but "
                f"{len(self.track_names)} track names were given"
            )
        if len(set(self.track_names)) != len(self.track_names):
            raise ValueError("track_names must be unique — they become parquet column names")
        if self.tracks.height != len(self.track_names):
            raise ValueError(
                f"tracks has {self.tracks.height} rows but there are {len(self.track_names)} tracks"
            )

        object.__setattr__(self, "counts", matrix)
        object.__setattr__(self, "track_names", [str(name) for name in self.track_names])

    @classmethod
    def from_arrays(
        cls,
        regions: pl.DataFrame,
        counts: np.ndarray,
        track_names: list[str],
        tracks: pl.DataFrame | None = None,
    ) -> RegionCountData:
        """Build from parts, writing a minimal ``track_name`` table if *tracks* is None."""
        if tracks is None:
            tracks = pl.DataFrame({"track_name": list(track_names)})
        return cls(regions=regions, counts=counts, track_names=list(track_names), tracks=tracks)

    @classmethod
    def read(cls, run_dir: str | Path) -> RegionCountData:
        """Read one run directory written by :meth:`write`.

        Raises:
            FileNotFoundError: if the directory is not a count-dataset run.
        """
        directory = Path(run_dir)
        regions_path = directory / REGIONS_FILENAME
        counts_path = directory / COUNTS_FILENAME
        if not regions_path.exists() or not counts_path.exists():
            raise FileNotFoundError(
                f"{directory} is not a count-dataset run "
                f"(expected {REGIONS_FILENAME} and {COUNTS_FILENAME})"
            )

        counts_frame = pl.read_parquet(counts_path)
        tracks_path = directory / TRACKS_FILENAME
        return cls.from_arrays(
            regions=pl.read_parquet(regions_path),
            counts=counts_frame.to_numpy(),
            track_names=list(counts_frame.columns),
            tracks=pl.read_parquet(tracks_path) if tracks_path.exists() else None,
        )

    def write(self, run_dir: str | Path) -> dict[str, Path]:
        """Write the three-file layout into *run_dir* (created if absent).

        Returns:
            A mapping of ``"regions"``/``"counts"``/``"tracks"`` to written paths.
        """
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "regions": directory / REGIONS_FILENAME,
            "counts": directory / COUNTS_FILENAME,
            "tracks": directory / TRACKS_FILENAME,
        }
        frames = {
            "regions": self.regions,
            "counts": pl.DataFrame(
                {name: self.counts[:, index] for index, name in enumerate(self.track_names)}
            ),
            "tracks": self.tracks,
        }
        # Write-then-rename: a reader started while a resumed/re-run job is
        # still writing never sees a truncated parquet file.
        for name, frame in frames.items():
            partial = paths[name].with_name(f".{paths[name].name}.partial")
            frame.write_parquet(partial)
            os.replace(partial, paths[name])
        return paths

    @property
    def n_regions(self) -> int:
        return self.regions.height

    @property
    def n_tracks(self) -> int:
        return len(self.track_names)

    def track_index(self, name: str) -> int:
        """Column index of track *name*.

        Raises:
            KeyError: if *name* is not a track.
        """
        try:
            return self.track_names.index(name)
        except ValueError:
            raise KeyError(f"track {name!r} not found; known: {self.track_names}") from None

    def log_size_factors(self) -> np.ndarray:
        """Per-track log size factors (the count model's fixed offsets), ``[n_tracks]``.

        Raises:
            KeyError: if ``tracks`` has no ``log_size_factor`` column.
            ValueError: if any track has no finite ``log_size_factor``.
        """
        if "log_size_factor" not in self.tracks.columns:
            raise KeyError("tracks has no 'log_size_factor' column; not an anchored count dataset")
        values = self.tracks["log_size_factor"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        bad = [self.track_names[i] for i in np.flatnonzero(~np.isfinite(values))]
        if bad:
            raise ValueError(f"tracks without a finite log_size_factor: {bad}")
        return values

    def take(self, mask: np.ndarray) -> RegionCountData:
        """Keep the regions where boolean *mask* is true, preserving row alignment."""
        return self._take(np.asarray(mask, dtype=bool))

    def split(self, name: str) -> RegionCountData:
        """Keep only regions whose ``split`` column equals *name*, preserving row alignment.

        Raises:
            KeyError: if there is no ``split`` column.
        """
        if "split" not in self.regions.columns:
            raise KeyError("regions has no 'split' column")
        return self._take((self.regions["split"] == name).to_numpy())

    def count_mask_thresholds(self, quantile: float = 0.999, factor: float = 20.0) -> np.ndarray:
        """Per-track count above which a label is treated as an artefact, ``[n_tracks]``.

        ``factor`` times the track's *quantile* count over these regions
        (call it on the train split, so evaluation never informs it), floored
        at ``factor`` so a sparse track's threshold cannot collapse to zero.
        """
        levels = np.nanquantile(self.counts.astype(np.float64), quantile, axis=0)
        return factor * np.maximum(levels, 1.0)

    def mask_counts(self, thresholds: np.ndarray) -> RegionCountData:
        """Set counts above their track's threshold to ``NaN`` (skipped by loss and metrics)."""
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if thresholds.shape != (self.n_tracks,):
            raise ValueError(
                f"need one threshold per track ({self.n_tracks}); got {thresholds.shape}"
            )
        counts = self.counts.copy()
        counts[counts > thresholds] = np.nan
        return RegionCountData(
            regions=self.regions, counts=counts, track_names=self.track_names, tracks=self.tracks
        )

    def gini(self) -> np.ndarray:
        """Per-region Gini coefficient of the count vector across tracks, ``[n_regions]``.

        0 for a region equally active everywhere, towards 1 for one active in
        a single track. Negative counts are clipped to 0.
        """
        values = np.sort(self.counts.clip(0, None) + 1e-7, axis=1)
        n = values.shape[1]
        rank_weights = 2 * np.arange(1, n + 1) - n - 1
        return (values @ rank_weights) / (n * values.sum(axis=1))

    def specific_mask(self, gini_std_threshold: float = 1.0) -> np.ndarray:
        """Region mask keeping cell type-specific regions: Gini above ``mean + std * threshold``.

        The threshold is computed over every region passed in, so call this
        before :meth:`split`. NaN-aware: a region whose every replicate of
        some group is masked has a NaN rate there, and one NaN must not void
        the threshold.
        """
        gini = self.gini()
        return gini > np.nanmean(gini) + gini_std_threshold * np.nanstd(gini)

    def contrast(self, track: str) -> np.ndarray:
        """Per-region ``log1p`` signal for *track* minus the region's mean, ``[n_regions]``."""
        logged = np.log1p(self.counts.clip(0, None))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (masked counts)
            return logged[:, self.track_index(track)] - np.nanmean(logged, axis=1)

    def _take(self, mask: np.ndarray) -> RegionCountData:
        return RegionCountData(
            regions=self.regions.filter(pl.Series(mask)),
            counts=self.counts[mask],
            track_names=self.track_names,
            tracks=self.tracks,
        )
