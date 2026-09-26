"""Torch dataset joining region-count labels to cached backbone embeddings.

:class:`CachedRegionDataset` is the training-time counterpart of
:mod:`regulonado.embeddings.cache`: it yields ``{"features": [K, D], "labels":
[n_tracks]}`` items for :class:`~regulonado.training.cached.model.RegionCountModel`, joining a
:class:`~regulonado.counts.dataset.RegionCountData` (BAM counts,
:mod:`regulonado.counts`) to an :class:`~regulonado.embeddings.cache.EmbeddingStore`
(cached frozen-backbone features, :mod:`regulonado.embeddings.cache`) via each region's
``region_row`` -- its row index into the *original*, unfiltered region table the
embeddings cache was built from.

Because :meth:`~regulonado.counts.dataset.RegionCountData.take`/``split`` reorder and
subset rows, ``region_row`` has to be attached once, right after the count dataset is
read and *before* any masking/filtering/splitting -- see :func:`attach_region_rows`,
which also validates the embeddings manifest in the same step (both need the
untouched region table). :func:`prepare_region_data` does this and the UEF-ported
stage logic (count masking, ``specific_only``, ``contrast_weighting``) in one call.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from regulonado.counts.dataset import RegionCountData, group_count_rates
from regulonado.embeddings.cache import EmbeddingManifest, EmbeddingStore, validate_manifest
from regulonado.target_specificity import group_index_from_records

logger = logging.getLogger(__name__)

__all__ = [
    "CachedRegionDataset",
    "PreparedRegionData",
    "RegionsDataConfig",
    "attach_region_rows",
    "group_names_and_track_groups",
    "overlaps_any",
    "prepare_region_data",
]


def group_names_and_track_groups(
    tracks: pl.DataFrame, column: str = "group"
) -> tuple[list[str], np.ndarray]:
    """Distinct, sorted group names and each track's group index, from a track table.

    Thin wrapper around :func:`regulonado.target_specificity.group_index_from_records`
    that requires every track to have a group (unlike that function's general
    exclusion support -- a track the count head has no group for cannot be handed a
    ``CountHead`` output slot).

    Raises:
        ValueError: if any track's *column* value is missing/``None``.
    """
    records = tracks.to_dicts()
    group_index, group_names = group_index_from_records(records, key=column)
    missing = [i for i, index in enumerate(group_index) if index < 0]
    if missing:
        names = [tracks["track_name"][i] if "track_name" in tracks.columns else i for i in missing]
        raise ValueError(f"tracks has {len(missing)} track(s) with no {column!r}: {names[:5]}")
    return group_names, group_index


def _grouped_view(
    data: RegionCountData, group_names: list[str], track_groups: np.ndarray
) -> RegionCountData:
    """Pool *data*'s per-track counts into per-group rates, as a same-shape ``RegionCountData``.

    The result's own :meth:`~regulonado.counts.dataset.RegionCountData.gini`/
    ``specific_mask``/``contrast`` then operate at group (cell type) level rather than
    per-track replicate level -- the view UEF's ``TrainingData.grouped_rates`` computes
    Gini/specificity/contrast on for a count dataset.
    """
    rates = group_count_rates(data.counts, track_groups, data.log_size_factors(), len(group_names))
    return RegionCountData.from_arrays(data.regions, rates, list(group_names))


def attach_region_rows(
    data: RegionCountData, embeddings_dir: str | Path
) -> tuple[RegionCountData, EmbeddingManifest]:
    """Validate *data* against the embeddings manifest and attach its ``region_row`` column.

    Must be called before any masking/filtering/splitting of *data*: both the manifest
    check (:func:`~regulonado.embeddings.cache.validate_manifest`) and the row index
    that later joins are keyed on assume *data.regions* is still in the exact row order
    the embeddings cache was built from.

    Raises:
        FileNotFoundError: if *embeddings_dir* has no manifest yet.
        ValueError: if the manifest was built from a different region table.
    """
    manifest = validate_manifest(embeddings_dir, data.regions)
    regions = data.regions.with_row_index("region_row").with_columns(
        pl.col("region_row").cast(pl.Int64)
    )
    return (
        RegionCountData(
            regions=regions, counts=data.counts, track_names=data.track_names, tracks=data.tracks
        ),
        manifest,
    )


@dataclass(slots=True)
class RegionsDataConfig:
    """``data.*`` settings for :func:`prepare_region_data` and the region-training CLI.

    Parameters
    ----------
    path
        :class:`~regulonado.counts.dataset.RegionCountData` run directory.
    embeddings_dir
        :mod:`regulonado.embeddings.cache` directory to read features from.
    target_group
        The group (cell type) name stage 2/3 knobs below are relative to (also used as
        :class:`~regulonado.training.cached.metrics.GroupedCountMetrics`'s ``top_decile_task``).
    specific_only, gini_std_threshold
        Stage 2: keep only cell type-specific regions -- Gini (at group level) above
        ``mean + gini_std_threshold * std``, computed over every region before
        splitting. Ported from UEF ``TrainingData.specific``/``specific_mask``.
    contrast_weighting, contrast_weight_floor, contrast_weight_cap
        Stage 3: weight each *train* region by ``target_group``'s ``|contrast|``
        (clipped to ``[floor, cap]`` in units of the train split's mean ``|contrast|``)
        rather than filtering to a positive set, so the shared-activity background
        stays in the gradient. Ported from UEF ``build_trainer``.
    count_mask_quantile, count_mask_factor
        Per-track artefact-count masking, thresholds computed on the train split only
        (:meth:`~regulonado.counts.dataset.RegionCountData.count_mask_thresholds`).
        ``count_mask_factor <= 0`` disables masking.
    enable_rc_aug
        Randomly serve the reverse-complement pass in train mode, when the cache has
        one (:attr:`~regulonado.embeddings.cache.EmbeddingStore.has_rc`).
    preload
        Load every served region's features into RAM up front
        (:meth:`~regulonado.embeddings.cache.EmbeddingStore.preload`).
    drop_missing_from_cache
        A region absent from the embeddings cache is a clear error by default (a
        cache/region-table mismatch that should never be silently masked); set this to
        drop such regions instead, with a warning.
    exclude_regions
        BED/parquet of held-out sequences (e.g. benchmark candidate peaks): every *train*
        region whose ``start``/``end`` overlaps one is dropped before anything else, so
        the candidates never enter training at any stage. val/test are never touched.
        Ported from UEF ``--exclude_bed``.
    """

    path: str = ""
    embeddings_dir: str = ""
    target_group: str | None = None
    specific_only: bool = False
    gini_std_threshold: float = 1.0
    contrast_weighting: bool = False
    contrast_weight_floor: float = 0.25
    contrast_weight_cap: float = 4.0
    count_mask_quantile: float = 0.999
    count_mask_factor: float = 20.0
    enable_rc_aug: bool = True
    preload: bool = False
    drop_missing_from_cache: bool = False
    exclude_regions: str | None = None


def overlaps_any(regions: pl.DataFrame, intervals: Sequence[tuple[str, int, int]]) -> np.ndarray:
    """Boolean mask: which *regions* rows (``chrom``/``start``/``end``) overlap any interval."""
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in intervals:
        by_chrom.setdefault(str(chrom), []).append((int(start), int(end)))
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, spans in by_chrom.items():
        spans.sort()
        starts: list[int] = []
        ends: list[int] = []
        for start, end in spans:
            if starts and start <= ends[-1]:
                ends[-1] = max(ends[-1], end)
            else:
                starts.append(start)
                ends.append(end)
        merged[chrom] = (np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64))

    mask = np.zeros(regions.height, dtype=bool)
    chroms = regions["chrom"].cast(pl.Utf8).to_numpy()
    region_starts = regions["start"].cast(pl.Int64).to_numpy()
    region_ends = regions["end"].cast(pl.Int64).to_numpy()
    for chrom, (starts, ends) in merged.items():
        rows = np.flatnonzero(chroms == chrom)
        if rows.size == 0:
            continue
        # Last merged interval starting before each region's end; merged intervals are
        # disjoint and sorted, so it is the only one that can reach past the region start.
        index = np.searchsorted(starts, region_ends[rows], side="left") - 1
        hit = index >= 0
        hit[hit] = ends[index[hit]] > region_starts[rows][hit]
        mask[rows] = hit
    return mask


@dataclass(frozen=True)
class PreparedRegionData:
    """:func:`prepare_region_data`'s result: a region-count dataset ready to split and train on."""

    data: RegionCountData
    group_names: list[str]
    track_groups: np.ndarray
    train_sample_weights: np.ndarray | None
    count_mask_thresholds: np.ndarray | None


def prepare_region_data(data: RegionCountData, cfg: RegionsDataConfig) -> PreparedRegionData:
    """Apply the UEF-ported stage pipeline to *data* (which must already carry ``region_row``).

    In order (matching UEF ``build_trainer``): ``exclude_regions`` (train split only) ->
    count masking (thresholds from the train split only) -> ``specific_only`` filtering
    (over every region, before splitting) ->
    ``contrast_weighting`` sample weights (from the train split only, after filtering).
    Splitting into train/val/test itself is left to the caller/:class:`CachedRegionDataset`,
    since a single :class:`PreparedRegionData` is shared to build every split's dataset.

    Raises:
        ValueError: if ``cfg.contrast_weighting`` is set without ``cfg.target_group``,
            or ``cfg.target_group`` does not name one of the track table's groups.
    """
    if "region_row" not in data.regions.columns:
        raise ValueError(
            "data has no 'region_row' column; call attach_region_rows() before "
            "prepare_region_data()"
        )
    group_names, track_groups = group_names_and_track_groups(data.tracks)

    if cfg.exclude_regions:
        from regulonado.normalization import read_regions

        excluded = overlaps_any(data.regions, read_regions(Path(cfg.exclude_regions)))
        excluded &= (data.regions["split"] == "train").to_numpy()
        logger.info(
            "exclude_regions: dropping %d train region(s) overlapping %s",
            int(excluded.sum()),
            cfg.exclude_regions,
        )
        data = data.take(~excluded)

    thresholds = None
    if cfg.count_mask_factor > 0:
        thresholds = data.split("train").count_mask_thresholds(
            cfg.count_mask_quantile, cfg.count_mask_factor
        )
        data = data.mask_counts(thresholds)

    if cfg.specific_only:
        mask = _grouped_view(data, group_names, track_groups).specific_mask(cfg.gini_std_threshold)
        data = data.take(mask)

    sample_weights = None
    if cfg.contrast_weighting:
        if not cfg.target_group:
            raise ValueError("data.contrast_weighting requires data.target_group")
        if cfg.target_group not in group_names:
            raise ValueError(f"data.target_group {cfg.target_group!r} not in groups {group_names}")
        train_grouped = _grouped_view(data.split("train"), group_names, track_groups)
        absolute = np.nan_to_num(np.abs(train_grouped.contrast(cfg.target_group)), nan=0.0)
        scale = float(absolute.mean())
        sample_weights = np.clip(
            absolute / scale if scale > 0 else np.ones_like(absolute),
            cfg.contrast_weight_floor,
            cfg.contrast_weight_cap,
        ).astype(np.float32)

    return PreparedRegionData(
        data=data,
        group_names=group_names,
        track_groups=track_groups,
        train_sample_weights=sample_weights,
        count_mask_thresholds=thresholds,
    )


class CachedRegionDataset(Dataset):
    """One split of a :class:`RegionCountData`, joined to an :class:`EmbeddingStore` by
    ``region_row``.

    Yields ``{"features": [K, D], "labels": [n_tracks]}`` (plus ``"sample_weight"``, a
    scalar, when *sample_weights* is given): both directly consumable by
    :class:`~regulonado.training.cached.model.RegionCountModel` and a plain
    ``torch.utils.data.dataloader.default_collate``/
    ``transformers.default_data_collator``, since every item has the same shapes (the
    manifest's fixed ``K``/``D`` and *region_dataset*'s fixed ``n_tracks``). ``features``
    keeps the cache's ``float16`` dtype; the model casts to float32 itself.
    """

    def __init__(
        self,
        region_dataset: RegionCountData | str | Path,
        embeddings_dir: str | Path | EmbeddingStore,
        split: str | None = None,
        *,
        train: bool = False,
        sample_weights: np.ndarray | None = None,
        enable_rc_aug: bool = True,
        preload: bool = False,
        drop_missing_from_cache: bool = False,
    ) -> None:
        data = (
            region_dataset
            if isinstance(region_dataset, RegionCountData)
            else RegionCountData.read(region_dataset)
        )
        if "region_row" not in data.regions.columns:
            raise ValueError(
                "region_dataset has no 'region_row' column; build it with "
                "attach_region_rows()/prepare_region_data() before constructing "
                "CachedRegionDataset (splitting/filtering must happen on the full, "
                "row-indexed table so region_row stays valid)"
            )
        if split is not None:
            data = data.split(split)

        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)
            if sample_weights.shape != (data.n_regions,):
                raise ValueError(
                    f"sample_weights has shape {sample_weights.shape}; "
                    f"expected ({data.n_regions},) for split {split!r}"
                )

        self.store = (
            embeddings_dir
            if isinstance(embeddings_dir, EmbeddingStore)
            else EmbeddingStore(embeddings_dir)
        )
        region_rows = data.regions["region_row"].cast(pl.Int64).to_numpy()
        available = set(self.store.region_rows)
        missing = [int(row) for row in region_rows if int(row) not in available]
        if missing:
            if not drop_missing_from_cache:
                raise KeyError(
                    f"{len(missing)} of {len(region_rows)} region(s) in split {split!r} are "
                    f"missing from the embeddings cache at {self.store.dir} (first few "
                    f"region_row: {missing[:5]}); pass drop_missing_from_cache=True to drop "
                    "them, or re-run `regulonado embed regions` to fill the gap"
                )
            warnings.warn(
                f"dropping {len(missing)} of {len(region_rows)} region(s) in split {split!r} "
                f"missing from the embeddings cache at {self.store.dir}",
                RuntimeWarning,
                stacklevel=2,
            )
            keep = np.array([int(row) in available for row in region_rows], dtype=bool)
            data = data.take(keep)
            if sample_weights is not None:
                sample_weights = sample_weights[keep]
            region_rows = data.regions["region_row"].cast(pl.Int64).to_numpy()

        self.data = data
        self.train = train
        self.enable_rc_aug = enable_rc_aug
        self.sample_weights = sample_weights
        self._region_rows = region_rows
        if preload:
            self.store.preload(self._region_rows.tolist())

    def __len__(self) -> int:
        return self.data.n_regions

    def __getitem__(self, idx: int) -> dict[str, Any]:
        region_row = int(self._region_rows[idx])
        rc = (
            self.train
            and self.enable_rc_aug
            and self.store.has_rc
            and bool(torch.rand(()) < 0.5)
        )
        features = np.ascontiguousarray(self.store.get(region_row, rc=rc))
        item: dict[str, Any] = {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(self.data.counts[idx].astype(np.float32, copy=False)),
        }
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]))
        return item
