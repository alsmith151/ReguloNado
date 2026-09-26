"""``regulonado.training.cached.data``: cache join, RC augmentation, UEF-ported stage pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from regulonado.counts.dataset import RegionCountData
from regulonado.embeddings.cache import EmbeddingStore, region_table_hash
from regulonado.training.cached import data as regions_data
from regulonado.training.cached.data import (
    CachedRegionDataset,
    RegionsDataConfig,
    attach_region_rows,
    group_names_and_track_groups,
    prepare_region_data,
)


def _toy_data(n_per_split: int = 4) -> RegionCountData:
    """Two groups (hl60: 2 tracks, k562: 1 track), train/val/test splits, all-finite counts."""
    splits = ["train"] * n_per_split + ["val"] * n_per_split + ["test"] * n_per_split
    n = len(splits)
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"] * n,
            "start": [i * 2000 for i in range(n)],
            "end": [i * 2000 + 2000 for i in range(n)],
            "target_start": [i * 2000 + 500 for i in range(n)],
            "target_end": [i * 2000 + 1500 for i in range(n)],
            "split": splits,
        }
    )
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=5.0, size=(n, 3)).astype(np.float32)
    tracks = pl.DataFrame(
        {
            "track_name": ["hl60_a", "hl60_b", "k562_a"],
            "group": ["hl60", "hl60", "k562"],
            "log_size_factor": [0.0, 0.1, -0.1],
        }
    )
    return RegionCountData.from_arrays(regions, counts, ["hl60_a", "hl60_b", "k562_a"], tracks)


def _write_cache(
    out_dir, regions_df: pl.DataFrame, *, k: int, d: int, rc: bool = False, seed: int = 0
):
    n = regions_df.height
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, k, d)).astype(np.float16)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.DataFrame(
        {
            "backbone": ["stub"],
            "checkpoint": ["ckpt"],
            "bin_size": [32],
            "k": [k],
            "d": [d],
            "context": [1024],
            "stride": [512],
            "pool_to": [None],
            "rc": [rc],
            "region_hash": [region_table_hash(regions_df)],
            "n_regions": [n],
        },
        schema={
            "backbone": pl.Utf8,
            "checkpoint": pl.Utf8,
            "bin_size": pl.Int64,
            "k": pl.Int64,
            "d": pl.Int64,
            "context": pl.Int64,
            "stride": pl.Int64,
            "pool_to": pl.Int64,
            "rc": pl.Boolean,
            "region_hash": pl.Utf8,
            "n_regions": pl.Int64,
        },
    )
    manifest.write_parquet(out_dir / "manifest.parquet")

    columns = {
        "region_row": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "features": pa.FixedSizeListArray.from_arrays(
            pa.array(features.reshape(-1), type=pa.float16()), k * d
        ),
    }
    if rc:
        features_rc = rng.standard_normal((n, k, d)).astype(np.float16)
        columns["features_rc"] = pa.FixedSizeListArray.from_arrays(
            pa.array(features_rc.reshape(-1), type=pa.float16()), k * d
        )
    pq.write_table(pa.table(columns), out_dir / "chr1.parquet")
    return features, (features_rc if rc else None)


# --------------------------------------------------------------------------- #
# group_names_and_track_groups
# --------------------------------------------------------------------------- #


def test_group_names_and_track_groups_sorted_and_indexed():
    tracks = pl.DataFrame({"track_name": ["a", "b", "c"], "group": ["k562", "hl60", "hl60"]})
    names, index = group_names_and_track_groups(tracks)
    assert names == ["hl60", "k562"]
    assert index.tolist() == [1, 0, 0]


def test_group_names_and_track_groups_missing_group_raises():
    tracks = pl.DataFrame({"track_name": ["a", "b"], "group": ["hl60", None]})
    with pytest.raises(ValueError, match="no 'group'"):
        group_names_and_track_groups(tracks)


# --------------------------------------------------------------------------- #
# attach_region_rows
# --------------------------------------------------------------------------- #


def test_attach_region_rows_adds_column_and_validates_manifest(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, manifest = attach_region_rows(data, tmp_path / "emb")
    assert attached.regions["region_row"].to_list() == list(range(data.n_regions))
    assert manifest.k == 2
    assert manifest.d == 3
    assert manifest.backbone == "stub"


def test_attach_region_rows_mismatch_raises(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    other = data.take(np.arange(data.n_regions) < data.n_regions - 1)
    with pytest.raises(ValueError, match="different region table"):
        attach_region_rows(other, tmp_path / "emb")


# --------------------------------------------------------------------------- #
# prepare_region_data
# --------------------------------------------------------------------------- #


def _prepared(data, **kwargs):
    attached, _ = attach_region_rows(data, kwargs.pop("embeddings_dir"))
    cfg = RegionsDataConfig(**kwargs)
    return prepare_region_data(attached, cfg)


def test_prepare_region_data_requires_region_row():
    data = _toy_data()
    with pytest.raises(ValueError, match="region_row"):
        prepare_region_data(data, RegionsDataConfig())


def test_prepare_region_data_masks_artefact_counts(tmp_path):
    data = _toy_data()
    # Make one train-split count an obvious single-track outlier.
    counts = data.counts.copy()
    counts[0, 0] = 1_000_000.0
    data = RegionCountData(
        regions=data.regions, counts=counts, track_names=data.track_names, tracks=data.tracks
    )
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    # A low quantile picks the threshold from the bulk of normal counts, not the outlier
    # itself (see RegionCountData.count_mask_thresholds / test_counts_dataset.py).
    prepared = _prepared(
        data, embeddings_dir=tmp_path / "emb", count_mask_factor=20.0, count_mask_quantile=0.0
    )
    assert np.isnan(prepared.data.counts[0, 0])
    assert prepared.count_mask_thresholds is not None
    # Untouched labels stay finite.
    assert np.isfinite(prepared.data.counts[1:, :]).all()


def test_overlaps_any_merges_intervals_and_is_half_open():
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [0, 100, 300, 0],
            "end": [50, 200, 400, 50],
        }
    )
    # [40, 60) and [55, 120) merge into [40, 120); [400, 500) only touches row 2's end.
    mask = regions_data.overlaps_any(
        regions, [("chr1", 40, 60), ("chr1", 55, 120), ("chr1", 400, 500)]
    )
    assert mask.tolist() == [True, True, False, False]


def test_prepare_region_data_exclude_regions_drops_only_overlapping_train_regions(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    # Rows 0 and 1 are train; rows 4 and 5 are val (n_per_split=4, 2 kb apart).
    exclude = tmp_path / "exclude.bed"
    exclude.write_text("chr1\t100\t200\nchr1\t2500\t2600\nchr1\t8100\t10100\n")
    prepared = _prepared(
        data, embeddings_dir=tmp_path / "emb", count_mask_factor=0.0, exclude_regions=str(exclude)
    )
    kept = prepared.data.regions
    assert kept.filter(pl.col("split") == "train").height == 2
    assert 0 not in kept["start"].to_list() and 2000 not in kept["start"].to_list()
    # val/test are the benchmark: never touched.
    assert kept.filter(pl.col("split") == "val").height == 4


def test_prepare_region_data_count_mask_factor_zero_disables_masking(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    prepared = _prepared(data, embeddings_dir=tmp_path / "emb", count_mask_factor=0.0)
    assert prepared.count_mask_thresholds is None
    assert np.isfinite(prepared.data.counts).all()


def test_prepare_region_data_specific_only_filters_low_gini_regions(tmp_path):
    n = 12
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"] * n,
            "start": [i * 2000 for i in range(n)],
            "end": [i * 2000 + 2000 for i in range(n)],
            "target_start": [i * 2000 + 500 for i in range(n)],
            "target_end": [i * 2000 + 1500 for i in range(n)],
            "split": ["train"] * n,
        }
    )
    # Every region flat except one, which is HL-60-only: unambiguously the highest Gini.
    counts = np.full((n, 3), 5.0, dtype=np.float32)
    counts[0] = [500.0, 500.0, 0.0]
    tracks = pl.DataFrame(
        {
            "track_name": ["hl60_a", "hl60_b", "k562_a"],
            "group": ["hl60", "hl60", "k562"],
            "log_size_factor": [0.0, 0.0, 0.0],
        }
    )
    data = RegionCountData.from_arrays(regions, counts, ["hl60_a", "hl60_b", "k562_a"], tracks)
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    prepared = _prepared(
        data,
        embeddings_dir=tmp_path / "emb",
        count_mask_factor=0.0,
        specific_only=True,
        gini_std_threshold=1.0,
    )
    assert prepared.data.n_regions < n
    assert prepared.data.regions["region_row"].to_list() == [0]


def test_prepare_region_data_contrast_weighting_requires_target_group(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    with pytest.raises(ValueError, match="target_group"):
        _prepared(
            data, embeddings_dir=tmp_path / "emb", count_mask_factor=0.0, contrast_weighting=True
        )


def test_prepare_region_data_contrast_weighting_unknown_group_raises(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    with pytest.raises(ValueError, match="not in groups"):
        _prepared(
            data,
            embeddings_dir=tmp_path / "emb",
            count_mask_factor=0.0,
            contrast_weighting=True,
            target_group="nope",
        )


def test_prepare_region_data_contrast_weighting_produces_train_sample_weights(tmp_path):
    data = _toy_data(n_per_split=6)
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    prepared = _prepared(
        data,
        embeddings_dir=tmp_path / "emb",
        count_mask_factor=0.0,
        contrast_weighting=True,
        target_group="hl60",
        contrast_weight_floor=0.25,
        contrast_weight_cap=4.0,
    )
    n_train = (prepared.data.regions["split"] == "train").sum()
    assert prepared.train_sample_weights is not None
    assert prepared.train_sample_weights.shape == (n_train,)
    assert (prepared.train_sample_weights >= 0.25).all()
    assert (prepared.train_sample_weights <= 4.0).all()


# --------------------------------------------------------------------------- #
# CachedRegionDataset
# --------------------------------------------------------------------------- #


def test_cached_region_dataset_requires_region_row(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    with pytest.raises(ValueError, match="region_row"):
        CachedRegionDataset(data, tmp_path / "emb", split="train")


def test_cached_region_dataset_joins_via_region_row(tmp_path):
    data = _toy_data()
    features, _ = _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    train = attached.split("train")
    dataset = CachedRegionDataset(train, tmp_path / "emb", split=None, train=False)
    assert len(dataset) == train.n_regions
    for i in range(len(dataset)):
        item = dataset[i]
        region_row = int(train.regions["region_row"][i])
        assert item["features"].shape == (2, 3)
        expected = torch.from_numpy(features[region_row]).float()
        assert torch.allclose(item["features"].float(), expected)
        assert item["labels"].shape == (3,)
        assert "sample_weight" not in item


def test_cached_region_dataset_splits_internally(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    dataset = CachedRegionDataset(attached, tmp_path / "emb", split="val", train=False)
    assert len(dataset) == (data.regions["split"] == "val").sum()


def test_cached_region_dataset_rc_augmentation_in_train_mode(tmp_path, monkeypatch):
    data = _toy_data()
    features, features_rc = _write_cache(tmp_path / "emb", data.regions, k=2, d=3, rc=True)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    train = attached.split("train")

    dataset = CachedRegionDataset(
        train, tmp_path / "emb", split=None, train=True, enable_rc_aug=True
    )
    monkeypatch.setattr(regions_data.torch, "rand", lambda size: torch.tensor(0.0))
    region_row = int(train.regions["region_row"][0])
    item = dataset[0]
    expected_rc = torch.from_numpy(features_rc[region_row]).float()
    assert torch.allclose(item["features"].float(), expected_rc)

    monkeypatch.setattr(regions_data.torch, "rand", lambda size: torch.tensor(1.0))
    item = dataset[0]
    expected = torch.from_numpy(features[region_row]).float()
    assert torch.allclose(item["features"].float(), expected)


def test_cached_region_dataset_no_rc_when_disabled(tmp_path, monkeypatch):
    data = _toy_data()
    features, features_rc = _write_cache(tmp_path / "emb", data.regions, k=2, d=3, rc=True)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    train = attached.split("train")
    dataset = CachedRegionDataset(
        train, tmp_path / "emb", split=None, train=True, enable_rc_aug=False
    )
    monkeypatch.setattr(regions_data.torch, "rand", lambda size: torch.tensor(0.0))
    region_row = int(train.regions["region_row"][0])
    item = dataset[0]
    expected = torch.from_numpy(features[region_row]).float()
    assert torch.allclose(item["features"].float(), expected)


def test_cached_region_dataset_missing_region_raises_by_default(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    store = EmbeddingStore(tmp_path / "emb")
    # Drop one region_row from the store's index to simulate a partially-embedded cache.
    dropped = next(iter(store._index))
    del store._index[dropped]
    with pytest.raises(KeyError, match="missing from the embeddings cache"):
        CachedRegionDataset(attached, store, split=None, train=False)


def test_cached_region_dataset_drop_missing_from_cache(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    store = EmbeddingStore(tmp_path / "emb")
    dropped = next(iter(store._index))
    del store._index[dropped]
    with pytest.warns(RuntimeWarning, match="dropping"):
        dataset = CachedRegionDataset(
            attached, store, split=None, train=False, drop_missing_from_cache=True
        )
    assert len(dataset) == attached.n_regions - 1
    assert dropped not in dataset.data.regions["region_row"].to_list()


def test_cached_region_dataset_sample_weight_shape_mismatch_raises(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    with pytest.raises(ValueError, match="sample_weights"):
        CachedRegionDataset(
            attached, tmp_path / "emb", split=None, train=True, sample_weights=np.ones(3)
        )


def test_cached_region_dataset_preload(tmp_path):
    data = _toy_data()
    _write_cache(tmp_path / "emb", data.regions, k=2, d=3)
    attached, _ = attach_region_rows(data, tmp_path / "emb")
    dataset = CachedRegionDataset(attached, tmp_path / "emb", split="train", preload=True)
    assert len(dataset.store._preloaded) == len(dataset)
