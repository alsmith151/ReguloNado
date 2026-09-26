"""Frozen-backbone embedding cache: tiling, per-region extraction and parquet storage.

Ported design (see the plan's embedding-cache section): the trunk runs frozen, once, at
long context; the cache stores, per region, only the ``K`` backbone bins covering its
scored 1 kb target, so a small region-count head can train from the cache without ever
touching the (expensive) backbone again.

Tiling is derived purely from the adapter's geometry (:class:`BaseBackboneAdapter`), with
no backbone-specific cases:

- **Fixed-input backbones** (Borzoi, Enformer) run at their one supported input length and
  keep the whole of the backbone's own centre-cropped output as the "kept span". Windows
  tile the chromosome back-to-back by that kept span's width.
- **Flexible backbones** (AlphaGenome) run at ``--context`` bp and additionally keep only
  the central ``--stride`` bp of the (uncropped) output, discarding the rest as context-only
  margin. Windows tile by ``stride``.

Kept spans tile the chromosome back-to-back and contiguously, so every kept bin -- whichever
tile it came from -- is a valid central bin with full context. A region's ``K`` raw bins are
therefore found by *stitching*, not by adding extra windows: its raw-bin range is split into
one segment per tile it touches (almost always one, occasionally two adjacent tiles for a
region straddling a tile boundary; see :func:`_region_segments`), each tile is run at most
once regardless of how many regions' segments land in it, and a straddling region's bins are
reassembled by concatenating its segments in order. At HL-60 density (~0.4 regions/kb) most
tile boundaries have a straddler, so this keeps the forward-pass count equal to the tile
count -- extra one-off windows per straddler would otherwise add 35-100% more passes.
``--pool-to`` groups adjacent raw bins by simple averaging *after* stitching, so a pool group
that itself straddles a tile boundary still pools correctly.

Reverse-complementing: the whole window is complemented and reversed before the forward
pass; since flipping a bin-aligned array end-to-end also reverses the order of the bins
(without touching their internal alignment), flipping the RC pass's output back along the
bin axis restores forward genomic order, ready for the same window-relative slicing logic
as the forward pass. See :func:`_reverse_complement_onehot` and ``_embed_chrom``.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from regulonado.model.adapters import BaseBackboneAdapter
from regulonado.sequence import Genome, fetch_window

__all__ = [
    "MANIFEST_FILENAME",
    "EmbeddingManifest",
    "EmbeddingStore",
    "embed_regions",
    "read_manifest",
    "region_table_hash",
    "validate_manifest",
]

MANIFEST_FILENAME = "manifest.parquet"

_REQUIRED_REGION_COLUMNS = ("chrom", "target_start", "target_end")


def region_table_hash(regions_df: pl.DataFrame) -> str:
    """SHA-256 over ``chrom``/``target_start``/``target_end``, order-sensitive.

    Used to detect a region table that has changed (rows added/removed/reordered) between
    two invocations sharing an embeddings directory, so a stale cache is never silently
    read against a different region set.
    """
    missing = [c for c in _REQUIRED_REGION_COLUMNS if c not in regions_df.columns]
    if missing:
        raise ValueError(f"regions is missing columns needed for region_table_hash: {missing}")
    hasher = hashlib.sha256()
    hasher.update("\n".join(regions_df["chrom"].cast(pl.Utf8).to_list()).encode("utf-8"))
    hasher.update(np.ascontiguousarray(regions_df["target_start"].cast(pl.Int64).to_numpy()).tobytes())
    hasher.update(np.ascontiguousarray(regions_df["target_end"].cast(pl.Int64).to_numpy()).tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class EmbeddingManifest:
    """One embeddings directory's settings, written once and checked on every rerun.

    Attributes
    ----------
    backbone, checkpoint
        Backbone family name and pretrained checkpoint identifier, for provenance.
    bin_size
        Effective bin width in bp, *after* ``--pool-to`` (equal to the backbone's own
        ``output_bin_size`` when no pooling was requested).
    k, d
        Bins per region and feature dimension per bin -- the cached array's shape.
    context, stride
        The tiling parameters used (see module docstring); ``context`` is the backbone's
        fixed input length for fixed-input backbones.
    pool_to
        The requested pooled bin width, or ``None`` when no pooling was applied.
    rc
        Whether a ``features_rc`` column was cached alongside ``features``.
    region_hash, n_regions
        :func:`region_table_hash` and row count of the region table this cache was built
        from -- checked by :func:`validate_manifest` on every rerun.
    """

    backbone: str
    checkpoint: str
    bin_size: int
    k: int
    d: int
    context: int
    stride: int
    pool_to: int | None
    rc: bool
    region_hash: str
    n_regions: int


def _manifest_frame(manifest: EmbeddingManifest) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "backbone": [manifest.backbone],
            "checkpoint": [manifest.checkpoint],
            "bin_size": [manifest.bin_size],
            "k": [manifest.k],
            "d": [manifest.d],
            "context": [manifest.context],
            "stride": [manifest.stride],
            "pool_to": [manifest.pool_to],
            "rc": [manifest.rc],
            "region_hash": [manifest.region_hash],
            "n_regions": [manifest.n_regions],
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


def read_manifest(embeddings_dir: str | Path) -> EmbeddingManifest:
    """Read ``<embeddings_dir>/manifest.parquet``.

    Raises:
        FileNotFoundError: if no manifest has been written there yet.
    """
    path = Path(embeddings_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No {MANIFEST_FILENAME} found in {embeddings_dir}")
    row = pl.read_parquet(path).row(0, named=True)
    pool_to = row["pool_to"]
    return EmbeddingManifest(
        backbone=row["backbone"],
        checkpoint=row["checkpoint"],
        bin_size=int(row["bin_size"]),
        k=int(row["k"]),
        d=int(row["d"]),
        context=int(row["context"]),
        stride=int(row["stride"]),
        pool_to=None if pool_to is None else int(pool_to),
        rc=bool(row["rc"]),
        region_hash=row["region_hash"],
        n_regions=int(row["n_regions"]),
    )


def validate_manifest(embeddings_dir: str | Path, regions_df: pl.DataFrame) -> EmbeddingManifest:
    """Read the manifest and check it was built from *regions_df*.

    Raises:
        FileNotFoundError: if no manifest exists yet.
        ValueError: if the manifest's region hash/count does not match *regions_df* --
            this is the guard that stops a rerun with a different region table (or
            different settings) from silently mixing chromosome files written under
            different assumptions.
    """
    manifest = read_manifest(embeddings_dir)
    expected_hash = region_table_hash(regions_df)
    if manifest.region_hash != expected_hash or manifest.n_regions != regions_df.height:
        raise ValueError(
            f"manifest at {embeddings_dir} was built from a different region table "
            f"({manifest.n_regions} regions, hash {manifest.region_hash[:12]}...) than "
            f"the one given now ({regions_df.height} regions, hash {expected_hash[:12]}...)"
        )
    return manifest


def _write_manifest(embeddings_dir: Path, manifest: EmbeddingManifest) -> None:
    path = embeddings_dir / MANIFEST_FILENAME
    partial = path.with_name(f".{path.name}.partial")
    _manifest_frame(manifest).write_parquet(partial)
    os.replace(partial, path)


def _window_geometry(
    adapter: BaseBackboneAdapter, context: int, stride: int
) -> tuple[int, int, int]:
    """Derive ``(input_length, keep_offset_bp, keep_bp)`` for tiling, from adapter geometry.

    ``keep_offset_bp``/``keep_bp`` describe the "kept span" (see module docstring): the
    part of one window's output that is actually stored, relative to that window's own
    input start. Windows tile the chromosome back-to-back by ``keep_bp``.
    """
    bin_size = adapter.output_bin_size
    if adapter.fixed_input_length is not None:
        input_length = adapter.fixed_input_length
        offset_bp, n_bins = adapter.output_span(input_length)
        return input_length, offset_bp, n_bins * bin_size

    if context % adapter.input_multiple != 0:
        raise ValueError(f"--context {context} is not a multiple of {adapter.input_multiple}")
    if stride % bin_size != 0:
        raise ValueError(f"--stride {stride} is not a multiple of the backbone bin size {bin_size}")
    input_length = context
    offset_bp, n_bins = adapter.output_span(input_length)
    full_kept_bp = n_bins * bin_size
    if stride > full_kept_bp:
        raise ValueError(
            f"--stride {stride} exceeds the backbone's output span for --context {context} "
            f"({full_kept_bp} bp)"
        )
    extra_offset_bins = (n_bins - stride // bin_size) // 2
    keep_offset_bp = offset_bp + extra_offset_bins * bin_size
    return input_length, keep_offset_bp, stride


def _reverse_complement_onehot(x: np.ndarray) -> np.ndarray:
    """Reverse-complement a one-hot window, rows ``A C G T`` (see ``sequence.py``)."""
    return np.ascontiguousarray(x[[3, 2, 1, 0], :][:, ::-1])


def _target_width(regions_df: pl.DataFrame) -> int:
    widths = (regions_df["target_end"] - regions_df["target_start"]).unique()
    if widths.len() != 1:
        raise ValueError(
            "embed_regions requires a constant target width across all regions "
            f"(K bins must be the same for every region); got widths {sorted(widths.to_list())}"
        )
    return int(widths[0])


def _region_segments(
    chrom_regions: pl.DataFrame, *, bin_size: int, pool_factor: int, k: int, keep_bp: int
) -> tuple[list[tuple[int, list[tuple[int, int, int]]]], list[int]]:
    """Split each region's raw-bin range into per-tile segments, and list the tiles needed.

    Tiles are numbered ``0, 1, 2, ...`` by their (bin-aligned) kept-span start
    ``tile_index * keep_bp``; tile ``t``'s kept span holds raw bins
    ``[t * n_bins_per_tile, (t + 1) * n_bins_per_tile)``. A region's raw-bin range almost
    always falls inside one tile; one that straddles a boundary is split into one segment
    per tile it touches (in practice at most two, since ``K`` bins are tiny next to one
    tile's span), so it can be reassembled by concatenating those tiles' bins in order
    without ever running an extra window (see module docstring).

    Returns
    -------
    tuple[list, list[int]]
        ``(segments_by_region, tile_indices)``: ``segments_by_region`` is
        ``[(region_row, [(tile_index, local_start_bin, n_bins), ...]), ...]``, one entry
        per region, in the order it appeared in *chrom_regions*; ``tile_indices`` is the
        sorted, deduplicated list of every tile index any region's segments touch -- the
        complete set of windows that actually need a forward pass.
    """
    n_bins_per_tile = keep_bp // bin_size
    segments_by_region: list[tuple[int, list[tuple[int, int, int]]]] = []
    tile_indices: set[int] = set()
    for region_row, target_start in zip(
        chrom_regions["region_row"].to_list(), chrom_regions["target_start"].to_list()
    ):
        eff_bin_size = bin_size * pool_factor
        raw_first = (target_start // eff_bin_size) * pool_factor
        raw_count = k * pool_factor

        segments: list[tuple[int, int, int]] = []
        pos_bin = raw_first
        remaining = raw_count
        while remaining > 0:
            tile_index, local_start = divmod(pos_bin, n_bins_per_tile)
            take = min(n_bins_per_tile - local_start, remaining)
            segments.append((tile_index, local_start, take))
            tile_indices.add(tile_index)
            pos_bin += take
            remaining -= take
        segments_by_region.append((region_row, segments))
    return segments_by_region, sorted(tile_indices)


def _decode_fixed_size_list(table: pa.Table, name: str, k: int, d: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    flat = column.flatten()
    values = flat.to_numpy(zero_copy_only=False).copy()
    return values.reshape(table.num_rows, k, d)


def _chrom_schema(k: int, d: int, rc: bool) -> pa.Schema:
    fields = [
        pa.field("region_row", pa.int64()),
        pa.field("features", pa.list_(pa.float16(), k * d)),
    ]
    if rc:
        fields.append(pa.field("features_rc", pa.list_(pa.float16(), k * d)))
    return pa.schema(fields)


def _chrom_batch(
    schema: pa.Schema,
    region_rows: Sequence[int],
    features: Sequence[np.ndarray],
    features_rc: Sequence[np.ndarray] | None,
    k: int,
    d: int,
) -> pa.RecordBatch:
    def _column(arrays: Sequence[np.ndarray]) -> pa.Array:
        flat = np.stack(arrays).astype(np.float16).reshape(-1)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float16()), k * d)

    rows = pa.array(np.asarray(region_rows, dtype=np.int64), type=pa.int64())
    columns = [rows, _column(features)]
    if features_rc is not None:
        columns.append(_column(features_rc))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _embed_chrom(
    *,
    chrom: str,
    chrom_regions: pl.DataFrame,
    genome: Genome,
    adapter: BaseBackboneAdapter,
    input_length: int,
    keep_offset_bp: int,
    keep_bp: int,
    bin_size: int,
    pool_factor: int,
    k: int,
    d: int,
    rc: bool,
    batch_size: int,
    device: str,
    row_group_size: int,
    chrom_path: Path,
) -> None:
    """Embed one chromosome's regions, streaming tiles and row groups to bound memory.

    Tiles run in genomic order and are dropped as soon as no pending region needs them, and
    finished regions are written out in float16 row groups as they fill, so peak memory is
    a few tiles plus one row group -- not the whole chromosome.
    """
    segments_by_region, tile_indices = _region_segments(
        chrom_regions, bin_size=bin_size, pool_factor=pool_factor, k=k, keep_bp=keep_bp
    )
    # Regions complete in order of their last tile; sorting by it lets a single pass over
    # the tiles emit each region as soon as every tile it needs has been run.
    pending = sorted(segments_by_region, key=lambda item: (item[1][-1][0], item[1][0][0]))
    offset_bp, _n_bins_bb = adapter.output_span(input_length)
    local_bin_offset = (keep_offset_bp - offset_bp) // bin_size
    n_bins_kept = keep_bp // bin_size
    kept_slice = slice(local_bin_offset, local_bin_offset + n_bins_kept)
    row_group_size = max(row_group_size, 1)

    schema = _chrom_schema(k, d, rc)
    partial = chrom_path.with_name(f".{chrom_path.name}.partial")
    writer = pq.ParquetWriter(partial, schema, compression="none")

    # Each tile is run at most once, however many regions' segments touch it -- straddling
    # regions are handled by stitching tiles' bins together, not by extra windows.
    tile_features: dict[int, np.ndarray] = {}
    tile_features_rc: dict[int, np.ndarray] = {}
    rows_out: list[int] = []
    features_out: list[np.ndarray] = []
    features_rc_out: list[np.ndarray] = []
    next_region = 0

    def _stitch(tiles: dict[int, np.ndarray], segments: list[tuple[int, int, int]]) -> np.ndarray:
        # [D, raw_count], stitched across tiles when the region straddled a boundary.
        raw = np.concatenate(
            [tiles[t][:, start : start + count] for t, start, count in segments], axis=1
        )
        return raw.reshape(d, k, pool_factor).mean(axis=-1).T.astype(np.float16)  # [K, D]

    def _flush() -> None:
        if rows_out:
            writer.write_batch(
                _chrom_batch(schema, rows_out, features_out, features_rc_out if rc else None, k, d),
                row_group_size=row_group_size,
            )
            rows_out.clear()
            features_out.clear()
            features_rc_out.clear()

    try:
        with torch.inference_mode():
            step = max(batch_size, 1)
            for batch_index in range(0, len(tile_indices), step):
                batch_tiles = tile_indices[batch_index : batch_index + step]
                inputs = [
                    fetch_window(
                        genome,
                        chrom,
                        tile_index * keep_bp - keep_offset_bp,
                        tile_index * keep_bp - keep_offset_bp + input_length,
                    )
                    for tile_index in batch_tiles
                ]
                batch_tensor = torch.from_numpy(np.stack(inputs)).to(device)
                features_full = adapter.forward_features(batch_tensor)
                features_full_np = features_full.detach().to(torch.float32).cpu().numpy()

                features_rc_full_np = None
                if rc:
                    rc_inputs = np.stack([_reverse_complement_onehot(x) for x in inputs])
                    rc_tensor = torch.from_numpy(rc_inputs).to(device)
                    features_rc_full = adapter.forward_features(rc_tensor)
                    # Flip back along the bin axis (per window, before any stitching): see
                    # module docstring.
                    features_rc_full_np = features_rc_full.detach().to(torch.float32).cpu().numpy()
                    features_rc_full_np = features_rc_full_np[:, :, ::-1]

                for local_index, tile_index in enumerate(batch_tiles):
                    # Copy, so the kept slice does not pin the whole batch output in memory.
                    tile_features[tile_index] = features_full_np[local_index][:, kept_slice].copy()
                    if features_rc_full_np is not None:
                        tile_features_rc[tile_index] = (
                            features_rc_full_np[local_index][:, kept_slice].copy()
                        )
                last_run = batch_tiles[-1]

                while next_region < len(pending) and pending[next_region][1][-1][0] <= last_run:
                    region_row, segments = pending[next_region]
                    rows_out.append(int(region_row))
                    features_out.append(_stitch(tile_features, segments))
                    if rc:
                        features_rc_out.append(_stitch(tile_features_rc, segments))
                    next_region += 1
                    if len(rows_out) >= row_group_size:
                        _flush()

                # Drop tiles no pending region can still need. Sorting by (last, first) tile
                # means every remaining region starts at or after the next one's first tile.
                if next_region < len(pending):
                    keep_from = pending[next_region][1][0][0]
                else:
                    keep_from = last_run + 1
                for tile_index in [t for t in tile_features if t < keep_from]:
                    del tile_features[tile_index]
                    tile_features_rc.pop(tile_index, None)
        _flush()
        writer.close()
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, chrom_path)


def embed_regions(
    regions_df: pl.DataFrame,
    genome: Genome,
    adapter: BaseBackboneAdapter,
    out_dir: str | Path,
    *,
    backbone: str,
    checkpoint: str = "",
    chroms: Sequence[str] | None = None,
    rc: bool = False,
    pool_to: int | None = None,
    context: int = 1_048_576,
    stride: int = 524_288,
    batch_size: int = 1,
    device: str = "cpu",
    row_group_size: int = 256,
) -> None:
    """Cache *adapter*'s frozen embeddings over every region's target in *regions_df*.

    Writes one ``<out_dir>/<chrom>.parquet`` per chromosome and one shared
    ``<out_dir>/manifest.parquet`` (written on first use, checked on every rerun --
    see :func:`validate_manifest`). Already-finished chromosome files are skipped, so
    this can be called once per chromosome (e.g. one Slurm array job per chromosome via
    ``chroms=[name]``) and resumed freely.

    Parameters
    ----------
    regions_df
        The region dataset's ``regions.parquet`` (``RegionCountData.regions``): needs
        ``chrom``, ``target_start``, ``target_end``. Row order gives ``region_row``.
    genome
        Opened via :func:`regulonado.sequence.open_genome`.
    adapter
        A built backbone adapter (or, in tests, any :class:`BaseBackboneAdapter`
        subclass) -- passed in directly so tests can inject a stub.
    backbone, checkpoint
        Recorded in the manifest for provenance; not otherwise interpreted.
    chroms
        Only cache these chromosomes; default is every chromosome in *regions_df*.
    pool_to
        Average adjacent raw bins up to this bp width. Must be a multiple of the
        backbone's ``output_bin_size``.
    context, stride
        Tiling parameters for flexible backbones (see module docstring); ignored for
        fixed-input backbones other than being recorded in the manifest.
    device
        Already-resolved torch device string (``"cpu"``/``"cuda"``/``"mps"``); CLI-level
        ``--device auto`` resolution happens before this call.

    Raises
    ------
    ValueError
        If ``target_end - target_start`` is not constant across regions, if ``pool_to``
        is not a multiple of the backbone bin size, or if an existing manifest does not
        match *regions_df* or the settings given here.
    """
    for column in _REQUIRED_REGION_COLUMNS:
        if column not in regions_df.columns:
            raise ValueError(f"regions_df is missing required column {column!r}")

    bin_size = adapter.output_bin_size
    if pool_to is not None:
        if pool_to % bin_size != 0 or pool_to < bin_size:
            raise ValueError(f"--pool-to {pool_to} must be a positive multiple of {bin_size}")
    effective_bin_size = pool_to if pool_to is not None else bin_size
    pool_factor = effective_bin_size // bin_size

    target_width = _target_width(regions_df)
    k = math.ceil(target_width / effective_bin_size) + 1
    d = int(adapter.feature_dim)

    input_length, keep_offset_bp, keep_bp = _window_geometry(adapter, context, stride)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = EmbeddingManifest(
        backbone=backbone,
        checkpoint=checkpoint,
        bin_size=effective_bin_size,
        k=k,
        d=d,
        context=context,
        stride=stride,
        pool_to=pool_to,
        rc=rc,
        region_hash=region_table_hash(regions_df),
        n_regions=regions_df.height,
    )
    manifest_path = out_path / MANIFEST_FILENAME
    if manifest_path.exists():
        existing = read_manifest(out_path)
        if existing != manifest:
            raise ValueError(
                f"{manifest_path} already holds different settings than this run "
                f"({existing} vs {manifest}); use a fresh --out directory for a different "
                "backbone/region-table/pool/rc/context/stride combination"
            )
    else:
        _write_manifest(out_path, manifest)

    regions_indexed = regions_df.with_row_index("region_row").with_columns(
        pl.col("region_row").cast(pl.Int64)
    )
    chrom_list = (
        list(chroms) if chroms is not None else sorted(regions_indexed["chrom"].unique().to_list())
    )

    adapter = adapter.to(device)
    adapter.eval()

    for chrom in chrom_list:
        chrom_path = out_path / f"{chrom}.parquet"
        if chrom_path.exists():
            continue
        chrom_regions = regions_indexed.filter(pl.col("chrom") == chrom)
        _embed_chrom(
            chrom=chrom,
            chrom_regions=chrom_regions,
            genome=genome,
            adapter=adapter,
            input_length=input_length,
            keep_offset_bp=keep_offset_bp,
            keep_bp=keep_bp,
            bin_size=bin_size,
            pool_factor=pool_factor,
            k=k,
            d=d,
            rc=rc,
            batch_size=batch_size,
            device=device,
            row_group_size=row_group_size,
            chrom_path=chrom_path,
        )


class EmbeddingStore:
    """Random access, by ``region_row``, over one embeddings directory.

    Reuses ``training/data.py``'s ``WindowParquetDataset`` pattern: shard footers are
    read once in the main process to build an index, and no Parquet file handle is
    opened until the first :meth:`get` call -- so nothing keeps an open file descriptor
    across a ``DataLoader`` worker fork. Each worker lazily opens (and caches) its own
    ``pq.ParquetFile`` per chromosome shard.

    Unlike ``WindowParquetDataset``, the index here maps a possibly-sparse,
    non-contiguous ``region_row`` (only regions on chromosomes that have been embedded
    so far) to its shard/row-group/local-row location, since a directory built one
    chromosome at a time may not yet cover every region.
    """

    def __init__(self, embeddings_dir: str | Path) -> None:
        self.dir = Path(embeddings_dir)
        self.manifest = read_manifest(self.dir)
        self.k = self.manifest.k
        self.d = self.manifest.d
        self.has_rc = self.manifest.rc
        self._columns = ["region_row", "features"] + (["features_rc"] if self.has_rc else [])

        paths = sorted(p for p in self.dir.glob("*.parquet") if p.name != MANIFEST_FILENAME)
        self._paths = paths
        self._metadata = [pq.read_metadata(str(p)) for p in paths]

        index: dict[int, tuple[int, int, int]] = {}
        for shard_idx, metadata in enumerate(self._metadata):
            file = pq.ParquetFile(str(paths[shard_idx]), metadata=metadata)
            for row_group_idx in range(metadata.num_row_groups):
                region_rows = (
                    file.read_row_group(row_group_idx, columns=["region_row"])
                    .column("region_row")
                    .to_numpy()
                )
                for local_row, region_row in enumerate(region_rows):
                    index[int(region_row)] = (shard_idx, row_group_idx, local_row)
        self._index = index
        self._files: dict[int, pq.ParquetFile] = {}
        self._preloaded: dict[int, dict[str, np.ndarray]] = {}

    @property
    def region_rows(self) -> list[int]:
        """The ``region_row`` values currently present in this store, unordered."""
        return list(self._index.keys())

    def _file(self, shard_idx: int) -> pq.ParquetFile:
        file = self._files.get(shard_idx)
        if file is None:
            file = pq.ParquetFile(
                str(self._paths[shard_idx]), metadata=self._metadata[shard_idx], pre_buffer=True
            )
            self._files[shard_idx] = file
        return file

    def preload(self, region_rows: Iterable[int] | None = None) -> None:
        """Load features for *region_rows* (default: everything) into RAM."""
        rows = list(region_rows) if region_rows is not None else self.region_rows
        for region_row in rows:
            if region_row not in self._preloaded:
                self._preloaded[region_row] = self._read(region_row)

    def get(self, region_row: int, rc: bool = False) -> np.ndarray:
        """``[K, D]`` float16 features for *region_row* (forward, or RC if requested).

        Raises:
            KeyError: if *region_row* is not in this store.
            ValueError: if ``rc=True`` but this store has no ``features_rc`` column.
        """
        if rc and not self.has_rc:
            raise ValueError(f"{self.dir} has no features_rc column (built without --rc)")
        data = self._preloaded.get(region_row)
        if data is None:
            data = self._read(region_row)
        return data["features_rc" if rc else "features"]

    def _read(self, region_row: int) -> dict[str, np.ndarray]:
        try:
            shard_idx, row_group_idx, local_row = self._index[region_row]
        except KeyError:
            raise KeyError(f"region_row {region_row} not found in {self.dir}") from None
        table = self._file(shard_idx).read_row_group(row_group_idx, columns=self._columns)
        result = {"features": _decode_fixed_size_list(table, "features", self.k, self.d)[local_row]}
        if self.has_rc:
            result["features_rc"] = _decode_fixed_size_list(table, "features_rc", self.k, self.d)[
                local_row
            ]
        return result

    def __getstate__(self) -> dict:
        # Never carry open file handles across a pickle/fork boundary (DataLoader workers).
        state = self.__dict__.copy()
        state["_files"] = {}
        return state
