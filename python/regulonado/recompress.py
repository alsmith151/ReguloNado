"""
Recompress a HuggingFace Arrow dataset to ZSTD IPC body compression.

Reads each shard one at a time (stream, no full-dataset RAM spike), rewrites
with ZSTD, then copies all non-Arrow metadata files verbatim so the output
directory is a drop-in replacement for load_from_disk().

Use through the CLI:
    regulonado recompress-dataset <src> <dst> --level 3 --workers 4
"""

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from loguru import logger


def recompress_shard(
    src: Path, dst: Path, level: int, max_batch_size: int | None
) -> tuple[int, int]:
    opts = ipc.IpcWriteOptions(compression=pa.Codec("zstd", level))
    with open(src, "rb") as fh:
        reader = ipc.open_stream(fh)
        schema = reader.schema
        with open(dst, "wb") as out_fh:
            with ipc.new_stream(out_fh, schema, options=opts) as writer:
                for batch in reader:
                    if max_batch_size and batch.num_rows > max_batch_size:
                        for start in range(0, batch.num_rows, max_batch_size):
                            writer.write_batch(
                                batch.slice(start, min(max_batch_size, batch.num_rows - start))
                            )
                    else:
                        writer.write_batch(batch)

    return src.stat().st_size, dst.stat().st_size


def recompress_split(
    split_src: Path, split_dst: Path, level: int, workers: int, max_batch_size: int | None
) -> None:
    split_dst.mkdir(parents=True, exist_ok=True)

    # Copy metadata files verbatim
    for f in split_src.iterdir():
        if not f.name.endswith(".arrow"):
            shutil.copy2(f, split_dst / f.name)

    shards = sorted(split_src.glob("*.arrow"))
    total_src = total_dst = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(recompress_shard, s, split_dst / s.name, level, max_batch_size): s
            for s in shards
        }
        for i, fut in enumerate(as_completed(futures), 1):
            shard = futures[fut]
            try:
                s, d = fut.result()
                total_src += s
                total_dst += d
                logger.info(
                    f"  [{i:3d}/{len(shards)}] {shard.name}  "
                    f"{s/1e6:.0f} MB → {d/1e6:.0f} MB  "
                    f"({d/s:.2f}x)"
                )
            except Exception as e:
                logger.error(f"  ERROR {shard.name}: {e}")
                raise

    ratio = total_dst / total_src if total_src else 1.0
    logger.info(
        f"  split total: {total_src/1e9:.2f} GB → {total_dst/1e9:.2f} GB  "
        f"({ratio:.2f}x, saved {(total_src-total_dst)/1e9:.2f} GB)"
    )


def recompress_dataset(
    src: Path,
    dst: Path,
    *,
    level: int = 3,
    workers: int = 4,
    max_batch_size: int | None = None,
    remove_src: bool = False,
    overwrite: bool = False,
) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be distinct, non-nested directories")
    if dst.exists():
        if not overwrite:
            raise ValueError(f"Destination already exists — remove it first: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary_dst = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        # Copy top-level metadata files verbatim (dataset_dict.json, regulonado_metadata.json, etc.)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, temporary_dst / f.name)
                logger.info(f"copied  {f.name}")

        splits = [d for d in src.iterdir() if d.is_dir()]
        for split in sorted(splits):
            logger.info(f"\n=== {split.name} ===")
            recompress_split(split, temporary_dst / split.name, level, workers, max_batch_size)

        backup_dst: Path | None = None
        if dst.exists():
            backup_dst = Path(tempfile.mkdtemp(prefix=f".{dst.name}.backup.", dir=dst.parent))
            backup_dst.rmdir()
            dst.rename(backup_dst)
        try:
            temporary_dst.rename(dst)
        except Exception:
            if backup_dst is not None and not dst.exists():
                backup_dst.rename(dst)
            raise
        if backup_dst is not None:
            shutil.rmtree(backup_dst)
            backup_dst = None
    finally:
        if temporary_dst.exists():
            shutil.rmtree(temporary_dst)

    logger.info(f"\nDone. Output: {dst}")

    if remove_src:
        logger.info(f"\nRemoving source: {src}")
        shutil.rmtree(src)
        logger.info("Source removed.")
