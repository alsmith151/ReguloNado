"""
Recompress a HuggingFace Arrow dataset to ZSTD IPC body compression.

Reads each shard one at a time (stream, no full-dataset RAM spike), rewrites
with ZSTD, then copies all non-Arrow metadata files verbatim so the output
directory is a drop-in replacement for load_from_disk().

Usage:
    python scripts/recompress_dataset.py <src> <dst> [--level 3] [--workers 4]
"""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def recompress_shard(
    src: Path, dst: Path, level: int, max_batch_size: int | None
) -> tuple[int, int]:
    opts = ipc.IpcWriteOptions(compression=pa.Codec("zstd", level))
    with open(src, "rb") as fh:
        reader = ipc.open_stream(fh)
        schema = reader.schema
        batches = list(reader)

    with open(dst, "wb") as fh:
        with ipc.new_stream(fh, schema, options=opts) as writer:
            for batch in batches:
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
                print(
                    f"  [{i:3d}/{len(shards)}] {shard.name}  "
                    f"{s/1e6:.0f} MB → {d/1e6:.0f} MB  "
                    f"({d/s:.2f}x)",
                    flush=True,
                )
            except Exception as e:
                print(f"  ERROR {shard.name}: {e}", file=sys.stderr)
                raise

    ratio = total_dst / total_src if total_src else 1.0
    print(
        f"  split total: {total_src/1e9:.2f} GB → {total_dst/1e9:.2f} GB  "
        f"({ratio:.2f}x, saved {(total_src-total_dst)/1e9:.2f} GB)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Source dataset directory (load_from_disk path)")
    ap.add_argument("dst", help="Destination directory (must not exist)")
    ap.add_argument("--level", type=int, default=3, help="ZSTD compression level (default 3)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel shard workers per split")
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        dest="max_batch_size",
        help=(
            "Split each Arrow record batch into sub-batches of at most this many rows. "
            "Recommended: 4–8. Reduces peak worker RAM from O(batch_rows × sample_size) "
            "to O(max_batch_size × sample_size) — critical for large-track datasets "
            "where 200+ row batches decompress to tens of GB per DataLoader worker."
        ),
    )
    ap.add_argument("--remove-src", action="store_true", help="Delete source dataset after successful recompression")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.exists():
        sys.exit(f"Source not found: {src}")
    if dst.exists():
        sys.exit(f"Destination already exists — remove it first: {dst}")

    dst.mkdir(parents=True)

    # Copy top-level metadata files verbatim (dataset_dict.json, regulonado_metadata.json, etc.)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            print(f"copied  {f.name}")

    # Recompress each split
    splits = [d for d in src.iterdir() if d.is_dir()]
    for split in sorted(splits):
        print(f"\n=== {split.name} ===")
        recompress_split(split, dst / split.name, args.level, args.workers, args.max_batch_size)

    print(f"\nDone. Output: {dst}")

    if args.remove_src:
        print(f"\nRemoving source: {src}")
        shutil.rmtree(src)
        print("Source removed.")


if __name__ == "__main__":
    main()
