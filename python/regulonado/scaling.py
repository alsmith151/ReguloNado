from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

BAMNADO = os.environ.get("BAMNADO", "bamnado")

log = logging.getLogger(__name__)


def _check_bamnado() -> None:
    """Raise a clear error if the bamnado binary is not on PATH."""
    import shutil
    if shutil.which(BAMNADO) is None:
        raise RuntimeError(
            f"'{BAMNADO}' binary not found on PATH.\n"
            "Install it from https://github.com/alsmith151/BamNado/releases or set "
            "the BAMNADO environment variable to the full path of the binary.\n"
            "Alternatively, skip this step and set scale_factor=1 manually in the "
            "metadata if your BigWigs are already in raw-count units."
        )


def infer_scale_factor(bw: Path) -> pd.Series:
    path = bw.resolve()
    cmd = [BAMNADO, "bigwig-infer-scale", "--bigwig", str(path), "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.warning(
            "bamnado failed for %s (exit %d): %s — using scale_factor=1, library_size=0",
            path,
            proc.returncode,
            proc.stderr.strip(),
        )
        return pd.Series({"scale_factor": 1.0, "library_size": 0, "samplename": bw.stem, "path": str(path)})
    result = json.loads(proc.stdout)
    result["samplename"] = bw.stem
    result["path"] = str(path)
    return pd.Series(result)


def infer_scale_factors(
    bw_files: list[Path],
    max_workers: int = 16,
) -> pd.DataFrame:
    _check_bamnado()
    rows: list[pd.Series] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(infer_scale_factor, bw): bw for bw in bw_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring scale factors"):
            rows.append(future.result())
    return pd.concat(rows, axis=1).T.reset_index(drop=True)


def compute_clip_thresholds(
    df: pd.DataFrame,
    *,
    soft_reads_per_million: float = 7.0,
    hard_reads_per_million: float = 16.0,
) -> pd.DataFrame:
    """Add ``clip_soft`` and ``clip_hard`` columns to a scale-factors DataFrame.

    Thresholds are in raw-count space (i.e. after multiplying the BigWig signal
    by ``scale_factor``).  They scale linearly with ``library_size`` so that the
    same fraction of the expected signal distribution is preserved regardless of
    sequencing depth.

    The defaults (7 and 16 reads per million) are calibrated against the
    hardcoded fallbacks used in ``_resolve_scale_and_clip`` (348 / 796), which
    were tuned empirically for ~50 M-read ChIP-seq libraries:
        50 × 7  ≈ 350   50 × 16 ≈ 800

    Soft clip:  linear below threshold, sqrt-compressed above (in squash space).
    Hard clip:  absolute ceiling applied before the squash transform.
    """
    df = df.copy()
    lib = df["library_size"].astype(float)
    df["clip_soft"] = (soft_reads_per_million * lib / 1e6).round(1)
    df["clip_hard"] = (hard_reads_per_million * lib / 1e6).round(1)
    return df


def _read_arrow_shard_shape(shard_path: Path) -> tuple[int, int]:
    """Return (n_tracks, n_bins) from the labels field metadata of the first shard."""
    import pyarrow.ipc as ipc

    reader = ipc.open_stream(str(shard_path))
    schema = reader.schema_arrow
    meta = schema.field("labels").metadata
    if meta is None:
        raise ValueError(f"No field metadata on 'labels' in {shard_path}")
    shape_str = json.loads(meta[b"ARROW:extension:metadata"])
    n_tracks, n_bins = shape_str[0]
    return int(n_tracks), int(n_bins)


def read_dataset_means(
    dataset_dir: Path,
    *,
    split: str = "train",
) -> tuple[np.ndarray, int, int]:
    """Read per-sample per-track mean RPKM from Arrow IPC shards.

    Returns:
        means:    (n_samples, n_tracks) float32 mean signal per region
        n_tracks: number of signal tracks
        n_bins:   number of bins per sample (including shift buffer)
    """
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    shard_dir = dataset_dir / split
    shards = sorted(shard_dir.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No .arrow shards found in {shard_dir}")

    n_tracks, n_bins = _read_arrow_shard_shape(shards[0])

    chunks: list[np.ndarray] = []
    for shard_path in tqdm(shards, desc="Reading shards"):
        reader = ipc.open_stream(str(shard_path))
        for batch in reader:
            n_rows = batch.num_rows
            col = batch.column("labels")
            # Flatten list<list<float>> → flat float array then reshape
            flat = pc.list_flatten(pc.list_flatten(col))
            arr = np.frombuffer(flat.buffers()[-1], dtype=np.float32)
            # Buffer may be larger than needed due to Arrow alignment padding
            arr = arr[: n_rows * n_tracks * n_bins].reshape(n_rows, n_tracks, n_bins)
            chunks.append(arr.mean(axis=2))  # (n_rows, n_tracks)

    return np.concatenate(chunks, axis=0), n_tracks, n_bins


def compute_tmm_factors(
    means: np.ndarray,
    library_sizes: np.ndarray,
    region_length_kb: float,
    *,
    trim_m: float = 0.3,
    trim_a: float = 0.05,
    min_count: float = 1.0,
) -> np.ndarray:
    """Compute edgeR-style TMM normalisation factors from per-region mean RPKM.

    Each row of ``means`` is one genomic region (~200 kb).  The algorithm is
    identical to edgeR::calcNormFactors(method="TMM"):

    1. Convert mean RPKM to pseudo-counts using library sizes.
    2. Pick the reference track whose 75th-percentile pseudo-count is closest
       to the cross-track mean (edgeR default).
    3. For each track k vs reference r:
       - Compute M = log2(y_k / y_r) and A = 0.5*(log2 y_k + log2 y_r)
         where y = count / library_size.
       - Trim the top/bottom ``trim_m`` of M and ``trim_a`` of A.
       - Weighted mean of remaining M values (precision weights from a
         binomial model).
       - TMM_k = 2 ^ weighted_mean_M.
    4. Normalise so the geometric mean of all factors equals 1.

    Args:
        means:             (n_samples, n_tracks) mean RPKM per region.
        library_sizes:     (n_tracks,) mapped-read counts (not millions).
        region_length_kb:  Length of each region in kilobases.
                           Used to convert mean RPKM → pseudo-counts:
                           count = mean_RPKM * region_length_kb * (lib / 1e6).
        trim_m:            Fraction to trim from each tail of M values (default 0.3).
        trim_a:            Fraction to trim from each tail of A values (default 0.05).
        min_count:         Minimum pseudo-count for a region to be included.

    Returns:
        (n_tracks,) TMM factors normalised to geometric mean = 1.
        Divide the per-track scale_factor by the corresponding TMM factor to
        obtain TMM-adjusted raw-count scale factors.
    """
    means = np.asarray(means, dtype=np.float64)
    lib = np.asarray(library_sizes, dtype=np.float64)
    n_samples, n_tracks = means.shape

    # Pseudo-counts: mean_RPKM * region_kb * (lib / 1e6)
    counts = means * (region_length_kb * lib[np.newaxis, :] / 1e6)  # (n_samples, n_tracks)

    # Reference: track with 75th-percentile pseudo-count closest to cross-track mean
    uq75 = np.nanquantile(counts, 0.75, axis=0)
    ref_idx = int(np.argmin(np.abs(uq75 - uq75.mean())))

    ref_counts = counts[:, ref_idx]
    L_r = lib[ref_idx]

    tmm = np.ones(n_tracks, dtype=np.float64)

    for k in range(n_tracks):
        if k == ref_idx:
            continue

        N_k = counts[:, k]
        L_k = lib[k]

        mask = (N_k >= min_count) & (ref_counts >= min_count)
        if mask.sum() < 10:
            continue

        N_k_m, N_r_m = N_k[mask], ref_counts[mask]

        M = np.log2(N_k_m / L_k) - np.log2(N_r_m / L_r)
        A = 0.5 * (np.log2(N_k_m / L_k) + np.log2(N_r_m / L_r))

        valid = np.isfinite(M) & np.isfinite(A)
        M, A = M[valid], A[valid]
        N_k_m, N_r_m = N_k_m[valid], N_r_m[valid]

        if len(M) < 10:
            continue

        m_lo, m_hi = np.quantile(M, [trim_m, 1.0 - trim_m])
        a_lo, a_hi = np.quantile(A, [trim_a, 1.0 - trim_a])
        keep = (M >= m_lo) & (M <= m_hi) & (A >= a_lo) & (A <= a_hi)

        if keep.sum() < 5:
            continue

        M_k = M[keep]
        N_k_f, N_r_f = N_k_m[keep], N_r_m[keep]

        # Precision weights: inverse variance under a binomial count model
        w = (L_k - N_k_f) / (L_k * N_k_f) + (L_r - N_r_f) / (L_r * N_r_f)
        w = np.maximum(w, 1e-10)

        tmm[k] = 2.0 ** (np.sum(w * M_k) / np.sum(w))

    # Normalise to geometric mean = 1 so no track is arbitrarily chosen as baseline
    tmm /= np.exp(np.mean(np.log(tmm)))
    return tmm


def save_scale_factors(
    df: pd.DataFrame,
    output: Path,
    fmt: Literal["csv", "parquet"] = "parquet",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)
