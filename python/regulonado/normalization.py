from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

BAMNADO = os.environ.get("BAMNADO", "bamnado")

log = logging.getLogger(__name__)


def read_regions(path: Path) -> list[tuple[str, int, int]]:
    """Read ``chrom/start/end`` regions from parquet or a BED-like file."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        columns = {str(c).lower(): c for c in frame.columns}
        try:
            cols = [columns[name] for name in ("chrom", "start", "end")]
        except KeyError as exc:
            raise ValueError(
                f"Region parquet must contain chrom/start/end columns: {path}"
            ) from exc
        return [
            (str(chrom), int(start), int(end))
            for chrom, start, end in frame[cols].itertuples(index=False, name=None)
        ]
    import bioframe as bf

    frame = bf.read_table(str(path), schema="bed")
    return [
        (str(chrom), int(start), int(end))
        for chrom, start, end in frame[["chrom", "start", "end"]].itertuples(
            index=False, name=None
        )
    ]


def track_window_stat(reader, windows, *, bin_size: int, window_stat_bp: int) -> np.ndarray:
    """Summarise each window using the maximum rolling mean at model bin size."""
    stats = []
    width = max(1, round(window_stat_bp / bin_size))
    for chrom, start, end in windows:
        n_bins = max(1, int(np.ceil((end - start) / bin_size)))
        values = np.asarray(
            reader.values(chrom, start, end, bins=n_bins, summary="mean", exact=True, missing=0),
            dtype=np.float32,
        )
        np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if width >= len(values):
            stats.append(float(values.mean()))
        else:
            means = np.convolve(values, np.ones(width, dtype=np.float32) / width, mode="valid")
            stats.append(float(means.max()))
    return np.asarray(stats, dtype=np.float32)


def anchor_scale_factors(
    track_paths: Sequence[Path],
    anchor_regions: Path,
    background_regions: Path,
    *,
    heldout_regions: Path | None = None,
    bin_size: int,
    window_stat_bp: int = 1000,
    window_stat_bp_by_assay: Mapping[str, int] | None = None,
    assays: Sequence[str | None] | None = None,
    background_sample: int | None = 5000,
    max_workers: int = 16,
    clip_soft_anchors: float = 10.0,
    clip_hard_anchors: float = 20.0,
) -> pd.DataFrame:
    """Calculate biological anchor normalisation factors directly from BigWigs."""
    import pybigtools

    anchors = read_regions(anchor_regions)
    backgrounds = read_regions(background_regions)
    heldout = read_regions(heldout_regions) if heldout_regions is not None else None
    if background_sample is not None and len(backgrounds) > background_sample:
        rng = np.random.default_rng(0)
        backgrounds = [
            backgrounds[i]
            for i in rng.choice(len(backgrounds), background_sample, replace=False)
        ]
    assays = list(assays) if assays is not None else [None] * len(track_paths)

    def calculate(index: int) -> dict:
        path = Path(track_paths[index])
        assay = assays[index] if index < len(assays) else None
        stat_bp = (window_stat_bp_by_assay or {}).get(str(assay), window_stat_bp)
        reader = pybigtools.open(str(path))
        try:
            anchor = track_window_stat(reader, anchors, bin_size=bin_size, window_stat_bp=stat_bp)
            background = track_window_stat(
                reader, backgrounds, bin_size=bin_size, window_stat_bp=stat_bp
            )
            heldout_stat = (
                track_window_stat(reader, heldout, bin_size=bin_size, window_stat_bp=stat_bp)
                if heldout is not None else np.array([], dtype=np.float32)
            )
        finally:
            close = getattr(reader, "close", None)
            if close:
                close()
        bg_q50 = float(np.quantile(background, 0.5))
        bg_q99 = float(np.quantile(background, 0.99))
        anchor_ref = float(np.median(anchor))
        recovery = float(np.mean(heldout_stat > bg_q99)) if heldout_stat.size else float("nan")
        quality = "failed" if anchor_ref <= bg_q50 else (
            "informative" if not heldout_stat.size or recovery >= 0.70 else
            "weak" if recovery >= 0.30 else "failed"
        )
        sf = 1.0 / (anchor_ref - bg_q50) if anchor_ref > bg_q50 else 1.0
        return {
            "track_index": index,
            "scale_factor": sf,
            "background": bg_q50,
            "clip_soft": clip_soft_anchors,
            "clip_hard": clip_hard_anchors,
            "anchor_reference": anchor_ref,
            "background_q50": bg_q50,
            "background_q99": bg_q99,
            "heldout_reference": float(np.median(heldout_stat)) if heldout_stat.size else np.nan,
            "heldout_recovery": recovery,
            "window_stat_bp": stat_bp,
            "quality": quality,
            "path": str(path),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(calculate, range(len(track_paths))))
    return pd.DataFrame(rows)


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
        return pd.Series(
            {
                "scale_factor": 1.0,
                "library_size": 0,
                "samplename": bw.stem,
                "path": str(path),
            }
        )
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
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Inferring scale factors"
        ):
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


def compute_bamnado_norm_factors(
    bam_paths: list[Path],
    *,
    method: str = "csaw-background",
    bin_size_bp: int = 10_000,
    exclude_top_percent: float = 5.0,
    reference_sample: str | None = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    exogenous_prefix: str | None = None,
) -> np.ndarray:
    """Run ``bamnado bam-normalize`` over BAM files and return per-track norm factors.

    Returns bamnado's ``norm_factors`` (geometric-mean-normalised correction
    factors), in the same order as ``bam_paths`` — the direct analogue of
    :func:`compute_tmm_factors`'s output, but estimated by bamnado directly
    from whole-genome BAM coverage rather than regulonado's own TMM
    implementation restricted to the dataset's region subset.
    """
    _check_bamnado()

    cmd = [BAMNADO, "bam-normalize"]
    for bam in bam_paths:
        cmd += ["--bams", str(Path(bam).resolve())]
    cmd += [
        "--method", method,
        "--bin-size", str(bin_size_bp),
        "--exclude-top-percent", str(exclude_top_percent),
        "--logratio-trim", str(logratio_trim),
        "--sum-trim", str(sum_trim),
        "--format", "json",
    ]
    if reference_sample is not None:
        cmd += ["--reference-sample", reference_sample]
    if exogenous_prefix is not None:
        cmd += ["--exogenous-prefix", exogenous_prefix]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"'{BAMNADO} bam-normalize' failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )

    result = json.loads(proc.stdout)
    sample_names = result["sample_names"]
    expected = [Path(bam).stem for bam in bam_paths]
    if sample_names != expected:
        raise RuntimeError(
            "bamnado returned samples in an unexpected order: "
            f"expected {expected}, got {sample_names}"
        )
    return np.asarray(result["norm_factors"], dtype=np.float64)


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
