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
from scipy.stats import rankdata
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
    from regulonado.genomics import read_intervals

    frame = read_intervals(path)
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
    """Infer one BigWig's raw-count scale factor and library size via bamnado.

    Runs ``bamnado bigwig-infer-scale`` on ``bw`` and returns its JSON result
    (``scale_factor``, ``library_size``, ...) as a ``pd.Series``, augmented
    with ``samplename`` (the file stem) and the resolved ``path``. If bamnado
    fails, logs a warning and falls back to ``scale_factor=1.0,
    library_size=0`` rather than raising, so a single bad track does not
    abort a batch run via :func:`infer_scale_factors`.
    """
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
    """Infer raw-count scale factors and library sizes for many BigWigs.

    Runs :func:`infer_scale_factor` over ``bw_files`` concurrently (bamnado
    is invoked once per file as a subprocess) and collects the results into
    one DataFrame, in completion order rather than input order.

    Raises:
        RuntimeError: if the ``bamnado`` binary is not on PATH (see
            :func:`_check_bamnado`).
    """
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


def _tmm_reference_index(counts: np.ndarray, lib: np.ndarray) -> int:
    """Pick the edgeR TMM reference column.

    edgeR's ``.calcFactorQuantile`` computes the 75th-percentile of
    *count / library_size* (not of the raw counts) for every track, then
    picks the track whose value is closest to the cross-track mean. Using
    raw-count quantiles instead (as an earlier version of this function did)
    can select a different reference track whenever library sizes differ
    across tracks, silently changing every downstream M/A comparison.
    """
    rate = counts / lib[np.newaxis, :]
    uq75 = np.quantile(rate, 0.75, axis=0)
    return int(np.argmin(np.abs(uq75 - uq75.mean())))


def _tmm_pair_factor(
    obs_counts: np.ndarray,
    ref_counts: np.ndarray,
    lib_obs: float,
    lib_ref: float,
    *,
    trim_m: float,
    trim_a: float,
    min_count: float,
) -> float:
    """Compute one track's raw (pre-renormalisation) TMM factor vs the reference.

    Mirrors edgeR's ``.calcFactorTMM`` exactly:

    - M = log2((obs/lib_obs) / (ref/lib_ref)), A = 0.5*log2((obs/lib_obs)*(ref/lib_ref)).
    - Regions are kept by *rank* position (edgeR's ``rank()``-based cutoffs),
      not by interpolated quantile boundaries — the two disagree whenever M or A
      contain ties, which is common with integer-like pseudo-counts.
    - The weighted mean of M uses precision weights ``1/v`` (inverse of the
      binomial-approximation variance ``v``); weighting by ``v`` directly (as an
      earlier version of this function did) inverts the intended effect, giving
      the *least* reliable (highest-variance) regions the most influence.
    """
    mask = (obs_counts >= min_count) & (ref_counts >= min_count)
    if mask.sum() < 10:
        return 1.0

    obs_m, ref_m = obs_counts[mask], ref_counts[mask]
    M = np.log2(obs_m / lib_obs) - np.log2(ref_m / lib_ref)
    A = 0.5 * (np.log2(obs_m / lib_obs) + np.log2(ref_m / lib_ref))

    valid = np.isfinite(M) & np.isfinite(A)
    M, A = M[valid], A[valid]
    obs_m, ref_m = obs_m[valid], ref_m[valid]
    n = len(M)
    if n < 10:
        return 1.0

    lo_m = np.floor(n * trim_m) + 1
    hi_m = n + 1 - lo_m
    lo_a = np.floor(n * trim_a) + 1
    hi_a = n + 1 - lo_a
    rank_m = rankdata(M)
    rank_a = rankdata(A)
    keep = (rank_m >= lo_m) & (rank_m <= hi_m) & (rank_a >= lo_a) & (rank_a <= hi_a)
    if keep.sum() < 5:
        return 1.0

    M_k, obs_k, ref_k = M[keep], obs_m[keep], ref_m[keep]

    # Binomial-approximation variance of M; precision weight is its inverse.
    variance = (lib_obs - obs_k) / (lib_obs * obs_k) + (lib_ref - ref_k) / (lib_ref * ref_k)
    variance = np.maximum(variance, 1e-10)
    weight = 1.0 / variance

    return float(2.0 ** (np.sum(weight * M_k) / np.sum(weight)))


def _tmm_from_counts(
    counts: np.ndarray,
    lib: np.ndarray,
    *,
    trim_m: float = 0.3,
    trim_a: float = 0.05,
    min_count: float = 1.0,
) -> np.ndarray:
    """Compute edgeR-equivalent TMM factors from a pseudo-count matrix.

    Pure counts -> factors kernel, factored out of :func:`compute_tmm_factors`
    so it can be golden-tested directly against edgeR::calcNormFactors output
    without going through the RPKM->pseudo-count conversion.

    Args:
        counts: (n_samples, n_tracks) non-negative pseudo-counts.
        lib:    (n_tracks,) library sizes used to normalise counts to rates.

    Returns:
        (n_tracks,) TMM factors normalised to geometric mean = 1.
    """
    n_tracks = counts.shape[1]
    ref_idx = _tmm_reference_index(counts, lib)
    ref_counts = counts[:, ref_idx]
    lib_ref = lib[ref_idx]

    tmm = np.ones(n_tracks, dtype=np.float64)
    for k in range(n_tracks):
        if k == ref_idx:
            continue
        tmm[k] = _tmm_pair_factor(
            counts[:, k],
            ref_counts,
            lib[k],
            lib_ref,
            trim_m=trim_m,
            trim_a=trim_a,
            min_count=min_count,
        )

    # Normalise to geometric mean = 1 so no track is arbitrarily chosen as baseline
    tmm /= np.exp(np.mean(np.log(tmm)))
    return tmm


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

    Each row of ``means`` is one genomic region (~200 kb).  The algorithm
    matches edgeR::calcNormFactors(method="TMM") called with an explicit
    ``lib.size`` (verified against edgeR 4.8.2 to machine precision — see
    ``tests/test_normalization_tmm.py`` and ``tests/data/tmm/``):

    1. Convert mean RPKM to pseudo-counts using library sizes.
    2. Pick the reference track whose 75th-percentile *count/library_size*
       rate is closest to the cross-track mean (edgeR default; see
       :func:`_tmm_reference_index`).
    3. For each track k vs reference r (see :func:`_tmm_pair_factor`):
       - Compute M = log2(y_k / y_r) and A = 0.5*(log2 y_k + log2 y_r)
         where y = count / library_size.
       - Trim the top/bottom ``trim_m`` of M and ``trim_a`` of A by rank.
       - Weighted mean of remaining M values, weighted by inverse variance
         under a binomial count model.
       - TMM_k = 2 ^ weighted_mean_M.
    4. Normalise so the geometric mean of all factors equals 1.

    Note that ``library_sizes`` here is deliberately independent of
    ``means``: ``means`` covers only the curated normalisation regions,
    while ``library_sizes`` is each track's true whole-library mapped-read
    count. This mirrors calling edgeR with an explicit ``lib.size`` that
    differs from ``colSums`` of the input matrix, and is why this kernel is
    hand-rolled rather than delegated to a library (e.g. rnanorm) whose TMM
    implementation always derives library size from the input matrix itself.

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

    # Pseudo-counts: mean_RPKM * region_kb * (lib / 1e6)
    counts = means * (region_length_kb * lib[np.newaxis, :] / 1e6)  # (n_samples, n_tracks)

    return _tmm_from_counts(counts, lib, trim_m=trim_m, trim_a=trim_a, min_count=min_count)


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
    """Write a scale-factors DataFrame to ``output``, creating parent dirs.

    ``fmt`` selects the on-disk format independently of ``output``'s
    extension: ``"parquet"`` (default) writes Parquet, anything else writes
    CSV.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)
