from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from transformers import EvalPrediction

from regulonado.training.group_contrast import group_contrast_labels, reduce_target_score
from regulonado.training.losses import mask_missing_bins, specificity_stats

# Column layout of the group-contrast stats tensor `make_preprocess_logits_for_metrics`
# returns as the second element of its tuple output, shape [B, G, N_GROUP_STAT_COLS]:
#   0-5:   (sp, st, spt, sp2, st2, n) over masked bins, per group -> group_contrast_pearson
#   6-11:  (sp, st, spt, sp2, st2, n) for the target group's reduced score `s`, replicated
#          identically across every group row (see make_compute_metrics: read row 0)
#   12-13: (rank1_pred_numerator, mask_count) replicated across every group row
#   14-15: (rank1_observed_numerator, mask_count) replicated across every group row
#   16:    per-example gauge-residual scalar (median |per-bin median over groups| over this
#          example's masked bins), replicated across every group row; not summable — read
#          raw (unsummed) at compute time and medianed across examples, like `q99_p`/`q99_t`.
N_GROUP_STAT_COLS = 17


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return float("nan") if not np.isfinite(r) else r


def make_preprocess_logits_for_metrics(
    topk_bins: int,
    *,
    contrast_group_weights: torch.Tensor | None = None,
    contrast_region_bins: int = 16,
    contrast_pseudocount: float = 0.1,
    contrast_active_fraction: float = 0.1,
    log_pseudocount: float = 0.1,
    label_divisor: torch.Tensor | None = None,
    group_contrast_group_weights: torch.Tensor | None = None,
    group_contrast_unit_k: torch.Tensor | None = None,
    group_contrast_unit_b: torch.Tensor | None = None,
    group_contrast_target_index: int | None = None,
    group_contrast_smoothing_bins: int = 31,
    group_contrast_pseudocount: float = 0.1,
    group_contrast_gauge: str = "median",
    group_contrast_floor: float = 0.139,
    group_contrast_clamp: tuple[float, float] = (-6.0, 6.0),
    group_contrast_quantile: float = 0.9,
) -> Callable:
    """Return a preprocess_logits_for_metrics function that accumulates Pearson sufficient stats.

    Metrics are computed in the model's output units: ``label_divisor`` (the count label
    space's per-track exposure) divides labels first. Missing (NaN) label bins are
    excluded — zeroed in prediction and target, and left out of every bin count ``n``.

    When ``group_contrast_group_weights`` is given (a group-contrast head is configured and
    scored), the returned callable takes an extra ``group_logits`` argument (the model's
    ``[B, G, L]`` group-contrast channels, sliced out by the caller before the per-track
    ``logits`` it also receives) and returns a 2-tuple ``(track_stats, group_stats)``
    instead of a single tensor — ``transformers``' evaluation loop (``pad_across_processes``
    / ``nested_concat`` / ``nested_numpify``, verified directly against the installed
    version in ``tests/test_group_contrast_metrics.py``) nests arbitrary tuples of tensors
    transparently, concatenating each leaf independently along the batch dimension. When
    ``group_contrast_group_weights`` is ``None`` (no group head, or one configured without a
    resolvable group-contrast config), the return value is the original single ``[B, T, 31]``
    tensor, byte-identical to a build without any group-contrast arguments — ``group_logits``
    is accepted but ignored in that case, so existing callers/metrics are unaffected.

    Returns [B, T, 31] (the per-track tensor) per batch:
      cols 0-5:  (sum_p, sum_t, sum_pt, sum_p², sum_t², n)  over all bins  — per-bin Pearson
      cols 6-11: same statistics restricted to the top-K bins by target signal
      cols 12-17: (sp, st, sp*st, sp², st², 1.0)  where sp/st are per-example
                  track totals — sufficient stats for pearson_total_median
      cols 18-19: per-example q99 prediction and target values
      cols 20-24: (sum_log_p, sum_log_t, sum_log_p*log_t, sum_log_t², n) over bins, with
                   log(x + log_pseudocount) — sufficient stats for dispersion_slope_median
      cols 25-30: (sum_c_p, sum_c_t, sum_c_p*c_t, sum_c_p², sum_c_t², n) over active
                   regions — cross-track specificity stats (see specificity_stats)
    """

    def preprocess(
        logits: torch.Tensor | tuple,
        labels: torch.Tensor,
        group_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(logits, tuple):
            logits = logits[0]
        # float32 regardless of the model output dtype: every column below is a sum feeding
        # an n*Σx² - (Σx)² style difference, which bf16's ~0.8% resolution would swamp.
        p = logits.float()
        # Labels may be loaded as [B, L, T] by the HF datasets library; align to [B, T, L].
        t = labels if labels.shape[-2:] == p.shape[-2:] else labels.transpose(-2, -1)
        t = t.float()
        if label_divisor is not None:
            t = t / label_divisor.to(device=t.device, dtype=t.dtype)[:, None]
        valid = ~torch.isnan(t)
        p, t = mask_missing_bins(p, t)
        B, T, L = p.shape
        k = min(topk_bins, L)

        sp = p.sum(-1)
        st = t.sum(-1)
        spt = (p * t).sum(-1)
        sp2 = (p * p).sum(-1)
        st2 = (t * t).sum(-1)
        n = valid.sum(-1).to(dtype=p.dtype)

        topk_idx = t.topk(k, dim=-1).indices  # [B, T, k]
        p_k = p.gather(-1, topk_idx)
        t_k = t.gather(-1, topk_idx)
        sp_k = p_k.sum(-1)
        st_k = t_k.sum(-1)
        spt_k = (p_k * t_k).sum(-1)
        sp2_k = (p_k * p_k).sum(-1)
        st2_k = (t_k * t_k).sum(-1)
        n_k = torch.full((B, T), float(k), dtype=p.dtype, device=p.device)

        # Total-signal sufficient stats: sp, st are per-example track totals (summed over bins).
        # When accumulated across N eval examples: sum gives cross-example Pearson sufficient stats.
        ones = torch.ones((B, T), dtype=p.dtype, device=p.device)
        q99_p = torch.quantile(p.float(), 0.99, dim=-1).to(dtype=p.dtype)
        q99_t = torch.quantile(t.float(), 0.99, dim=-1).to(dtype=p.dtype)
        # A pseudocount rather than a 1e-8 floor: zero-count bins would otherwise sit at
        # log ~ -18 and dominate the regression.
        log_p = torch.log(p.float().clamp_min(0.0) + log_pseudocount) * valid
        log_t = torch.log(t.float().clamp_min(0.0) + log_pseudocount) * valid
        log_p_sum = log_p.sum(-1).to(dtype=p.dtype)
        log_t_sum = log_t.sum(-1).to(dtype=p.dtype)
        log_pt_sum = (log_p * log_t).sum(-1).to(dtype=p.dtype)
        log_t2_sum = (log_t * log_t).sum(-1).to(dtype=p.dtype)
        specificity = specificity_stats(
            p,
            t,
            contrast_group_weights,
            region_bins=contrast_region_bins,
            pseudocount=contrast_pseudocount,
            active_fraction=contrast_active_fraction,
        ).to(dtype=p.dtype)

        base = torch.stack(
            [
                sp,
                st,
                spt,
                sp2,
                st2,
                n,
                sp_k,
                st_k,
                spt_k,
                sp2_k,
                st2_k,
                n_k,
                sp,
                st,
                sp * st,
                sp * sp,
                st * st,
                ones,
                q99_p,
                q99_t,
                log_p_sum,
                log_t_sum,
                log_pt_sum,
                log_t2_sum,
                n,
            ],
            dim=-1,
        )
        track_stats = torch.cat([base, specificity], dim=-1)

        if group_contrast_group_weights is None or group_logits is None:
            return track_stats

        # `t` here is exactly `normalised_target` (label_divisor-scaled, NaN bins zeroed via
        # mask_missing_bins above) — the same quantity `_group_contrast_loss_terms` feeds
        # `group_contrast_labels` from, so the observed group-contrast label matches the
        # training-time one bin-for-bin.
        unit_k = group_contrast_unit_k.to(device=t.device, dtype=t.dtype)
        scaled_target = t * unit_k[None, :, None]
        unit_b = (
            None
            if group_contrast_unit_b is None
            else group_contrast_unit_b.to(device=t.device, dtype=t.dtype)
        )
        g_labels, g_mask = group_contrast_labels(
            scaled_target,
            group_weights=group_contrast_group_weights,
            background=unit_b,
            smoothing_bins=group_contrast_smoothing_bins,
            pseudocount=group_contrast_pseudocount,
            gauge=group_contrast_gauge,  # type: ignore[arg-type]
            floor=group_contrast_floor,
            clamp=group_contrast_clamp,
        )
        g_pred = group_logits.float()
        G = g_pred.shape[1]
        mask_flat = g_mask[:, 0, :].to(dtype=g_pred.dtype)  # [B, L]

        gp = g_pred * mask_flat[:, None, :]
        gt = g_labels * mask_flat[:, None, :]
        sp_g = gp.sum(-1)
        st_g = gt.sum(-1)
        spt_g = (gp * gt).sum(-1)
        sp2_g = (gp * gp).sum(-1)
        st2_g = (gt * gt).sum(-1)
        n_g = mask_flat.sum(-1, keepdim=True).expand(-1, G)

        Bsz = g_pred.shape[0]
        if group_contrast_target_index is not None:
            pred_score = reduce_target_score(
                g_pred, target_index=group_contrast_target_index, quantile=group_contrast_quantile
            )
            label_score = reduce_target_score(
                g_labels, target_index=group_contrast_target_index, quantile=group_contrast_quantile
            )
            ps = pred_score * mask_flat
            ts = label_score * mask_flat
            sp_s = ps.sum(-1)
            st_s = ts.sum(-1)
            spt_s = (ps * ts).sum(-1)
            sp2_s = (ps * ps).sum(-1)
            st2_s = (ts * ts).sum(-1)
            n_s = mask_flat.sum(-1)

            pred_argmax = g_pred.argmax(dim=1)
            rank1_pred_num = (
                (pred_argmax == group_contrast_target_index).to(dtype=g_pred.dtype) * mask_flat
            ).sum(-1)
            obs_argmax = g_labels.argmax(dim=1)
            rank1_obs_num = (
                (obs_argmax == group_contrast_target_index).to(dtype=g_pred.dtype) * mask_flat
            ).sum(-1)
            rank1_denom = mask_flat.sum(-1)
        else:
            zeros = torch.zeros(Bsz, dtype=g_pred.dtype, device=g_pred.device)
            sp_s = st_s = spt_s = sp2_s = st2_s = n_s = zeros
            rank1_pred_num = rank1_obs_num = rank1_denom = zeros

        # Per-example diagnostic: the predicted gauge should sit near zero. Two-stage
        # median (per-bin median over groups, then median over this example's masked
        # bins) rather than storing full [B, G, L] tensors across the eval set; the raw
        # per-example value is read back unsummed in make_compute_metrics and medianed
        # again across examples, matching the q99_p/q99_t pattern above.
        bin_median = torch.quantile(g_pred, 0.5, dim=1, interpolation="linear")  # [B, L]
        abs_med = bin_median.abs()
        masked_abs = torch.where(mask_flat > 0, abs_med, torch.full_like(abs_med, float("nan")))
        gauge_residual = torch.nanmedian(masked_abs, dim=-1).values  # [B]

        def _rep(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(-1).expand(-1, G)

        group_stats = torch.stack(
            [
                sp_g,
                st_g,
                spt_g,
                sp2_g,
                st2_g,
                n_g,
                _rep(sp_s),
                _rep(st_s),
                _rep(spt_s),
                _rep(sp2_s),
                _rep(st2_s),
                _rep(n_s),
                _rep(rank1_pred_num),
                _rep(rank1_denom),
                _rep(rank1_obs_num),
                _rep(rank1_denom),
                _rep(gauge_residual),
            ],
            dim=-1,
        )
        assert group_stats.shape[-1] == N_GROUP_STAT_COLS
        return track_stats, group_stats

    return preprocess


def make_compute_metrics(
    n_tracks: int,
    calibration_shape_pearson_weight: float = 0.1,
    *,
    per_track_sink: Callable[[dict[str, np.ndarray]], None] | None = None,
    per_group_sink: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Build ``compute_metrics`` over the stats from ``make_preprocess_logits_for_metrics``.

    Logged metrics are medians over tracks (see docs/training.md#evaluation-metrics).
    ``per_track_sink`` receives the per-track ``[T]`` arrays behind those medians.

    When ``eval_pred.predictions`` is a 2-tuple ``(track_stats, group_stats)`` (a
    group-contrast head configured with ``group_contrast_group_weights`` on the
    preprocess side — see :func:`make_preprocess_logits_for_metrics`), the group-space
    metrics documented on :data:`N_GROUP_STAT_COLS` are added to the returned dict, and
    ``per_group_sink`` (if given) receives the per-group ``group_contrast_pearson`` array
    (the caller is expected to already know the group names, e.g. from the same
    ``group_contrast_group_names`` the head was built with). With a plain (non-tuple)
    ``predictions``, behaviour
    is unchanged from a build without any group-contrast arguments.
    """

    def _pearson_from_stats(
        sp: np.ndarray,
        st: np.ndarray,
        spt: np.ndarray,
        sp2: np.ndarray,
        st2: np.ndarray,
        n: np.ndarray,
    ) -> np.ndarray:
        num = n * spt - sp * st
        denom = np.sqrt(np.maximum(n * sp2 - sp**2, 0.0) * np.maximum(n * st2 - st**2, 0.0))
        result = np.full_like(num, np.nan, dtype=np.float64)
        return np.divide(num, denom, out=result, where=denom > 0)

    def _group_metrics(group_stats_raw: np.ndarray) -> dict[str, float]:
        g = np.asarray(group_stats_raw, dtype=np.float64)  # [N, G, N_GROUP_STAT_COLS]
        gs = g.sum(axis=0)  # [G, N_GROUP_STAT_COLS] — summable columns only

        r_group = _pearson_from_stats(gs[:, 0], gs[:, 1], gs[:, 2], gs[:, 3], gs[:, 4], gs[:, 5])
        fin_group = r_group[np.isfinite(r_group)]

        if per_group_sink is not None:
            per_group_sink({"group_contrast_pearson": r_group})

        # Columns 6+ are replicated identically across every group row (see
        # make_preprocess_logits_for_metrics), so summing over examples (axis 0) leaves
        # every row of `gs` equal for those columns; row 0 is as good as any other.
        sp_s, st_s, spt_s, sp2_s, st2_s, n_s = gs[0, 6:12]
        score_pearson = float(
            _pearson_from_stats(
                np.array(sp_s),
                np.array(st_s),
                np.array(spt_s),
                np.array(sp2_s),
                np.array(st2_s),
                np.array(n_s),
            )
        )
        var_p = max(n_s * sp2_s - sp_s**2, 0.0)
        var_t = n_s * st2_s - st_s**2
        score_sd_ratio = float(np.sqrt(var_p / var_t)) if var_t > 1e-12 else float("nan")

        rank1_pred_num, rank1_denom_a = gs[0, 12:14]
        rank1_obs_num, rank1_denom_b = gs[0, 14:16]
        rank1_pred_fraction = (
            float(rank1_pred_num / rank1_denom_a) if rank1_denom_a > 0 else float("nan")
        )
        rank1_obs_fraction = (
            float(rank1_obs_num / rank1_denom_b) if rank1_denom_b > 0 else float("nan")
        )

        # Column 16 (gauge residual) is a per-example diagnostic, not a summable
        # sufficient statistic — read the raw (unsummed) values and median across examples.
        gauge_raw = g[:, 0, 16]
        finite_gauge = gauge_raw[np.isfinite(gauge_raw)]
        gauge_median = float(np.median(finite_gauge)) if finite_gauge.size else float("nan")

        return {
            "group_contrast_pearson_median": (
                float(np.median(fin_group)) if fin_group.size else float("nan")
            ),
            "group_score_pearson": score_pearson,
            "group_score_sd_ratio": score_sd_ratio,
            "group_target_rank1_fraction": rank1_pred_fraction,
            "group_target_rank1_fraction_observed": rank1_obs_fraction,
            "group_gauge_residual_median": gauge_median,
        }

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = eval_pred.predictions
        group_stats_raw: np.ndarray | None = None
        if isinstance(predictions, (tuple, list)) and len(predictions) == 2:
            predictions, group_stats_raw = predictions

        # predictions: [N, T, 31] — sufficient stats accumulated over the full eval set
        stats = np.asarray(predictions, dtype=np.float64)
        s = stats.sum(axis=0)  # [T, 31] global sums

        r_all = _pearson_from_stats(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])
        r_topk = _pearson_from_stats(s[:, 6], s[:, 7], s[:, 8], s[:, 9], s[:, 10], s[:, 11])
        r_total = _pearson_from_stats(s[:, 12], s[:, 13], s[:, 14], s[:, 15], s[:, 16], s[:, 17])

        fin_all = r_all[np.isfinite(r_all)]
        fin_topk = r_topk[np.isfinite(r_topk)]
        fin_total = r_total[np.isfinite(r_total)]
        topk_n = int(round(float(stats[0, 0, 11]))) if stats.shape[0] > 0 else 0

        total_ratios = np.divide(
            s[:, 0], s[:, 1], out=np.full(n_tracks, np.nan), where=np.abs(s[:, 1]) > 1e-12
        )
        finite_ratios = total_ratios[np.isfinite(total_ratios) & (total_ratios > 0)]
        log_ratios = np.log(finite_ratios)
        amplitude_ratios = np.divide(
            stats[:, :, 18],
            stats[:, :, 19],
            out=np.full(stats[:, :, 18].shape, np.nan),
            where=np.abs(stats[:, :, 19]) > 1e-12,
        )
        amplitude_ratios = np.where(amplitude_ratios > 0, amplitude_ratios, np.nan)
        with warnings.catch_warnings():
            # Tracks with no positive q99 target in any example are all-NaN columns.
            warnings.simplefilter("ignore", RuntimeWarning)
            track_amplitude_ratios = np.nanmedian(amplitude_ratios, axis=0)
        amplitude_ratios = amplitude_ratios[np.isfinite(amplitude_ratios)]

        # Regress log(pred) on log(target) over all evaluated bins. A slope below one
        # indicates under-dispersion; a pure scale error has slope approximately one.
        slope_num = s[:, 24] * s[:, 22] - s[:, 20] * s[:, 21]
        slope_den = s[:, 24] * s[:, 23] - s[:, 21] * s[:, 21]
        slopes = np.divide(
            slope_num, slope_den, out=np.full(n_tracks, np.nan), where=slope_den > 1e-12
        )
        finite_slopes = slopes[np.isfinite(slopes)]
        # Keep the historical Pearson metric's NaN-excluding median for reporting,
        # but score undefined (flat) tracks as zero so they cannot disappear from
        # the sweep objective.
        objective_pearson = float(np.median(np.nan_to_num(r_all, nan=0.0)))
        abs_log_ratio = float(abs(np.median(log_ratios))) if log_ratios.size else float("nan")

        # Cross-track specificity per track. The sd ratio is the stretch a post-hoc rescale
        # would need to undo; with r it also gives the regression slope (r x sd ratio).
        cn = s[:, 30]
        r_contrast = _pearson_from_stats(s[:, 25], s[:, 26], s[:, 27], s[:, 28], s[:, 29], cn)
        var_contrast_p = np.maximum(cn * s[:, 28] - s[:, 25] ** 2, 0.0)
        var_contrast_t = cn * s[:, 29] - s[:, 26] ** 2
        has_contrast = var_contrast_t > 1e-12
        contrast_sd_ratios = np.sqrt(
            np.divide(
                var_contrast_p, var_contrast_t, out=np.full(n_tracks, np.nan), where=has_contrast
            )
        )
        fin_contrast = r_contrast[np.isfinite(r_contrast)]
        fin_contrast_sd_ratios = contrast_sd_ratios[np.isfinite(contrast_sd_ratios)]
        contrast_pearson = float(np.median(fin_contrast)) if fin_contrast.size else float("nan")
        contrast_sd_ratio = (
            float(np.median(fin_contrast_sd_ratios))
            if fin_contrast_sd_ratios.size
            else float("nan")
        )
        # Design consumes these per-track channels after replicate averaging. Correlation
        # alone rewards the right ordering even when group differences collapse toward
        # zero; |log(sd ratio)| is symmetric around the desired ratio of one and makes
        # compression and over-dispersion equally costly.
        design_contrast_objective = (
            abs(float(np.log(contrast_sd_ratio))) - contrast_pearson
            if np.isfinite(contrast_pearson)
            and np.isfinite(contrast_sd_ratio)
            and contrast_sd_ratio > 0.0
            else float("nan")
        )

        if per_track_sink is not None:
            per_track_sink(
                {
                    "pearson_bin": r_all,
                    f"pearson_top{topk_n}": r_topk,
                    "pearson_total": r_total,
                    "total_ratio": np.where(total_ratios > 0, total_ratios, np.nan),
                    "amplitude_ratio": track_amplitude_ratios,
                    "dispersion_slope": slopes,
                    "contrast_pearson": r_contrast,
                    "contrast_sd_ratio": contrast_sd_ratios,
                }
            )

        metrics: dict[str, float] = {
            "pearson_bin_median": float(np.median(fin_all)) if fin_all.size else float("nan"),
            f"pearson_top{topk_n}_median": (
                float(np.median(fin_topk)) if fin_topk.size else float("nan")
            ),
            "pearson_total_median": float(np.median(fin_total)) if fin_total.size else float("nan"),
            "total_ratio_median": float(np.median(finite_ratios))
            if finite_ratios.size
            else float("nan"),
            "abs_log_ratio_total_median": abs_log_ratio,
            "calibration_shape_objective": (
                abs_log_ratio - calibration_shape_pearson_weight * objective_pearson
                if np.isfinite(abs_log_ratio)
                else float("nan")
            ),
            "amplitude_ratio_median": float(np.median(amplitude_ratios))
            if amplitude_ratios.size
            else float("nan"),
            "contrast_pearson_median": contrast_pearson,
            "contrast_sd_ratio_median": contrast_sd_ratio,
            "design_contrast_objective": design_contrast_objective,
            # Cell-type differences first, with total calibration and profile shape as guards.
            "contrast_objective": (
                abs_log_ratio
                - contrast_pearson
                - calibration_shape_pearson_weight * objective_pearson
                if np.isfinite(abs_log_ratio) and np.isfinite(contrast_pearson)
                else float("nan")
            ),
            "dispersion_slope_median": float(np.median(finite_slopes))
            if finite_slopes.size
            else float("nan"),
        }
        if group_stats_raw is not None:
            metrics.update(_group_metrics(group_stats_raw))
        return metrics

    return compute_metrics
