"""Tests for regulonado.design.attribution (see docs/attribution.md)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import write_bed as _write_bed
from regulonado.config.models import AttributionConfig
from regulonado.design.attribution import (
    AttributionRecord,
    TrackReadout,
    _smooth,
    call_cores,
    ism_scan,
    write_attributions,
)
from regulonado.design.sequence import DatasetWindowIndex, Seed, resolve_seeds
from regulonado.genomics import Window, one_hot

# Tiny geometry for fast tests, matching tests/test_design.py's convention.
N_PRED_BINS = 8
BIN_SIZE = 10
PRED_BP = N_PRED_BINS * BIN_SIZE  # 80
CONTEXT = 400
MOTIF = slice(200, 240)


def _window() -> Window:
    return Window(
        chrom="chr1",
        pred_start=CONTEXT // 2 - PRED_BP // 2,
        pred_end=CONTEXT // 2 + PRED_BP // 2,
        ctx_start=0,
        ctx_end=CONTEXT,
    )


def _seed(name: str = "c1", start: int = 180, end: int = 280) -> Seed:
    return Seed(
        name=name,
        chrom="chr1",
        cand_start=start,
        cand_end=end,
        window=_window(),
        fold_label="test",
        editable=slice(start, end),
        bins=slice(0, N_PRED_BINS),
    )


class _MotifEnsemble:
    """2 folds x 3 tracks. Only track 1 responds, and only to 'A' content inside MOTIF."""

    track_names = ["t0", "t1", "t2"]

    def __init__(self, n_folds: int = 2, fold_scale: list[float] | None = None) -> None:
        self.n_folds = n_folds
        self.fold_scale = fold_scale or [1.0] * n_folds

    def predict(self, one_hot_batch):
        x = torch.as_tensor(np.asarray(one_hot_batch)).float()
        batch = x.shape[0]
        signal = x[:, 0, MOTIF].sum(dim=1)  # A-count inside the motif
        out = torch.zeros(self.n_folds, batch, 3, N_PRED_BINS)
        for fold, scale in enumerate(self.fold_scale):
            out[fold, :, 1, :] = signal.view(batch, 1) * scale
        return out


def _motif_context(seed_value: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed_value)
    context = one_hot("".join(rng.choice(list("ACGT"), CONTEXT)))
    context[:, MOTIF] = 0
    context[0, MOTIF] = 1  # a pure poly-A block
    return context


# --------------------------------------------------------------------------- #
# 1. TrackReadout                                                             #
# --------------------------------------------------------------------------- #
def test_track_readout_selects_the_named_track_and_bin_window():
    readout = TrackReadout(_MotifEnsemble(), track_indices=[1], bins=slice(0, N_PRED_BINS))
    context = _motif_context()
    scores, per_fold = readout(context[None])
    assert scores.shape == (1,)
    assert per_fold.shape == (2, 1)
    # 40 bp of pure A inside the motif.
    assert scores[0] == pytest.approx(40.0)

    # Track 0 is inert, so it must read exactly zero.
    other = TrackReadout(_MotifEnsemble(), track_indices=[0], bins=slice(0, N_PRED_BINS))
    assert other(context[None])[0][0] == pytest.approx(0.0)


def test_track_readout_bin_reductions_differ_and_topk_is_bounded():
    context = _motif_context()
    kwargs = dict(ensemble=_MotifEnsemble(), track_indices=[1], bins=slice(0, N_PRED_BINS))
    mean = TrackReadout(**kwargs, reduction="mean")(context[None])[0][0]
    maximum = TrackReadout(**kwargs, reduction="max")(context[None])[0][0]
    # The dummy is flat across bins, so all reductions agree; the point is that topk clamps k
    # to the available bins rather than indexing off the end.
    topk = TrackReadout(**kwargs, reduction="topk", topk_bins=1000)(context[None])[0][0]
    assert mean == pytest.approx(maximum) == pytest.approx(topk)


def test_track_readout_fold_reduction_mean_vs_median():
    # Three folds, one wild outlier: the median must ignore it, the mean must not.
    ensemble = _MotifEnsemble(n_folds=3, fold_scale=[1.0, 1.0, 10.0])
    context = _motif_context()
    kwargs = dict(ensemble=ensemble, track_indices=[1], bins=slice(0, N_PRED_BINS))
    mean = TrackReadout(**kwargs, fold_reduction="mean")(context[None])[0][0]
    median = TrackReadout(**kwargs, fold_reduction="median")(context[None])[0][0]
    assert median == pytest.approx(40.0)
    assert mean == pytest.approx(160.0)
    assert mean != pytest.approx(median)


def test_track_readout_averages_multiple_tracks():
    # A group readout with >1 track_indices averages them before any bin/fold reduction —
    # track 1 is the only responsive one (per _MotifEnsemble), track 0 always reads zero, so a
    # [0, 1] group must read exactly half of the single-track [1] readout.
    context = _motif_context()
    single = TrackReadout(_MotifEnsemble(), track_indices=[1], bins=slice(0, N_PRED_BINS))
    grouped = TrackReadout(_MotifEnsemble(), track_indices=[0, 1], bins=slice(0, N_PRED_BINS))
    assert grouped(context[None])[0][0] == pytest.approx(single(context[None])[0][0] / 2)


def test_track_readout_rejects_unknown_reductions():
    readout = TrackReadout(_MotifEnsemble(), [1], slice(0, N_PRED_BINS), reduction="nope")
    with pytest.raises(ValueError, match="Unknown reduction"):
        readout(_motif_context()[None])


# --------------------------------------------------------------------------- #
# 2. ism_scan                                                                 #
# --------------------------------------------------------------------------- #
def test_ism_scan_recovers_the_planted_motif():
    seed, context = _seed(), _motif_context()
    readout = TrackReadout(_MotifEnsemble(), [1], seed.bins)
    result = ism_scan(readout, seed, context, batch_size=32)

    assert result.effect.shape == (4, 100)
    assert result.importance.shape == (100,)
    assert result.per_fold_importance.shape == (2, 100)
    assert result.ref_score == pytest.approx(40.0)

    inside = result.importance[MOTIF.start - 180 : MOTIF.stop - 180]
    outside = np.r_[result.importance[: MOTIF.start - 180], result.importance[MOTIF.stop - 180 :]]
    # Mutating an A inside the motif costs exactly one unit of signal; outside costs nothing.
    assert np.allclose(inside, 1.0)
    assert np.allclose(outside, 0.0)


def test_ism_scan_zeroes_the_reference_row_and_fills_all_alternates():
    seed, context = _seed(), _motif_context()
    result = ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, batch_size=32)
    for column, ref_base in enumerate(result.ref_bases):
        assert result.effect[ref_base, column] == 0.0
        alts = [b for b in range(4) if b != ref_base]
        assert np.isfinite(result.effect[alts, column]).all()


def test_ism_scan_scores_all_four_bases_at_an_n():
    seed = _seed()
    context = _motif_context()
    context[:, 250] = 0  # an N: no clean one-hot, so no identity edit exists
    result = ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, batch_size=32)
    column = 250 - seed.editable.start
    assert result.ref_bases[column] == -1
    assert np.isfinite(result.effect[:, column]).all()  # all four, not three


def test_ism_scan_respects_stride_and_leaves_unscanned_positions_nan():
    seed, context = _seed(), _motif_context()
    result = ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, stride=5)
    assert list(result.positions) == list(range(180, 280, 5))
    scanned = result.positions - seed.editable.start
    assert np.isfinite(result.importance[scanned]).all()
    unscanned = np.setdiff1d(np.arange(100), scanned)
    assert np.isnan(result.importance[unscanned]).all()


def test_ism_scan_restricted_to_explicit_positions_stays_inside_editable():
    seed, context = _seed(), _motif_context()
    result = ism_scan(
        TrackReadout(_MotifEnsemble(), [1], seed.bins),
        seed,
        context,
        positions=[100, 205, 210, 999],  # 100 and 999 fall outside the editable span
    )
    assert list(result.positions) == [205, 210]


def test_ism_scan_without_scannable_positions_raises():
    seed, context = _seed(), _motif_context()
    with pytest.raises(ValueError, match="No positions to scan"):
        ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, positions=[])


def test_ism_scan_does_not_mutate_the_caller_context():
    seed, context = _seed(), _motif_context()
    before = context.copy()
    ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, batch_size=8)
    assert np.array_equal(context, before)


# --------------------------------------------------------------------------- #
# 3. call_cores                                                               #
# --------------------------------------------------------------------------- #
EDITABLE = slice(1000, 1300)
BOUNDS = (500, 2000)


def _profile(start: int, width: int, height: float = 1.0) -> np.ndarray:
    profile = np.zeros(300)
    profile[start : start + width] = height
    return profile


def _bump(center: int, sigma: float, height: float = 1.0, n: int = 300) -> np.ndarray:
    """A Gaussian bump. Real ISM profiles are smooth; a square block is degenerate for a
    quantile threshold, because the threshold lands exactly on the block height."""
    x = np.arange(n)
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def test_call_cores_recovers_a_block_width_after_undoing_smoothing_erosion():
    cores, diagnostics = call_cores(
        _profile(110, 80), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0
    )
    assert diagnostics["core_called"] is True
    assert len(cores) == 1
    # A centred box filter erodes each edge by smooth_bp/2; that is corrected for, so the called
    # span matches the planted block exactly rather than coming back ~smooth_bp narrower.
    assert (cores[0].start, cores[0].end) == (1110, 1190)


def test_call_cores_rejects_a_flat_profile():
    rng = np.random.default_rng(1)
    cores, diagnostics = call_cores(
        rng.normal(0, 0.01, 300), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.5
    )
    assert cores == []
    assert diagnostics["core_called"] is False


def test_call_cores_rejects_a_constant_profile_via_zscore():
    cores, diagnostics = call_cores(
        np.full(300, 0.7), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.5
    )
    assert cores == []
    assert diagnostics["reason"] == "low_zscore"


@pytest.mark.parametrize("start,expected", [(0, 1000), (220, 1300)])
def test_call_cores_keeps_segments_touching_the_candidate_edge(start, expected):
    """scipy's default border_value=0 erodes edge-touching runs; this guards that fix."""
    cores, _ = call_cores(
        _profile(start, 80), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0
    )
    assert cores, "an edge-touching core must survive binary_closing"
    edge = cores[0].start if start == 0 else cores[0].end
    assert edge == expected


def test_call_cores_bridges_a_small_dip_but_not_a_large_one():
    profile = _bump(150, 45)
    profile[147:153] = 0.0  # a 6 bp notch, narrower than merge_gap_bp
    merged, _ = call_cores(
        profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, quantile=0.7,
        smooth_bp=5, merge_gap_bp=20, min_width_bp=20, max_cores=5
    )
    assert len(merged) == 1

    profile = _bump(90, 25) + _bump(210, 25)  # two bumps 120 bp apart
    split, _ = call_cores(
        profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, quantile=0.7,
        smooth_bp=5, merge_gap_bp=20, min_width_bp=20, max_cores=5
    )
    assert len(split) == 2


def test_call_cores_drops_segments_narrower_than_min_width():
    cores, diagnostics = call_cores(
        _profile(140, 10), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0, min_width_bp=100
    )
    assert cores == []
    assert diagnostics["reason"] == "no_segment_meets_min_width"


def test_a_spike_narrower_than_the_smoothing_window_is_reported_at_smoothing_width():
    """min_width_bp is the width of the *emitted* core, measured after erosion correction.

    A 10 bp spike is narrower than the default 25 bp smoothing window, so it is smeared rather
    than eroded and comes back about the width of that window. Lower --smooth-bp to resolve
    features this narrow.
    """
    cores, _ = call_cores(
        _profile(140, 10), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0, min_width_bp=20
    )
    assert len(cores) == 1
    assert 20 <= cores[0].width <= 70


def test_call_cores_ranks_multiple_segments_by_score():
    profile = _bump(80, 25, height=1.0) + _bump(220, 25, height=0.5)
    cores, _ = call_cores(
        profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, quantile=0.7,
        smooth_bp=5, min_width_bp=20, max_cores=2
    )
    assert [core.rank for core in cores] == [0, 1]
    assert cores[0].score > cores[1].score
    assert cores[0].start < cores[1].start  # the taller bump is the earlier one


def test_call_cores_honours_max_cores_per_candidate():
    profile = _bump(80, 25) + _bump(220, 25)
    cores, _ = call_cores(
        profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, quantile=0.7,
        smooth_bp=5, min_width_bp=20, max_cores=1
    )
    assert len(cores) == 1


def test_call_cores_handles_stride_induced_nan_holes():
    profile = np.full(300, np.nan)
    profile[::5] = 0.0
    profile[100:180:5] = 1.0
    cores, diagnostics = call_cores(profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0)
    assert diagnostics["core_called"] is True
    assert cores[0].start < 1180 and cores[0].end > 1100


def test_call_cores_emits_whole_candidate_when_shorter_than_min_width():
    cores, diagnostics = call_cores(
        np.linspace(0, 1, 30), editable=slice(1000, 1030), bounds=BOUNDS, min_width_bp=50
    )
    assert diagnostics["reason"] == "candidate_shorter_than_min_width"
    assert (cores[0].start, cores[0].end) == (1000, 1030)


def test_call_cores_all_nan_profile_reports_no_finite_positions():
    cores, diagnostics = call_cores(np.full(300, np.nan), editable=EDITABLE, bounds=BOUNDS)
    assert cores == []
    assert diagnostics["reason"] == "no_finite_positions"


# --------------------------------------------------------------------------- #
# 4. fix_width (synthesis-ready cores)                                        #
# --------------------------------------------------------------------------- #
def test_fix_width_produces_exactly_n_bp_centred_on_the_anchor():
    cores, _ = call_cores(
        _profile(110, 80), editable=EDITABLE, bounds=BOUNDS, min_zscore=1.0, fix_width=200
    )
    core = cores[0]
    assert core.width == 200
    assert core.fixed_width is True
    assert core.clamped is False
    # The block spans 1110-1190, centre 1150, so a 200 bp window is 1050-1250.
    assert (core.start, core.end) == (1050, 1250)


def test_fix_width_is_clamped_inside_the_predicted_window():
    cores, _ = call_cores(
        _profile(110, 80), editable=EDITABLE, bounds=(1090, 1200), min_zscore=1.0, fix_width=200
    )
    core = cores[0]
    assert core.clamped is True
    assert core.start >= 1090 and core.end <= 1200


def test_peak_anchor_differs_from_centroid_on_a_skewed_profile():
    profile = np.zeros(300)
    profile[100:180] = 0.4
    profile[170:180] = 2.0  # mass is central, but the maximum sits at the right edge
    centroid = call_cores(
        profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, anchor="centroid"
    )[0][0]
    peak = call_cores(profile, editable=EDITABLE, bounds=BOUNDS, min_zscore=0.5, anchor="peak")[0][
        0
    ]
    assert peak.anchor > centroid.anchor


# --------------------------------------------------------------------------- #
# 5. Output + the round trip back into `design`                               #
# --------------------------------------------------------------------------- #
def _record(name: str, start: int, end: int) -> AttributionRecord:
    seed = _seed(name, start, end)
    context = _motif_context()
    result = ism_scan(TrackReadout(_MotifEnsemble(), [1], seed.bins), seed, context, batch_size=32)
    cores, diagnostics = call_cores(
        result.importance,
        editable=seed.editable,
        smooth_bp=11,
        quantile=0.75,
        min_width_bp=20,
        merge_gap_bp=5,
        min_zscore=1.0,
        bounds=(seed.window.pred_start, seed.window.pred_end),
    )
    return AttributionRecord(
        seed=seed,
        ism=result,
        smoothed=_smooth(result.importance, 11),
        cores=cores,
        diagnostics=diagnostics,
    )


def test_write_attributions_emits_every_output(tmp_path):
    records = [_record("c0", 180, 280)]
    write_attributions(
        tmp_path, records, run_info={"status": "ok"}, chrom_sizes={"chr1": CONTEXT}, bigwig=True
    )
    for name in (
        "core_regions.bed",
        "cores.tsv",
        "summary.tsv",
        "attributions.tsv",
        "run.json",
        "attributions.bw",
    ):
        assert (tmp_path / name).exists(), name

    bed = (tmp_path / "core_regions.bed").read_text().strip().split("\t")
    assert bed[0] == "chr1"
    assert bed[3] == "c0_core0"  # provenance survives into the name column
    assert (int(bed[1]), int(bed[2])) == (MOTIF.start, MOTIF.stop)


def test_attributions_tsv_carries_the_full_effect_matrix(tmp_path):
    import pandas as pd

    write_attributions(
        tmp_path, [_record("c0", 180, 280)], run_info={}, chrom_sizes={"chr1": CONTEXT},
        bigwig=False,
    )
    frame = pd.read_csv(tmp_path / "attributions.tsv", sep="\t")
    assert {"delta_A", "delta_C", "delta_G", "delta_T"} <= set(frame.columns)
    assert len(frame) == 100
    # The reference base's own delta is zero by construction.
    for _, row in frame.iterrows():
        assert row[f"delta_{row['ref_base']}"] == 0.0


def test_summary_records_uncalled_candidates_with_a_reason(tmp_path):
    record = _record("c0", 180, 280)
    record.cores = []
    record.diagnostics = dict(record.diagnostics, core_called=False, reason="low_zscore")
    write_attributions(tmp_path, [record], run_info={}, chrom_sizes={"chr1": CONTEXT}, bigwig=False)

    import pandas as pd

    summary = pd.read_csv(tmp_path / "summary.tsv", sep="\t")
    assert summary.loc[0, "core_called"] is np.False_ or not summary.loc[0, "core_called"]
    assert summary.loc[0, "reason"] == "low_zscore"
    # A rejected candidate must not leak into the BED that feeds design.
    assert (tmp_path / "core_regions.bed").read_text().strip() == ""


def test_bigwig_survives_overlapping_candidates(tmp_path):
    import pybigtools

    records = [_record("c0", 180, 280), _record("c1", 185, 285)]
    write_attributions(
        tmp_path, records, run_info={}, chrom_sizes={"chr1": CONTEXT}, bigwig=True
    )
    handle = pybigtools.open(str(tmp_path / "attributions.bw"))
    assert handle.chroms() == {"chr1": CONTEXT}
    assert len(list(handle.records("chr1"))) > 0


def test_called_cores_resolve_back_into_design(tmp_path):
    """The property design depends on: every emitted core matches a dataset window again."""
    intervals = _write_bed(tmp_path / "intervals.bed", [("chr1", 150, 250, "test")])
    index = DatasetWindowIndex.from_bed(
        intervals, context_length=CONTEXT, n_pred_bins=N_PRED_BINS, bin_size=BIN_SIZE
    )
    write_attributions(
        tmp_path, [_record("c0", 180, 280)], run_info={}, chrom_sizes={"chr1": CONTEXT},
        bigwig=False,
    )
    # on_missing="error" makes this assert itself: an unresolvable core raises.
    seeds = resolve_seeds(tmp_path / "core_regions.bed", index, on_missing="error")
    assert len(seeds) == 1
    assert seeds[0].name == "c0_core0"


# --------------------------------------------------------------------------- #
# 6. Config                                                                   #
# --------------------------------------------------------------------------- #
def test_attribution_config_parses_and_defaults():
    config = AttributionConfig(
        candidates="cands.bed", targets=[{"name": "hl60", "track": "atac_hl60"}]
    )
    assert config.shards == 1
    assert config.targets[0].track == "atac_hl60"


def test_attribution_target_group_readout_defaults_group_by_to_source():
    config = AttributionConfig(
        candidates="c.bed", targets=[{"name": "hl60", "target": "K562"}]
    )
    assert config.targets[0].track is None
    assert config.targets[0].group_by == "source"


def test_attribution_target_rejects_track_and_target_together():
    with pytest.raises(ValueError, match="exactly one of"):
        AttributionConfig(
            candidates="c.bed",
            targets=[{"name": "a", "track": "t", "target": "K562"}],
        )


def test_attribution_target_rejects_neither_track_nor_target():
    with pytest.raises(ValueError, match="exactly one of"):
        AttributionConfig(candidates="c.bed", targets=[{"name": "a"}])


def test_attribution_target_rejects_group_by_without_target():
    with pytest.raises(ValueError, match="only applies to a 'target'-based"):
        AttributionConfig(
            candidates="c.bed",
            targets=[{"name": "a", "track": "t", "group_by": "source"}],
        )


def test_attribution_config_rejects_duplicate_target_names():
    with pytest.raises(ValueError, match="must be unique"):
        AttributionConfig(
            candidates="c.bed",
            targets=[{"name": "a", "track": "t"}, {"name": "a", "track": "t"}],
        )


def test_attribution_config_forbids_unknown_keys():
    with pytest.raises(ValueError):
        AttributionConfig(
            candidates="c.bed", made_up=1, targets=[{"name": "a", "track": "t"}]
        )


def test_attribution_config_rejects_topk_bins_without_topk_reduction():
    """topk_bins is silently ignored by TrackReadout unless reduction='topk' (F03/S5): setting
    it alongside another reduction is almost certainly a mistake, so the config rejects it."""
    with pytest.raises(ValueError, match="topk_bins only applies"):
        AttributionConfig(
            candidates="c.bed", topk_bins=5, targets=[{"name": "a", "track": "t"}]
        )


def test_attribution_config_allows_default_topk_bins_with_other_reductions():
    # model_fields_set guards the validator above against false positives: topk_bins left at
    # its default (never explicitly set) must never trigger it.
    config = AttributionConfig(
        candidates="c.bed", bin_reduction="mean", targets=[{"name": "a", "track": "t"}]
    )
    assert config.topk_bins == 10


def test_attribution_config_rejects_stride_with_positions():
    """ism_scan ignores stride entirely once positions is given (attribution.py:ism_scan)."""
    with pytest.raises(ValueError, match="stride is ignored"):
        AttributionConfig(
            candidates="c.bed",
            stride=2,
            positions="motifs.bed",
            targets=[{"name": "a", "track": "t"}],
        )


def test_attribution_config_allows_positions_with_default_stride():
    config = AttributionConfig(
        candidates="c.bed", positions="motifs.bed", targets=[{"name": "a", "track": "t"}]
    )
    assert config.stride == 1
