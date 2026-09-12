"""Tests for regulonado.genomics: window geometry and one-hot sequence helpers.

``window_for_interval`` is the single implementation of a rule that used to be written out
four times (dataset/build.py, inference.py, design/sequence.py x2) — see docs/audit-2026-09-11.xml
findings R6/S6. These tests pin the exact arithmetic (including floor-division rounding and the
lack of chromosome-boundary clamping) so a future edit to this one function cannot silently
change what the dataset builder, inference and design modules agree the model saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from regulonado.genomics import (
    Window,
    decode,
    drop_overlaps,
    one_hot,
    reverse_complement,
    window_for_interval,
)


# --------------------------------------------------------------------------- #
# window_for_interval                                                        #
# --------------------------------------------------------------------------- #
def test_window_for_interval_even_widths_exact():
    # center = (480 + 520) // 2 = 500; pred_bp = 4 * 10 = 40.
    window = window_for_interval(
        "chr1", 480, 520, context_length=100, n_pred_bins=4, bin_size=10
    )
    assert window == Window(
        chrom="chr1", pred_start=480, pred_end=520, ctx_start=450, ctx_end=550
    )


def test_window_for_interval_odd_pred_bp_and_context_floor_rounds_down():
    # Odd interval sum (0 + 1 -> center 0 via floor division) and an odd pred_bp/context_length,
    # so both // 2 divisions round toward negative infinity for a negative numerator.
    window = window_for_interval(
        "chrX", 0, 1, context_length=101, n_pred_bins=3, bin_size=7
    )
    # center = (0 + 1) // 2 = 0; pred_bp = 21; pred_start = 0 - 21 // 2 = 0 - 10 = -10.
    # ctx_start = 0 - 101 // 2 = 0 - 50 = -50.
    assert window == Window(
        chrom="chrX", pred_start=-10, pred_end=11, ctx_start=-50, ctx_end=51
    )


def test_window_for_interval_odd_pred_bp_even_context():
    # center = (100 + 200) // 2 = 150; pred_bp = 5 * 3 = 15 (odd); context_length = 50 (even).
    window = window_for_interval(
        "chr2", 100, 200, context_length=50, n_pred_bins=5, bin_size=3
    )
    assert window == Window(
        chrom="chr2", pred_start=143, pred_end=158, ctx_start=125, ctx_end=175
    )


def test_window_for_interval_not_clamped_near_chrom_start():
    # A candidate near position 0 produces a negative ctx_start/pred_start; window_for_interval
    # does not clamp -- callers (e.g. one_hot_context) are responsible for that.
    window = window_for_interval("chr1", 0, 0, context_length=20, n_pred_bins=2, bin_size=2)
    assert window.pred_start < 0
    assert window.ctx_start < 0


# --------------------------------------------------------------------------- #
# one_hot / decode / reverse_complement                                      #
# --------------------------------------------------------------------------- #
def test_one_hot_decode_round_trip():
    seq = "ACGT"
    assert decode(one_hot(seq)) == seq


def test_one_hot_shape_and_values():
    encoded = one_hot("AC")
    assert encoded.shape == (4, 2)
    assert encoded.dtype == np.int8
    # A -> row 0, C -> row 1.
    assert encoded[:, 0].tolist() == [1, 0, 0, 0]
    assert encoded[:, 1].tolist() == [0, 1, 0, 0]


def test_one_hot_unknown_base_is_all_zero_column():
    encoded = one_hot("ACNGT")
    assert encoded[:, 2].sum() == 0
    assert decode(encoded) == "ACNGT"


def test_one_hot_lowercase_matches_uppercase():
    assert np.array_equal(one_hot("acgt"), one_hot("ACGT"))


def test_reverse_complement_pinned_value():
    # complement(AACGT) = TTGCA; reverse -> ACGTT.
    assert decode(reverse_complement(one_hot("AACGT"))) == "ACGTT"


def test_reverse_complement_is_involution():
    seq = "ACGGTTCA"
    encoded = one_hot(seq)
    assert np.array_equal(reverse_complement(reverse_complement(encoded)), encoded)


def test_reverse_complement_preserves_unknown_base_as_n():
    encoded = one_hot("ACNT")
    rc = reverse_complement(encoded)
    assert decode(rc) == "ANGT"


# --------------------------------------------------------------------------- #
# drop_overlaps                                                              #
# --------------------------------------------------------------------------- #
def test_drop_overlaps_removes_overlapping_interval_within_chrom():
    frame = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 5], "end": [10, 15], "value": [1.0, 2.0]}
    )
    kept, n_dropped = drop_overlaps(frame, ["chr1"])
    assert n_dropped == 1
    assert kept[["chrom", "start", "end", "value"]].values.tolist() == [["chr1", 0, 10, 1.0]]


def test_drop_overlaps_touching_intervals_are_not_overlapping():
    # [0,10) and [10,20) share no position (half-open) — both must be kept.
    frame = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20], "value": [1.0, 2.0]}
    )
    kept, n_dropped = drop_overlaps(frame, ["chr1"])
    assert n_dropped == 0
    assert kept[["start", "end"]].values.tolist() == [[0, 10], [10, 20]]


def test_drop_overlaps_does_not_compare_across_chrom_boundary():
    # chr1's interval ends at 100; chr2's starts at 5 — would look "overlapping" if the
    # running max leaked across the chromosome boundary.
    frame = pd.DataFrame(
        {"chrom": ["chr1", "chr2"], "start": [0, 5], "end": [100, 15], "value": [1.0, 2.0]}
    )
    kept, n_dropped = drop_overlaps(frame, ["chr1", "chr2"])
    assert n_dropped == 0
    assert len(kept) == 2


def test_drop_overlaps_sorts_by_chrom_order_argument_not_input_row_order():
    frame = pd.DataFrame(
        {"chrom": ["chr2", "chr1"], "start": [0, 0], "end": [10, 10], "value": [1.0, 2.0]}
    )
    # chrom_order need not match the frame's row order (a chrom_sizes dict, in file order).
    kept, n_dropped = drop_overlaps(frame, ["chr1", "chr2"])
    assert n_dropped == 0
    assert kept["chrom"].tolist() == ["chr1", "chr2"]


def test_drop_overlaps_running_max_end_survives_a_dropped_row():
    # Row 1 ([10,20)) is nested inside row 0 ([0,100)) and is dropped. Row 2 ([50,60)) must
    # still be compared against row 0's end (100) — the widest end seen so far, a cummax — not
    # against row 1's smaller end (20), which is what a naive "compare to previous row" would
    # do once row 1 had been dropped.
    frame = pd.DataFrame(
        {
            "chrom": ["chr1"] * 3,
            "start": [0, 10, 50],
            "end": [100, 20, 60],
            "value": [1.0, 2.0, 3.0],
        }
    )
    kept, n_dropped = drop_overlaps(frame, ["chr1"])
    assert n_dropped == 2
    assert kept[["start", "end"]].values.tolist() == [[0, 100]]
