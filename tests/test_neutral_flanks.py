"""Tests for ``design/sequence.py::apply_neutral_flanks`` (batch D).

Covers: identity of ``"genomic"``, no-mutation-of-input across every mode, ``keep`` staying
bit-identical, ``"shuffle"`` preserving per-channel column sums, ``"dinuc-shuffle"`` preserving
both mono- and dinucleotide counts (the test that would catch a wrong Eulerian-path
implementation), determinism under a fixed ``rng`` seed, ``"uniform"`` writing 0.25 everywhere
outside ``keep``, and edge cases (whole-context ``keep``, zero-length flanks, N/all-zero columns).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from regulonado.design.sequence import apply_neutral_flanks
from regulonado.genomics import decode, one_hot

MODES = ["genomic", "shuffle", "dinuc-shuffle", "uniform"]


def _random_context(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGT"), size=length))
    return one_hot(seq)


def _dinuc_counts(seq: str) -> Counter:
    return Counter(seq[i : i + 2] for i in range(len(seq) - 1))


# --------------------------------------------------------------------------- #
# 1. genomic is identity                                                     #
# --------------------------------------------------------------------------- #
def test_genomic_mode_is_identity():
    context = _random_context(200, seed=1)
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode="genomic", rng=rng)
    assert np.array_equal(out, context)
    assert out is not context  # returns a copy, not the same object


# --------------------------------------------------------------------------- #
# 2. input is never mutated                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_input_never_mutated(mode):
    context = _random_context(200, seed=2)
    before = context.copy()
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    apply_neutral_flanks(context, keep, mode=mode, rng=rng)
    assert np.array_equal(context, before)


# --------------------------------------------------------------------------- #
# 3. keep region is bit-identical under every mode                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_keep_region_is_bit_identical(mode):
    context = _random_context(200, seed=3)
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode=mode, rng=rng)
    assert np.array_equal(out[:, keep], context[:, keep].astype(out.dtype))


# --------------------------------------------------------------------------- #
# 4. shuffle preserves per-channel column sums over the flanks              #
# --------------------------------------------------------------------------- #
def test_shuffle_preserves_channel_sums():
    context = _random_context(300, seed=4)
    keep = slice(120, 180)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode="shuffle", rng=rng)

    flank_idx = np.concatenate([np.arange(0, 120), np.arange(180, 300)])
    before_sums = context[:, flank_idx].sum(axis=1)
    after_sums = out[:, flank_idx].sum(axis=1)
    assert np.array_equal(before_sums, after_sums)
    # It really did shuffle something (not a no-op) for a flank this long.
    assert not np.array_equal(context[:, flank_idx], out[:, flank_idx])


# --------------------------------------------------------------------------- #
# 5. dinuc-shuffle preserves mono- and dinucleotide counts over the flanks  #
# --------------------------------------------------------------------------- #
def test_dinuc_shuffle_preserves_mono_and_dinucleotide_counts():
    context = _random_context(400, seed=5)
    keep = slice(150, 250)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode="dinuc-shuffle", rng=rng)

    flank_idx = np.concatenate([np.arange(0, 150), np.arange(250, 400)])
    before_seq = decode(context[:, flank_idx])
    after_seq = decode(out[:, flank_idx])

    assert Counter(before_seq) == Counter(after_seq)
    assert _dinuc_counts(before_seq) == _dinuc_counts(after_seq)
    # It really did shuffle something (not a no-op) for a flank this long.
    assert before_seq != after_seq


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_dinuc_shuffle_preserves_counts_across_many_seeds(seed):
    """Repeat the strict count check across several random contexts/seeds to make a wrong
    Eulerian-path implementation (e.g. one that occasionally drops or duplicates an edge)
    very unlikely to slip through by luck of a single fixture.
    """
    context = _random_context(350, seed=seed)
    keep = slice(100, 200)
    rng = np.random.default_rng(seed)
    out = apply_neutral_flanks(context, keep, mode="dinuc-shuffle", rng=rng)

    flank_idx = np.concatenate([np.arange(0, 100), np.arange(200, 350)])
    before_seq = decode(context[:, flank_idx])
    after_seq = decode(out[:, flank_idx])

    assert Counter(before_seq) == Counter(after_seq)
    assert _dinuc_counts(before_seq) == _dinuc_counts(after_seq)


# --------------------------------------------------------------------------- #
# 6. determinism                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["shuffle", "dinuc-shuffle"])
def test_same_seed_is_deterministic(mode):
    context = _random_context(300, seed=6)
    keep = slice(120, 180)
    out1 = apply_neutral_flanks(context, keep, mode=mode, rng=np.random.default_rng(42))
    out2 = apply_neutral_flanks(context, keep, mode=mode, rng=np.random.default_rng(42))
    assert np.array_equal(out1, out2)


@pytest.mark.parametrize("mode", ["shuffle", "dinuc-shuffle"])
def test_different_seeds_differ(mode):
    context = _random_context(400, seed=7)
    keep = slice(150, 250)
    out1 = apply_neutral_flanks(context, keep, mode=mode, rng=np.random.default_rng(1))
    out2 = apply_neutral_flanks(context, keep, mode=mode, rng=np.random.default_rng(2))
    assert not np.array_equal(out1, out2)


# --------------------------------------------------------------------------- #
# 7. uniform writes 0.25 everywhere outside keep                            #
# --------------------------------------------------------------------------- #
def test_uniform_writes_quarter_everywhere_outside_keep():
    context = _random_context(200, seed=8)
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode="uniform", rng=rng)

    flank_idx = np.concatenate([np.arange(0, 80), np.arange(120, 200)])
    assert np.all(out[:, flank_idx] == 0.25)
    assert np.array_equal(out[:, keep], context[:, keep].astype(out.dtype))


# --------------------------------------------------------------------------- #
# 8. edge cases                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_keep_whole_context_is_unchanged(mode):
    context = _random_context(100, seed=9)
    keep = slice(0, 100)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode=mode, rng=rng)
    assert np.array_equal(out, context.astype(out.dtype))


@pytest.mark.parametrize("mode", MODES)
def test_zero_length_flanks_via_keep_bounds_outside_array(mode):
    """``keep`` extending past the array on both sides behaves the same as keeping everything.

    Note ``slice.indices`` (which normalises ``keep`` against the array length) treats a
    negative ``start`` as counting from the end, per ordinary Python slicing -- so a bound
    guaranteed to clamp to 0 on either side must be more negative/positive than the length.
    """
    context = _random_context(50, seed=10)
    keep = slice(-1000, 1000)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode=mode, rng=rng)
    assert np.array_equal(out, context.astype(out.dtype))


@pytest.mark.parametrize("mode", ["shuffle", "dinuc-shuffle"])
def test_n_columns_in_flank_are_left_in_place(mode):
    """All-zero (N) columns outside ``keep`` are not part of the shuffle pool: they must not
    break the shuffle, and they stay exactly where they were (untouched), per the module's
    documented handling of non-clean one-hot columns.
    """
    context = _random_context(200, seed=11)
    context[:, 10] = 0  # N in the left flank
    context[:, 190] = 0  # N in the right flank
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode=mode, rng=rng)

    assert np.array_equal(out[:, 10], context[:, 10])
    assert np.array_equal(out[:, 190], context[:, 190])
    # Every clean one-hot column outside keep is still a clean one-hot column (no N created).
    flank_idx = np.concatenate([np.arange(0, 80), np.arange(120, 200)])
    flank_out = out[:, flank_idx]
    is_n = np.all(flank_out == 0, axis=0)
    is_one_hot = (flank_out.sum(axis=0) == 1) & np.isin(flank_out, (0, 1)).all(axis=0)
    assert np.all(is_n | is_one_hot)
    assert is_n.sum() == 2


def test_n_columns_outside_keep_do_not_break_uniform():
    context = _random_context(200, seed=12)
    context[:, 10] = 0  # N in the left flank
    keep = slice(80, 120)
    rng = np.random.default_rng(0)
    out = apply_neutral_flanks(context, keep, mode="uniform", rng=rng)
    # uniform overwrites every flank column regardless of cleanliness, including the N column.
    assert np.all(out[:, 10] == 0.25)
