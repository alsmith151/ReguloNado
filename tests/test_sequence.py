"""``regulonado.sequence``: genome access, one-hot encoding and window padding."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyfastx")

from regulonado.sequence import (  # noqa: E402
    fetch_sequence,
    fetch_window,
    one_hot_encode,
    open_genome,
)


@pytest.fixture
def fasta_path(tmp_path):
    """A tiny two-contig FASTA: chr1 (20 bp), chr2 (10 bp)."""
    path = tmp_path / "genome.fa"
    path.write_text(">chr1\nACGTACGTACGTacgtACGT\n>chr2\nNNNNACGTNN\n")
    return path


def test_fetch_sequence_is_0_based_half_open_and_upper_cased(fasta_path):
    genome = open_genome(fasta_path)
    assert fetch_sequence(genome, "chr1", 0, 4) == "ACGT"
    assert fetch_sequence(genome, "chr1", 4, 8) == "ACGT"
    # Soft-masked bases upper-cased.
    assert fetch_sequence(genome, "chr1", 12, 16) == "ACGT"


def test_fetch_sequence_rejects_empty_or_negative_interval(fasta_path):
    genome = open_genome(fasta_path)
    with pytest.raises(ValueError):
        fetch_sequence(genome, "chr1", 5, 5)
    with pytest.raises(ValueError):
        fetch_sequence(genome, "chr1", -1, 5)


def test_open_genome_reopens_after_a_pid_change(fasta_path):
    """The lazy handle re-opens when the cached pid no longer matches (fork simulation)."""
    genome = open_genome(fasta_path)
    first = genome.fasta
    assert genome.contig_length("chr1") == 20

    # Simulate having been forked into a new process: the cached fasta/pid
    # must be dropped and a fresh handle opened on next access.
    genome._pid = os.getpid() + 1
    second = genome.fasta
    assert first is not second
    assert fetch_sequence(genome, "chr1", 0, 4) == "ACGT"


def test_open_genome_state_drops_on_pickle(fasta_path):
    """``__getstate__`` clears the cached handle, matching the dataset pid pattern."""
    import pickle

    genome = open_genome(fasta_path)
    _ = genome.fasta  # populate the cache
    restored = pickle.loads(pickle.dumps(genome))
    assert restored._fasta is None
    assert restored._pid is None
    assert fetch_sequence(restored, "chr1", 0, 4) == "ACGT"


def test_one_hot_encode_channel_order_and_n_handling():
    encoded = one_hot_encode("ACGTN")
    assert encoded.shape == (4, 5)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[:, 0], [1, 0, 0, 0])  # A
    np.testing.assert_array_equal(encoded[:, 1], [0, 1, 0, 0])  # C
    np.testing.assert_array_equal(encoded[:, 2], [0, 0, 1, 0])  # G
    np.testing.assert_array_equal(encoded[:, 3], [0, 0, 0, 1])  # T
    np.testing.assert_array_equal(encoded[:, 4], [0, 0, 0, 0])  # N: all-zero


def test_one_hot_encode_is_case_insensitive():
    upper = one_hot_encode("ACGT")
    lower = one_hot_encode("acgt")
    np.testing.assert_array_equal(upper, lower)


def test_fetch_window_within_bounds_matches_one_hot_encode(fasta_path):
    genome = open_genome(fasta_path)
    window = fetch_window(genome, "chr1", 0, 8)
    expected = one_hot_encode(fetch_sequence(genome, "chr1", 0, 8))
    np.testing.assert_array_equal(window, expected)


def test_fetch_window_pads_negative_start_with_zero_columns(fasta_path):
    genome = open_genome(fasta_path)
    window = fetch_window(genome, "chr1", -3, 4)
    assert window.shape == (4, 7)
    # First 3 columns (positions -3, -2, -1) are all-zero padding.
    np.testing.assert_array_equal(window[:, :3], np.zeros((4, 3)))
    np.testing.assert_array_equal(window[:, 3:], one_hot_encode("ACGT"))


def test_fetch_window_pads_past_contig_end_with_zero_columns(fasta_path):
    genome = open_genome(fasta_path)
    length = genome.contig_length("chr2")
    window = fetch_window(genome, "chr2", length - 4, length + 3)
    assert window.shape == (4, 7)
    expected = one_hot_encode(fetch_sequence(genome, "chr2", length - 4, length))
    np.testing.assert_array_equal(window[:, :4], expected)
    np.testing.assert_array_equal(window[:, 4:], np.zeros((4, 3)))


def test_fetch_window_pads_both_sides_when_window_exceeds_the_contig(fasta_path):
    genome = open_genome(fasta_path)
    length = genome.contig_length("chr2")
    window = fetch_window(genome, "chr2", -2, length + 2)
    assert window.shape == (4, length + 4)
    np.testing.assert_array_equal(window[:, :2], np.zeros((4, 2)))
    np.testing.assert_array_equal(window[:, -2:], np.zeros((4, 2)))
    expected = one_hot_encode(fetch_sequence(genome, "chr2", 0, length))
    np.testing.assert_array_equal(window[:, 2:-2], expected)


def test_fetch_window_rejects_empty_interval(fasta_path):
    genome = open_genome(fasta_path)
    with pytest.raises(ValueError):
        fetch_window(genome, "chr1", 5, 5)
