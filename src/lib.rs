//! regulonado_rs - Rust hot path for the Regulonado dataset builder.
//!
//! Two production writers are available:
//!
//! 1. `write_arrow_split_chrom_pass(...)` (recommended) — processes one
//!    chromosome at a time, decoding the binned signal of all tracks once
//!    into an in-RAM `(n_tracks, n_chrom_bins)` matrix and then slicing
//!    per-sample rows out of it. Output is one Arrow IPC shard per
//!    chromosome.
//!
//! 2. `write_arrow_split_from_bigwigs(...)` — sample-batched writer that
//!    reads each sample's interval from every BigWig per batch. Retained
//!    as a fallback / reference path for parity testing.
//!
//! The dense track-major scratch path (`extract_all_tracks_to_file` +
//! `write_arrow_split_from_track_major`) and the per-sample debug entry
//! points (`extract_bigwig_regions`, `extract_bigwig_to_file`,
//! `extract_all_tracks_to_dir`, `write_arrow_split_from_sample_major`)
//! remain available for benchmarking individual stages.

mod bigwig_io;
mod binning;
mod chrom_pass;
mod debug;
mod fasta;
mod io_utils;
mod schema;
mod signal_file;
mod writers;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "debug-writers")]
    {
        m.add_function(wrap_pyfunction!(debug::extract_bigwig_to_file, m)?)?;
        m.add_function(wrap_pyfunction!(debug::extract_bigwig_regions, m)?)?;
        m.add_function(wrap_pyfunction!(debug::extract_all_tracks_to_dir, m)?)?;
        m.add_function(wrap_pyfunction!(debug::extract_all_tracks_to_file, m)?)?;
        m.add_function(wrap_pyfunction!(
            writers::write_arrow_split_from_sample_major,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            writers::write_arrow_split_from_track_major,
            m
        )?)?;
    }
    m.add_function(wrap_pyfunction!(
        writers::write_arrow_split_from_bigwigs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        chrom_pass::write_arrow_split_chrom_pass,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        chrom_pass::write_arrow_splits_chrom_pass,
        m
    )?)?;
    Ok(())
}
