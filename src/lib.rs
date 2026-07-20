//! regulonado_rs - Rust hot path for the Regulonado dataset builder.
//!
//! Two production writers are available:
//!
//! 1. `write_arrow_splits_chrom_pass(...)` (recommended) — processes one
//!    chromosome at a time, decoding the binned signal of all tracks once
//!    into an in-RAM `(n_tracks, n_chrom_bins)` matrix and then slicing
//!    per-sample rows out of it. Output is one Arrow IPC shard per
//!    chromosome. All splits share a single scan, so building train/valid/
//!    test together costs one pass rather than three.
//!
//! 2. `write_arrow_split_from_bigwigs(...)` — sample-batched writer that
//!    reads each sample's interval from every BigWig per batch. Retained
//!    as a fallback / reference path for parity testing.

mod bigwig_io;
mod binning;
mod chromosome_scan_writer;
mod fasta;
mod io_utils;
mod arrow_schema;
mod sample_batch_writer;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(
        sample_batch_writer::write_arrow_split_from_bigwigs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        chromosome_scan_writer::write_arrow_splits_chrom_pass,
        m
    )?)?;
    Ok(())
}
