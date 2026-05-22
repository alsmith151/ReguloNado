//! regulonado_rs - Rust hot path for the Regulonado dataset builder.
//!
//! The production build path is deliberately split into two phases:
//!
//! 1. `write_arrow_split_from_bigwigs(...)` writes each Arrow split directly
//!    from BigWig + FASTA sources in sample batches. This avoids materializing a
//!    multi-TB dense scratch signal file during full builds.
//!
//! 2. `extract_all_tracks_to_file(...)` plus
//!    `write_arrow_split_from_track_major(...)` remain available for benchmarks
//!    and small builds where a dense track-major scratch file is acceptable.
//!
//! Older entry points such as `extract_bigwig_regions`,
//! `extract_bigwig_to_file`, `extract_all_tracks_to_dir`, and
//! `write_arrow_split_from_sample_major` remain useful for debugging and
//! benchmarking individual stages, but the full dataset builder should use the
//! track-major extraction plus direct Arrow writer.

mod binning;
mod fasta;
mod schema;
mod io_utils;
mod signal_file;
mod bigwig_io;
mod writers;
mod debug;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(debug::extract_bigwig_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(debug::extract_bigwig_regions, m)?)?;
    m.add_function(wrap_pyfunction!(debug::extract_all_tracks_to_dir, m)?)?;
    m.add_function(wrap_pyfunction!(debug::extract_all_tracks_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(writers::write_arrow_split_from_sample_major, m)?)?;
    m.add_function(wrap_pyfunction!(writers::write_arrow_split_from_track_major, m)?)?;
    m.add_function(wrap_pyfunction!(writers::write_arrow_split_from_bigwigs, m)?)?;
    Ok(())
}
