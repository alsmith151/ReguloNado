//! Indexed BAM events. BamNado provides filtering and CIGAR-aware spans;
//! the adapter deliberately preserves regulonado.counts.bam.CountSpec semantics.
use bamnado::{genomic_intervals::IntervalMaker, read_filter::BamReadFilter};
use noodles::{bam, bgzf, core::Region, sam};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
};

type Window = (String, i64, i64);
type Reader = bam::io::Reader<bgzf::io::MultithreadedReader<File>>;

#[derive(Clone, Deserialize)]
pub(crate) struct Spec {
    unit: String,
    tn5_shift: bool,
    min_mapq: i64,
    drop_duplicates: bool,
    proper_pairs_only: bool,
    max_fragment_length: Option<i64>,
    single_end_extension: Option<i64>,
    #[serde(default = "default_convention")]
    insertion_convention: String,
}
fn default_convention() -> String {
    "regulonado".into()
}
impl Spec {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        let spec: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if !["insertions", "fragments"].contains(&spec.unit.as_str()) {
            return Err("invalid count unit".into());
        }
        if !["regulonado", "chrombpnet", "unshifted"].contains(&spec.insertion_convention.as_str())
        {
            return Err("invalid insertion convention".into());
        }
        Ok(spec)
    }
}
#[derive(Default)]
pub(crate) struct Events {
    pub starts: Vec<i64>,
    pub ends: Vec<i64>,
    pub paired: u64,
    pub single: u64,
}

pub(crate) struct BamCounter {
    reader: Reader,
    index: Box<dyn noodles::csi::BinningIndex + Send + Sync>,
    header: sam::Header,
    sizes: ahash::HashMap<usize, u64>,
    spec: Spec,
    filter: BamReadFilter,
}
impl BamCounter {
    pub(crate) fn open(path: &str, spec: Spec, threads: usize) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let workers = std::num::NonZeroUsize::new(threads).ok_or("threads must be positive")?;
        let bgzf = bgzf::io::MultithreadedReader::with_worker_count(workers, file);
        let index: Box<dyn noodles::csi::BinningIndex + Send + Sync> =
            if let Ok(index) = bam::bai::fs::read(format!("{path}.bai")) {
                Box::new(index)
            } else if let Ok(index) =
                bam::bai::fs::read(std::path::Path::new(path).with_extension("bai"))
            {
                Box::new(index)
            } else {
                Box::new(
                    noodles::csi::fs::read(format!("{path}.csi"))
                        .map_err(|e| format!("{path}: BAM index required: {e}"))?,
                )
            };
        let mut reader = bam::io::Reader::from(bgzf);
        let header = reader.read_header().map_err(|e| format!("{path}: {e}"))?;
        let sizes = header
            .reference_sequences()
            .iter()
            .enumerate()
            .map(|(i, (_, r))| (i, r.length().get() as u64))
            .collect();
        let filter = BamReadFilter::default()
            .with_proper_pair(false)
            .with_min_mapq(spec.min_mapq.clamp(0, 255) as u8)
            .with_min_length(0)
            .with_max_length(u32::MAX)
            .with_exclude_duplicates(spec.drop_duplicates)
            .with_exclude_secondary(true)
            .with_exclude_supplementary(true);
        Ok(Self {
            reader,
            index,
            header,
            sizes,
            spec,
            filter,
        })
    }
    pub(crate) fn is_insertions(&self) -> bool {
        self.spec.unit == "insertions"
    }
    pub(crate) fn has_chrom(&self, chrom: &str) -> bool {
        self.header
            .reference_sequences()
            .contains_key(chrom.as_bytes())
    }
    pub(crate) fn events(&mut self, chrom: &str) -> Result<Events, String> {
        let mut out = Events::default();
        if !self.has_chrom(chrom) {
            return Ok(out);
        }
        let region = Region::new(chrom, ..);
        let query = self
            .reader
            .query(&self.header, &self.index, &region)
            .map_err(|e| format!("{chrom}: {e}"))?;
        for result in query.records() {
            let read = result.map_err(|e| format!("{chrom}: {e}"))?;
            let flags = read.flags();
            if u16::from(flags) & (0x4 | 0x100 | 0x200 | 0x800) != 0 || self.spec.min_mapq > 255 {
                continue;
            }
            let paired = flags.is_segmented();
            let tlen = i64::from(read.template_length());
            if paired
                && ((self.spec.proper_pairs_only && !flags.is_properly_segmented())
                    || self
                        .spec
                        .max_fragment_length
                        .is_some_and(|m| tlen.abs() > m))
            {
                continue;
            }
            let reverse = flags.is_reverse_complemented();
            // Read mode avoids BamNado's first-segment fragment selection and clamping.
            let Some((s, e, _)) = IntervalMaker::new(
                read,
                &self.header,
                &self.sizes,
                &self.filter,
                false,
                None,
                None,
            )
            .coords() else {
                continue;
            };
            // htslib/pysam assign one reference base to an alignment whose
            // nonempty CIGAR consumes no reference (e.g. an all-soft-clipped read).
            let (mut start, mut end) = (s as i64 - 1, (e as i64 - 1).max(s as i64));
            if self.spec.unit == "insertions" {
                let shift = self.spec.tn5_shift && self.spec.insertion_convention != "unshifted";
                start = if reverse {
                    end - 1
                        + if shift {
                            if self.spec.insertion_convention == "chrombpnet" {
                                -4
                            } else {
                                -5
                            }
                        } else {
                            0
                        }
                } else {
                    start + if shift { 4 } else { 0 }
                };
                end = start + 1;
            } else if paired {
                if tlen <= 0 {
                    continue;
                }
                end = start + tlen;
            } else if let Some(extension) = self.spec.single_end_extension {
                if reverse {
                    start = end - extension;
                } else {
                    end = start + extension;
                }
            }
            out.starts.push(start);
            out.ends.push(end);
            if paired {
                out.paired += 1;
            } else {
                out.single += 1;
            }
        }
        out.starts.sort_unstable();
        out.ends.sort_unstable();
        Ok(out)
    }
}
impl Events {
    fn overlap(&self, a: i64, b: i64) -> i64 {
        self.starts.partition_point(|&s| s < b) as i64
            - self.ends.partition_point(|&e| e <= a) as i64
    }
    pub(crate) fn measure(&self, windows: &[(i64, i64)], mode: &str) -> Vec<f64> {
        if mode == "events" {
            return windows
                .iter()
                .map(|&(a, b)| self.overlap(a, b) as f64)
                .collect();
        }
        // Integrated depth from independently sorted endpoints: integral(max(x-s,0))
        // minus integral(max(x-e,0)). Prefix sums avoid a chromosome-sized base array.
        let prefix = |v: &[i64]| {
            let mut p = vec![0i128];
            for &x in v {
                p.push(p.last().unwrap() + i128::from(x));
            }
            p
        };
        let sp = prefix(&self.starts);
        let ep = prefix(&self.ends);
        let integral = |x: i64| {
            let s = self.starts.partition_point(|&v| v < x);
            let e = self.ends.partition_point(|&v| v < x);
            i128::from(x) * (s as i128 - e as i128) - sp[s] + ep[e]
        };
        windows
            .iter()
            .map(|&(a, b)| {
                let sum = (integral(b) - integral(a)) as f64;
                if mode == "mean" {
                    sum / (b - a) as f64
                } else {
                    sum
                }
            })
            .collect()
    }
}
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}
fn bytes_i64(py: Python<'_>, v: &[i64]) -> Py<PyBytes> {
    PyBytes::new(
        py,
        &v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unbind()
}

#[pyfunction]
pub(crate) fn bam_events(
    py: Python<'_>,
    path: String,
    chrom: String,
    spec_json: String,
    threads: usize,
) -> PyResult<(Py<PyBytes>, Py<PyBytes>, u64, u64)> {
    let spec = Spec::parse(&spec_json).map_err(PyValueError::new_err)?;
    let events = py
        .allow_threads(|| BamCounter::open(&path, spec, threads)?.events(&chrom))
        .map_err(PyValueError::new_err)?;
    Ok((
        bytes_i64(py, &events.starts),
        bytes_i64(py, &events.ends),
        events.paired,
        events.single,
    ))
}
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn count_bam_windows(
    py: Python<'_>,
    path: String,
    spec_json: String,
    windows: Vec<Window>,
    anchors: Vec<Window>,
    background: Vec<Window>,
    scale_bp: i64,
    threads: usize,
    mode: String,
) -> PyResult<(Py<PyBytes>, u64, u64, f64, f64)> {
    if !["events", "mean", "sum"].contains(&mode.as_str()) {
        return Err(PyValueError::new_err("mode must be events, mean, or sum"));
    }
    if windows
        .iter()
        .chain(&anchors)
        .chain(&background)
        .any(|(_, a, b)| b <= a)
    {
        return Err(PyValueError::new_err("windows must have positive width"));
    }
    let spec = Spec::parse(&spec_json).map_err(PyValueError::new_err)?;
    let (counts, paired, single, high, low) = py
        .allow_threads(|| -> Result<_, String> {
            let mut bam = BamCounter::open(&path, spec, threads)?;
            let absent: BTreeSet<_> = windows
                .iter()
                .filter(|(c, _, _)| !bam.has_chrom(c))
                .map(|(c, _, _)| c)
                .collect();
            if !absent.is_empty() {
                return Err(format!(
                    "{path}: region contigs absent from BAM: {absent:?}"
                ));
            }
            let mut grouped: HashMap<String, Vec<(usize, i64, i64)>> = HashMap::new();
            for (i, (c, a, b)) in windows.iter().enumerate() {
                grouped.entry(c.clone()).or_default().push((i, *a, *b));
            }
            let chroms: BTreeSet<_> = windows
                .iter()
                .chain(&anchors)
                .chain(&background)
                .map(|(c, _, _)| c)
                .collect();
            let mut counts = vec![0.0; windows.len()];
            let (mut paired, mut single) = (0, 0);
            let mut high = Vec::new();
            let mut low = Vec::new();
            for chrom in chroms {
                let events = bam.events(chrom)?;
                paired += events.paired;
                single += events.single;
                if let Some(rows) = grouped.get(chrom) {
                    let coords: Vec<_> = rows.iter().map(|(_, a, b)| (*a, *b)).collect();
                    for ((i, _, _), v) in rows.iter().zip(events.measure(&coords, &mode)) {
                        counts[*i] = v;
                    }
                }
                let width = (scale_bp / 100 * 100).clamp(100, 1000);
                for (rows, scores) in [(&anchors, &mut high), (&background, &mut low)] {
                    for (_, a, _) in rows.iter().filter(|(c, _, _)| c == chrom) {
                        let best = (0..=1000 - width)
                            .step_by(100)
                            .map(|o| events.overlap(a + o, a + o + width))
                            .max()
                            .unwrap_or(0);
                        scores.push(best as f64);
                    }
                }
            }
            Ok((counts, paired, single, median(high), median(low)))
        })
        .map_err(PyValueError::new_err)?;
    let bytes = PyBytes::new(
        py,
        &counts
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unbind();
    Ok((bytes, paired, single, high, low))
}
