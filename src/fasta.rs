use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct FastaIndexRecord {
    pub(crate) len: u64,
    pub(crate) offset: u64,
    pub(crate) line_bases: u64,
    pub(crate) line_width: u64,
}

/// Load a samtools-style `.fai` index for random FASTA access.
///
/// The Rust Arrow writer only needs plain, indexed FASTA. For each sequence it
/// computes byte ranges directly from `.fai`, avoiding Python and avoiding
/// construction of a `GenomeIntervalDataset` during Arrow writing.
pub(crate) fn load_fasta_index(
    fasta_path: &str,
) -> Result<HashMap<String, FastaIndexRecord>, String> {
    let fai_path = format!("{fasta_path}.fai");
    let text = std::fs::read_to_string(&fai_path)
        .map_err(|e| format!("Cannot read FASTA index {fai_path}: {e}"))?;
    let mut out = HashMap::new();
    for (line_no, line) in text.lines().enumerate() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 5 {
            return Err(format!("Malformed FAI line {} in {fai_path}", line_no + 1));
        }
        out.insert(
            parts[0].to_string(),
            FastaIndexRecord {
                len: parts[1]
                    .parse()
                    .map_err(|e| format!("Bad length in {fai_path}: {e}"))?,
                offset: parts[2]
                    .parse()
                    .map_err(|e| format!("Bad offset in {fai_path}: {e}"))?,
                line_bases: parts[3]
                    .parse()
                    .map_err(|e| format!("Bad line_bases in {fai_path}: {e}"))?,
                line_width: parts[4]
                    .parse()
                    .map_err(|e| format!("Bad line_width in {fai_path}: {e}"))?,
            },
        );
    }
    Ok(out)
}

pub(crate) fn fasta_file_offset(rec: &FastaIndexRecord, pos: u64) -> u64 {
    rec.offset + (pos / rec.line_bases) * rec.line_width + (pos % rec.line_bases)
}

/// Read a centered FASTA interval and return one-hot sequence as `(4, context_len)`.
///
/// Rows are A, C, G, T. Out-of-bounds sequence and ambiguous bases remain zero.
/// The vector is laid out row-major so it can be wrapped directly as an Arrow
/// Array2D field.
pub(crate) fn read_one_hot_sequence(
    fasta: &std::fs::File,
    fai: &HashMap<String, FastaIndexRecord>,
    chrom: &str,
    bed_start: u32,
    bed_end: u32,
    context_len: usize,
) -> Result<Vec<i8>, String> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;

    let mut out = vec![0i8; 4 * context_len];
    let Some(rec) = fai.get(chrom) else {
        return Ok(out);
    };

    let center = (bed_start as i64 + bed_end as i64) / 2;
    let seq_start = center - context_len as i64 / 2;
    let read_start = seq_start.max(0) as u64;
    let read_end = (seq_start + context_len as i64).min(rec.len as i64).max(0) as u64;
    if read_start >= read_end {
        return Ok(out);
    }

    let first_offset = fasta_file_offset(rec, read_start);
    let last_offset = fasta_file_offset(rec, read_end - 1) + 1;
    let mut raw = vec![0u8; (last_offset - first_offset) as usize];

    #[cfg(unix)]
    fasta
        .read_exact_at(&mut raw, first_offset)
        .map_err(|e| format!("Cannot read FASTA sequence for {chrom}: {e}"))?;

    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut fh = fasta
            .try_clone()
            .map_err(|e| format!("Cannot clone FASTA handle: {e}"))?;
        fh.seek(SeekFrom::Start(first_offset))
            .map_err(|e| format!("Cannot seek FASTA: {e}"))?;
        fh.read_exact(&mut raw)
            .map_err(|e| format!("Cannot read FASTA sequence for {chrom}: {e}"))?;
    }

    let mut seq_idx = (read_start as i64 - seq_start) as usize;
    for base in raw {
        let channel = match base {
            b'A' | b'a' => Some(0usize),
            b'C' | b'c' => Some(1usize),
            b'G' | b'g' => Some(2usize),
            b'T' | b't' => Some(3usize),
            b'\n' | b'\r' => continue,
            _ => None,
        };
        if seq_idx >= context_len {
            break;
        }
        if let Some(c) = channel {
            out[c * context_len + seq_idx] = 1;
        }
        seq_idx += 1;
    }

    Ok(out)
}
