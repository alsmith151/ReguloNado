use std::collections::HashMap;

/// A single samtools .fai index record.
///
/// Represents one contig from a FASTA index file, enabling random access to
/// sequences by computing byte offsets directly.
#[derive(Clone, Debug)]
pub(crate) struct FastaIndexRecord {
    /// Length of the sequence in bases.
    pub(crate) len: u64,
    /// Byte offset of the sequence data in the FASTA file.
    pub(crate) offset: u64,
    /// Number of bases per line in the FASTA file (before newline).
    pub(crate) line_bases: u64,
    /// Number of bytes per line in the FASTA file (including newline).
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

/// Read a centered FASTA interval and return sequence tokens of length `context_len`.
///
/// Tokens are A0 C1 G2 T3 (case-insensitive); N, any other ambiguous base, and
/// out-of-contig padding are all 4.
pub(crate) fn read_sequence_tokens(
    fasta: &std::fs::File,
    fai: &HashMap<String, FastaIndexRecord>,
    chrom: &str,
    bed_start: u32,
    bed_end: u32,
    context_len: usize,
) -> Result<Vec<u8>, String> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;

    let mut out = vec![4u8; context_len];
    // A contig present in the BED but absent from the FASTA index used to yield an
    // all-zero one-hot, so a misspelled or mismatched contig name silently produced blank
    // training rows instead of failing. Treat it as an error: a caller that genuinely
    // wants to skip such regions should filter the BED before building.
    let Some(rec) = fai.get(chrom) else {
        return Err(format!(
            "Contig '{chrom}' is not present in the FASTA index. Check that the BED and \
             FASTA use the same contig naming (e.g. 'chr1' vs '1'), or filter the BED to \
             contigs present in the FASTA."
        ));
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
        let token = match base {
            b'A' | b'a' => 0u8,
            b'C' | b'c' => 1u8,
            b'G' | b'g' => 2u8,
            b'T' | b't' => 3u8,
            b'\n' | b'\r' => continue,
            _ => 4u8,
        };
        if seq_idx >= context_len {
            break;
        }
        out[seq_idx] = token;
        seq_idx += 1;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal FASTA + `.fai` pair for one contig and return the open
    /// FASTA handle plus its index, for exercising `read_sequence_tokens`.
    fn make_fasta(
        dir: &std::path::Path,
        name: &str,
        seq: &str,
    ) -> (std::fs::File, HashMap<String, FastaIndexRecord>) {
        let fasta_path = dir.join(format!("{name}.fa"));
        let mut f = std::fs::File::create(&fasta_path).unwrap();
        writeln!(f, ">{name}").unwrap();
        writeln!(f, "{seq}").unwrap();
        drop(f);

        let line_bases = seq.len() as u64;
        let mut fai = HashMap::new();
        fai.insert(
            name.to_string(),
            FastaIndexRecord {
                len: seq.len() as u64,
                offset: (name.len() + 2) as u64, // ">name\n"
                line_bases,
                line_width: line_bases + 1,
            },
        );
        (std::fs::File::open(&fasta_path).unwrap(), fai)
    }

    #[test]
    fn encodes_bases_case_insensitively() {
        let dir = tempfile_dir();
        let (fasta, fai) = make_fasta(&dir, "chr1", "acgtACGT");
        let tokens = read_sequence_tokens(&fasta, &fai, "chr1", 0, 8, 8).unwrap();
        assert_eq!(tokens, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn n_and_other_bases_are_token_four() {
        let dir = tempfile_dir();
        let (fasta, fai) = make_fasta(&dir, "chr1", "ANGTn");
        let tokens = read_sequence_tokens(&fasta, &fai, "chr1", 0, 5, 5).unwrap();
        assert_eq!(tokens, vec![0, 4, 2, 3, 4]);
    }

    #[test]
    fn out_of_contig_padding_is_token_four() {
        let dir = tempfile_dir();
        let (fasta, fai) = make_fasta(&dir, "chr1", "ACGT");
        // Centered window wider than the contig: context extends past both ends.
        let tokens = read_sequence_tokens(&fasta, &fai, "chr1", 0, 4, 10).unwrap();
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens.iter().filter(|&&t| t == 4).count() >= 6, true);
        // The in-bounds bases still decode correctly wherever they land.
        assert!(tokens.contains(&0));
        assert!(tokens.contains(&1));
        assert!(tokens.contains(&2));
        assert!(tokens.contains(&3));
    }

    fn tempfile_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "regulonado_rs_fasta_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
