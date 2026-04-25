use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::core::policy::CacheKey;
use crate::core::trace::{Request, RequestTrace};

/// Output of loading a single Wikipedia cache trace file.
pub struct WikiTrace {
    pub trace: RequestTrace,
    /// First-seen response_size (bytes) for each key.
    pub sizes: HashMap<CacheKey, usize>,
}

/// Load a Wikipedia cache trace TSV file.
///
/// Expected columns (tab-separated, one header row):
///   relative_unix | hashed_host_path_query | response_size | time_firstbyte
///
/// `hashed_host_path_query` is a signed 64-bit integer; we reinterpret its
/// bits as u64 for use as CacheKey.  `response_size` is recorded on the first
/// occurrence of each key and ignored on subsequent ones.
pub fn load(path: &Path) -> io::Result<WikiTrace> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(true)
        .from_path(path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut trace = RequestTrace::new();
    let mut sizes: HashMap<CacheKey, usize> = HashMap::new();

    for result in rdr.records() {
        let record = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let raw_key: i64 = record
            .get(1)
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "bad hashed_host_path_query")
            })?;

        let key: CacheKey = raw_key as u64;

        let size: usize = record
            .get(2)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);

        trace.push(Request::new(key));
        sizes.entry(key).or_insert(size);
    }

    Ok(WikiTrace { trace, sizes })
}
