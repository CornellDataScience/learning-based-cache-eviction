use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::data::pairwise_samples::{PAIRWISE_FEATURE_NAMES, PairwiseSample};

pub struct PairwiseCsvWriter;

impl PairwiseCsvWriter {
    pub fn write_to_path<P: AsRef<Path>>(
        path: P,
        samples: &[PairwiseSample],
        _decay_dims: usize,
    ) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        Self::write_to_writer(&mut writer, samples, 0)?;
        writer.flush()?;
        Ok(())
    }

    pub fn write_to_writer<W: Write>(
        writer: &mut W,
        samples: &[PairwiseSample],
        _decay_dims: usize,
    ) -> std::io::Result<()> {
        let expected_dim = PAIRWISE_FEATURE_NAMES.len();

        write!(
            writer,
            "trace_name,cache_size,request_index,tick,key0,key1,y"
        )?;
        for name in PAIRWISE_FEATURE_NAMES {
            write!(writer, ",{}", name)?;
        }
        writeln!(writer)?;

        if samples.is_empty() {
            return Ok(());
        }

        for sample in samples {
            if sample.x.len() != expected_dim {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "inconsistent feature dim: expected {}, got {}",
                        expected_dim,
                        sample.x.len()
                    ),
                ));
            }

            write!(
                writer,
                "{},{},{},{},{},{},{}",
                sample.trace_name,
                sample.cache_size,
                sample.request_index,
                sample.tick,
                sample.key0,
                sample.key1,
                sample.y
            )?;

            for val in &sample.x {
                write!(writer, ",{}", val)?;
            }

            writeln!(writer)?;
        }

        Ok(())
    }
}
