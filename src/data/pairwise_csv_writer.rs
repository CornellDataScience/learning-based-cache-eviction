use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::data::pairwise_samples::PairwiseSample;

pub struct PairwiseCsvWriter;

impl PairwiseCsvWriter {
    pub fn write_to_path<P: AsRef<Path>>(
        path: P,
        samples: &[PairwiseSample],
        decay_dims: usize,
    ) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        Self::write_to_writer(&mut writer, samples, decay_dims)?;
        writer.flush()?;
        Ok(())
    }

    pub fn write_to_writer<W: Write>(
        writer: &mut W,
        samples: &[PairwiseSample],
        decay_dims: usize,
    ) -> std::io::Result<()> {
        let expected_dim = 10 + decay_dims;

        let base_feature_names = [
            "resident_age_diff",
            "resident_time_since_last_diff",
            "resident_access_count_diff",
            "resident_frequency_diff",
            "global_age_since_first_request_diff",
            "global_time_since_last_request_diff",
            "global_total_request_count_diff",
            "last_interarrival_diff",
            "avg_interarrival_diff",
            "gap_count_diff",
        ];

        write!(writer, "trace_name,cache_size,request_index,tick,key0,key1,y")?;
        for name in base_feature_names {
            write!(writer, ",{}", name)?;
        }
        for i in 0..decay_dims {
            write!(writer, ",decay_{}_diff", i)?;
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