use std::path::Path;

use crate::core::cache::Cache;
use crate::core::policy::Policy;
use crate::core::trace::RequestTrace;

use crate::data::memory_builder::MainMemoryBuilder;
use crate::data::pairwise_csv_writer::PairwiseCsvWriter;
use crate::data::pairwise_samples::{PairwiseDatasetConfig, PairwiseDatasetGenerator};

pub fn generate_pairwise_csv<POL: Policy, const MM_SIZE: usize, PATH: AsRef<Path>>(
    trace_name: &str,
    trace: &RequestTrace,
    cache_capacity: usize,
    policy: POL,
    config: &PairwiseDatasetConfig,
    output_path: PATH,
    default_object_size: usize,
) -> std::io::Result<()> {
    let main_memory = MainMemoryBuilder::from_trace::<MM_SIZE>(trace, default_object_size);

    let mut cache = Cache::new(cache_capacity, policy, main_memory);
    let samples = PairwiseDatasetGenerator::generate(trace_name, trace, &mut cache, config);

    PairwiseCsvWriter::write_to_path(output_path, &samples, config.decay_factors.len())
}
