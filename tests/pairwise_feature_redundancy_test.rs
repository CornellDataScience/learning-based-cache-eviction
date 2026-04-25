use lbce::core::cache::Cache;
use lbce::core::trace::{Request, RequestTrace};
use lbce::data::memory_builder::MainMemoryBuilder;
use lbce::data::pairwise_samples::{
    PairwiseDatasetConfig,
    PairwiseDatasetGenerator,
};
use lbce::policies::naivelru::LruPolicy;

const MM_SIZE: usize = 64;

fn build_trace(keys: &[u64]) -> RequestTrace {
    let mut trace = RequestTrace::new();
    for &key in keys {
        trace.push(Request::new(key));
    }
    trace
}

fn generate_samples(trace: &RequestTrace) -> Vec<lbce::data::pairwise_samples::PairwiseSample> {
    let mm = MainMemoryBuilder::from_trace::<MM_SIZE>(trace, 1);
    let mut cache = Cache::new(2, LruPolicy::new(2), mm);
    let config = PairwiseDatasetConfig {
        add_swapped_pairs: true,
        skip_ties: true,
        decay_factors: vec![0.5, 0.8, 0.95],
    };

    PairwiseDatasetGenerator::generate("redundancy_test", trace, &mut cache, &config)
}

#[test]
fn resident_and_global_time_since_last_are_identical_for_resident_candidates() {
    let trace = build_trace(&[1, 2, 1, 3, 2, 1, 4, 2, 5, 1]);
    let samples = generate_samples(&trace);

    assert!(!samples.is_empty(), "expected at least one eviction sample");

    for sample in samples {
        assert_eq!(
            sample.x[1],
            sample.x[5],
            "resident_time_since_last_diff should match global_time_since_last_request_diff",
        );
    }
}

#[test]
fn gap_count_and_total_request_count_diffs_are_identical() {
    let trace = build_trace(&[1, 2, 1, 3, 2, 1, 4, 2, 5, 1]);
    let samples = generate_samples(&trace);

    assert!(!samples.is_empty(), "expected at least one eviction sample");

    for sample in samples {
        assert_eq!(
            sample.x[6],
            sample.x[9],
            "gap_count_diff should match global_total_request_count_diff",
        );
    }
}
