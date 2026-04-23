use crate::core::policy::CacheKey;
use crate::workloads::workload::Workload;
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

pub struct ZipfWorkload {
    keys: Vec<CacheKey>,
    dist: WeightedIndex<f64>,
    rng: StdRng,
    total_requests: usize,
    generated_requests: usize,
}

impl ZipfWorkload {
    pub fn new(keys: Vec<CacheKey>, total_requests: usize, skew: f64, seed: u64) -> Self {
        assert!(!keys.is_empty(), "ZipfWorkload requires at least one key");
        assert!(skew > 0.0, "Zipf skew parameter must be positive");

        let probabilities: Vec<f64> = keys
            .iter()
            .enumerate()
            .map(|(i, _)| 1.0 / ((i as f64 + 1.0).powf(skew)))
            .collect();

        Self {
            keys,
            dist: WeightedIndex::new(&probabilities).expect("Failed to create weighted index"),
            rng: StdRng::seed_from_u64(seed),
            total_requests,
            generated_requests: 0,
        }
    }

    fn sample_key(&mut self) -> CacheKey {
        let index = self.dist.sample(&mut self.rng);
        self.keys[index]
    }
}

impl Workload for ZipfWorkload {
    fn next_request(&mut self) -> Option<CacheKey> {
        if self.generated_requests >= self.total_requests {
            return None;
        }
        let key = self.sample_key();
        self.generated_requests += 1;
        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}
