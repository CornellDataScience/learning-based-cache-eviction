use crate::workloads::workload::Workload;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::thread_rng;

// currently has its own random seed parameter, which is a little annoying?
pub struct ZipfWorkload {
    keys: Vec<u64>,
    dist: WeightedIndex<f64>,
    rng: ThreadRng,
    total_requests: usize,
    generated_requests: usize,
}

impl ZipfWorkload {
    pub fn new(keys: Vec<u64>, total_requests: usize) -> Self {
        // assign probs based on (1/rank^s) where s is a skew parameter
        let s = 1.0;
        let probabilities: Vec<f64> = keys.iter().enumerate()
            .map(|(i, _)| {
                1.0 / ((i as f64 + 1.0).powf(s))
            }) // rank starts at 1
            .collect();
        
        
        Self {
            keys,
            // weightedindex automatically normalizes for us
            dist: WeightedIndex::new(&probabilities).expect("Failed to create weighted index"),
            rng: thread_rng(),
            total_requests,
            generated_requests: 0,
        }
    }

    fn sample_key(&mut self) -> u64 {
        // based on the probabilities vector, select one.
        let index = self.dist.sample(&mut self.rng);
        self.keys[index]
    }
}

impl Workload for ZipfWorkload {
    fn next_request(&mut self) -> Option<u64> {
        if self.generated_requests >= self.total_requests {
            return None; // workload is complete
        }
        let key = self.sample_key();
        self.generated_requests += 1;
        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}

