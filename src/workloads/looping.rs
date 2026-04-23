use crate::core::policy::CacheKey;
use crate::workloads::workload::Workload;

pub struct LoopingWorkload {
    keys: Vec<CacheKey>,
    current_index: usize,
    total_requests: usize,
    generated_requests: usize,
}

impl LoopingWorkload {
    pub fn new(keys: Vec<CacheKey>, total_requests: usize) -> Self {
        assert!(
            !keys.is_empty(),
            "LoopingWorkload requires at least one key"
        );

        Self {
            keys,
            current_index: 0,
            total_requests,
            generated_requests: 0,
        }
    }
}

impl Workload for LoopingWorkload {
    fn next_request(&mut self) -> Option<CacheKey> {
        if self.generated_requests >= self.total_requests {
            return None;
        }

        let key = self.keys[self.current_index];
        self.current_index = (self.current_index + 1) % self.keys.len();
        self.generated_requests += 1;

        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}
