use crate::core::policy::CacheKey;
use crate::workloads::workload::Workload;

pub struct BurstyWorkload {
    num_cycles: usize,
    quiet_length: usize,
    burst_length: usize,
    background_keys: usize, 
    burst_key: CacheKey,
    current_cycle: usize,
    position_in_cycle: usize,
    generated_requests: usize,
    total_requests: usize,
}

impl BurstyWorkload {
    pub fn new(
        num_cycles: usize,
        quiet_length: usize,
        burst_length: usize,
        background_keys: usize,
    ) -> Self {
        assert!(num_cycles > 0, "BurstyWorkload requires at least one cycle");
        assert!(
            background_keys > 0,
            "BurstyWorkload requires at least one background key"
        );

        let total_requests = num_cycles * (quiet_length + burst_length);
        let burst_key = background_keys as CacheKey;
        Self {
            num_cycles,
            quiet_length,
            burst_length,
            background_keys,
            burst_key,
            current_cycle: 0,
            position_in_cycle: 0,
            generated_requests: 0,
            total_requests,
        }
    }

    fn cycle_length(&self) -> usize {
        self.quiet_length + self.burst_length
    }
}

impl Workload for BurstyWorkload {
    fn next_request(&mut self) -> Option<CacheKey> {
        if self.is_complete() {
            return None;
        }

        let key = if self.position_in_cycle < self.quiet_length {
            (self.position_in_cycle % self.background_keys) as CacheKey
        } else {
            self.burst_key
        };

        self.position_in_cycle += 1;
        self.generated_requests += 1;

        if self.position_in_cycle >= self.cycle_length() {
            self.current_cycle += 1;
            self.position_in_cycle = 0;
        }

        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}