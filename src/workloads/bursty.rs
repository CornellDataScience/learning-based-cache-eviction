use crate::workloads::workload::Workload;

pub struct BurstyWorkload {
    num_cycles: u64, // number of quiet+burst cycles
    quiet_length: u64, // requests in a quiet period
    burst_length: u64, // requests in a burst period
    background_keys: u64, // size of key space during quiet periods (background key)
    burst_key: u64,
    current_cycle: u64, // tracks position within the current cycle 
    // (0 to quiet_length is quiet, quiet_length to cycle_length is burst)
    position_in_cycle: u64,
    generated_requests: usize,
    total_requests: usize,
}

impl BurstyWorkload {
    pub fn new(
        num_cycles: u64,
        quiet_length: u64,
        burst_length: u64,
        background_keys: u64,
    ) -> Self {
        let total_requests = (num_cycles * (quiet_length + burst_length)) as usize;
        let burst_key = background_keys;
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

    fn cycle_length(&self) -> u64 {
        self.quiet_length + self.burst_length
    }
}

impl Workload for BurstyWorkload {
    fn next_request(&mut self) -> Option<u64> {
        if self.is_complete() {
            return None; // workload is complete
        }

        let key = if self.position_in_cycle < self.quiet_length {
            // quiet: loop background keys
            self.position_in_cycle % self.background_keys
        } else {
            // burst: select single burst key
            self.burst_key
        };

        // update position within cycle
        self.position_in_cycle += 1;
        self.generated_requests += 1;

        // move to next cycle after current cycle
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