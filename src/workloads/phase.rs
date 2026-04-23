use crate::core::policy::CacheKey;
use crate::workloads::workload::Workload;

pub struct PhaseWorkload {
    num_phases: usize,
    keys_per_phase: usize,
    requests_per_phase: usize,
    current_phase: usize,
    current_request_in_phase: usize,
    generated_requests: usize,
    total_requests: usize,
}

impl PhaseWorkload {
    pub fn new(num_phases: usize, keys_per_phase: usize, requests_per_phase: usize) -> Self {
        assert!(num_phases > 0, "PhaseWorkload requires at least one phase");
        assert!(
            keys_per_phase > 0,
            "PhaseWorkload requires at least one key per phase"
        );
        assert!(
            requests_per_phase > 0,
            "PhaseWorkload requires at least one request per phase"
        );

        let total_requests = num_phases * requests_per_phase;

        Self {
            num_phases,
            keys_per_phase,
            requests_per_phase,
            current_phase: 0,
            current_request_in_phase: 0,
            generated_requests: 0,
            total_requests,
        }
    }
}

impl Workload for PhaseWorkload {
    fn next_request(&mut self) -> Option<CacheKey> {
        if self.is_complete() {
            return None;
        }

        let phase_start_key = (self.current_phase * self.keys_per_phase) as CacheKey;
        let key =
            phase_start_key + (self.current_request_in_phase % self.keys_per_phase) as CacheKey;

        self.current_request_in_phase += 1;
        self.generated_requests += 1;

        if self.current_request_in_phase >= self.requests_per_phase {
            self.current_phase += 1;
            self.current_request_in_phase = 0;
        }

        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}
