use crate::workloads::workload::Workload;

pub struct PhaseWorkload {
    num_phases: u64, // number of distinct access phases
    keys_per_phase: u64, // number of unique keys active in each phase
    requests_per_phase: u64, // number of requests in each phase
    current_phase: u64,
    current_request_in_phase: u64,
    generated_requests: usize,
    total_requests: usize,
}

impl PhaseWorkload {
    pub fn new(num_phases: u64, keys_per_phase: u64, requests_per_phase: u64) -> Self {
        let total_requests = (num_phases * requests_per_phase) as usize;
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
    fn next_request(&mut self) -> Option<u64> {
        if self.is_complete() {
            return None; // workload is complete
        }
        // compute disjoint key for current phase
        // each phase owns keys in range [phase * keys_per_phase, (phase+1) * keys_per_phase)
        let phase_start_key = self.current_phase * self.keys_per_phase;
        let key = phase_start_key + (self.current_request_in_phase % self.keys_per_phase);

        // update counters
        self.current_request_in_phase += 1;
        self.generated_requests += 1;

        // move to next phase if current phase is exhausted
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