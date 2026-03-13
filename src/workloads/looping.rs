use crate::workloads::workload::Workload;

pub struct LoopingWorkload {
    keys: Vec<u64>,
    current_index: usize,
    total_requests: usize,
    generated_requests: usize,
}

impl LoopingWorkload {
    pub fn new(keys: Vec<u64>, total_requests: usize) -> Self {
        Self {
            keys,
            current_index: 0,
            total_requests,
            generated_requests: 0,
        }
    }
}

impl Workload for LoopingWorkload {
    fn next_request(&mut self) -> Option<u64> {
        if self.generated_requests >= self.total_requests {
            return None; // workload is complete
        }
        let key = self.keys[self.current_index];
        self.current_index = (self.current_index + 1) % self.keys.len(); // loop back to start
        self.generated_requests += 1;
        Some(key)
    }

    fn is_complete(&self) -> bool {
        self.generated_requests >= self.total_requests
    }
}