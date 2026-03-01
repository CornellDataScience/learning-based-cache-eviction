pub struct Metrics {
    pub request_count: usize,
    pub hit_count: usize,
    pub eviction_count: usize,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hit_rate(&self) -> f64 {
        if self.request_count == 0 {
            return 0.0;
        }
        self.hit_count as f64 / self.request_count as f64
    }

    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    pub fn record_hit(&mut self) {
        self.request_count += 1;
        self.hit_count += 1;
    }

    pub fn record_eviction(&mut self) {
        self.eviction_count += 1;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    // track average memory access time (AMAT)? hit time (constant) + miss rate * miss penalty (constant)
    // e.g.
    // pub fn amat(&self, hit_ticks: f64, miss_penalty_ticks: f64) -> f64 {
    //     hit_ticks + (1.0 - self.hit_rate()) * miss_penalty_ticks
    // }
}