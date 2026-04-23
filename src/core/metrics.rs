#[derive(Default, Debug)]
pub struct Metrics {
    pub request_count: u64,
    pub hit_count: u64,
    pub eviction_count: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hit_rate(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.request_count as f64
        }
    }

    pub fn miss_rate(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            1.0 - self.hit_rate()
        }
    }

    pub fn record_hit(&mut self) {
        self.request_count += 1;
        self.hit_count += 1;
    }

    pub fn record_miss(&mut self) {
        self.request_count += 1;
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

// Test cases, simple and mostly just for learning how unit testing works in Rust.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let m = Metrics::new();
        assert_eq!(m.request_count, 0);
        assert_eq!(m.hit_count, 0);
        assert_eq!(m.eviction_count, 0);
        assert_eq!(m.hit_rate(), 0.0);
        assert_eq!(m.miss_rate(), 0.0);
    }

    #[test]
    fn test_record_hits_and_misses() {
        let mut m = Metrics::new();
        m.record_hit();
        m.record_hit();
        m.record_miss();

        assert_eq!(m.request_count, 3);
        assert_eq!(m.hit_count, 2);
        assert!((m.hit_rate() - 2.0 / 3.0).abs() < f64::EPSILON);
        assert!((m.miss_rate() - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_eviction() {
        let mut m = Metrics::new();
        m.record_eviction();
        m.record_eviction();
        assert_eq!(m.eviction_count, 2);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let mut m = Metrics::new();
        for _ in 0..5 {
            m.record_hit();
        }
        assert_eq!(m.hit_rate(), 1.0);
        assert_eq!(m.miss_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_all_misses() {
        let mut m = Metrics::new();
        for _ in 0..5 {
            m.record_miss();
        }
        assert_eq!(m.hit_rate(), 0.0);
        assert_eq!(m.miss_rate(), 1.0);
    }

    #[test]
    fn test_reset() {
        let mut m = Metrics::new();
        m.record_hit();
        m.record_miss();
        m.record_eviction();
        m.reset();

        assert_eq!(m.request_count, 0);
        assert_eq!(m.hit_count, 0);
        assert_eq!(m.eviction_count, 0);
    }
}
