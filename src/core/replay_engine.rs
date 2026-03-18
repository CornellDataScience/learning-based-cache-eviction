use crate::core::cache::Cache;
use crate::core::trace::CacheTrace;
use crate::core::policy::Policy;
use crate::core::metrics::Metrics;
use std::fmt;

pub struct ReplayResult {
    pub request_count: u64,
    pub eviction_count: u64,
    pub hit_count: u64,
    pub hit_rate: f64,
    pub miss_rate: f64,
}

impl ReplayResult {
    pub fn from_metrics(metrics: &Metrics) -> Self {
        Self {
            request_count: metrics.request_count,
            eviction_count: metrics.eviction_count,
            hit_count: metrics.hit_count,
            hit_rate: metrics.hit_rate(),
            miss_rate: metrics.miss_rate(),
        }
    }
}

impl fmt::Display for ReplayResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Request Count: {}\nHit count: {}\nEviction Count: {}\nHit Rate: {:.3}\nMiss Rate: {:.3}",
            self.request_count,
            self.hit_count,
            self.eviction_count,
            self.hit_rate,
            self.miss_rate
        )
    }
}

pub fn replay_trace<P: Policy, const MM_SIZE: usize>(
    trace: &RequestTrace,
    cache: &mut Cache<P, MM_SIZE>,
) -> ReplayResult {
    for request in trace.requests() {
        cache.access(request.key);
    }

    ReplayResult::from_metrics(&cache.metrics)
}