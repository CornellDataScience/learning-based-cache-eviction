use crate::core::policy::{CacheKey, Policy};
use crate::policies::lru::LruPolicy;
use std::collections::HashMap;
use crate::core::trace::RequestTrace;

#[derive(Clone)]
pub struct DataEntry {
    pub time_between_accesses: Vec<f64>,
    pub exp_decay_counters: Vec<f64>,
    pub access_cnt: f64,
    pub recent_rank: f64,
    pub first_access_time: f64,
    pub last_access_time: f64,
    pub avg_time_between_accesses: f64,
}

impl DataEntry {
    /// Collect all features: 32 time-between-accesses (zero-padded), 10 exp decay counters,
    /// then access_cnt, recent_rank, first_access_time, last_access_time, avg_time_between_accesses.
    pub fn collect_all_features(&self) -> Vec<f64> {
        const TBA_WINDOW: usize = 32;

        // Last 32 time-between-accesses, zero-padded at the front if fewer than 32
        let tba = &self.time_between_accesses;
        let pad = TBA_WINDOW.saturating_sub(tba.len());
        let mut features = vec![0.0f64; pad];
        let start = tba.len().saturating_sub(TBA_WINDOW);
        features.extend_from_slice(&tba[start..]);

        // All 10 exp decay counters
        features.extend_from_slice(&self.exp_decay_counters);

        // Scalar features
        features.push(self.access_cnt);
        features.push(self.recent_rank);
        features.push(self.first_access_time);
        features.push(self.last_access_time);
        features.push(self.avg_time_between_accesses);

        features
    }
}

#[derive(Clone)]
pub struct CacheState {
    entries: HashMap<CacheKey, DataEntry>,
}

impl CacheState {
    pub fn new() -> Self {
        CacheState {
            entries: HashMap::new(),
        }
    }
}

pub struct Generator;

impl Generator {
    pub fn calc_next_use(trace: RequestTrace) -> Vec<usize> {
        let len = trace.len();
        let mut next_use = vec![usize::MAX; len];
        let mut next_pos = HashMap::new();

        let requests = trace.requests();
        for i in (0..len).rev() {
            let key = requests[i].key;
            if let Some(&next_index) = next_pos.get(&key) {
                next_use[i] = next_index;
            }
            next_pos.insert(key, i);
        }
        next_use
    }

    pub fn make_dataset(rt: RequestTrace) -> Vec<CacheState> {
        let mut dataset = Vec::new();
        let mut state = CacheState::new();
        let mut current_time = 0.0;
        let capacity = rt.len();
        let requests = rt.requests();
        let mut lru = LruPolicy::new(capacity);

        for request in requests {
            let key = request.key;

            if let Some(entry) = state.entries.get_mut(&key) {
                let delta_t = current_time - entry.last_access_time;

                // Shift left: push new time delta onto history
                entry.time_between_accesses.push(delta_t);

                // C_i = 1 + C_i * 2^{-delta_t / 2^{9+i}}
                for i in 0..10 {
                    let exponent = -delta_t / ((1u64 << (9 + i)) as f64);
                    entry.exp_decay_counters[i] = 1.0 + entry.exp_decay_counters[i] * exponent.exp2();
                }

                // avg = (avg * n + delta_t) / (n + 1)
                entry.avg_time_between_accesses =
                    (entry.avg_time_between_accesses * entry.access_cnt + delta_t)
                    / (entry.access_cnt + 1.0);

                entry.access_cnt += 1.0;
                entry.last_access_time = current_time;
                lru.on_hit(key);
            } else {
                // First access: init all counters to zero
                state.entries.insert(
                    key,
                    DataEntry {
                        time_between_accesses: Vec::new(),
                        exp_decay_counters: vec![0.0; 10],
                        access_cnt: 1.0,
                        recent_rank: 0.0,
                        first_access_time: current_time,
                        last_access_time: current_time,
                        avg_time_between_accesses: 0.0,
                    },
                );
                lru.insert(key);
            }

            // Walk LRU list from tail (rank 0, most recent) to head, updating all ranks
            for (k, rank) in lru.ranks() {
                if let Some(entry) = state.entries.get_mut(&k) {
                    entry.recent_rank = rank as f64;
                }
            }

            current_time += 1.0;
            dataset.push(state.clone());
        }

        dataset
    }
}
