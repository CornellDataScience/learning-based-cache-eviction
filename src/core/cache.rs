use std::collections::HashMap;

use crate::core::entry::Entry;
use crate::core::mainmemory::MainMemory;
use crate::core::metrics::Metrics;
use crate::core::policy::{CacheKey, Policy};
use crate::core::time::Clock;

pub struct Cache<P: Policy, const MM_SIZE: usize> {
    pub capacity: usize,
    pub store: HashMap<CacheKey, Entry<CacheKey>>,
    pub policy: P,
    pub metrics: Metrics,
    pub clock: Clock,
    pub main_memory: MainMemory<MM_SIZE>,
}

impl<P: Policy, const MM_SIZE: usize> Cache<P, MM_SIZE> {
    pub fn new(capacity: usize, policy: P, main_memory: MainMemory<MM_SIZE>) -> Self {
        Self {
            capacity,
            store: HashMap::with_capacity(capacity),
            policy,
            metrics: Metrics::new(),
            clock: Clock::new(),
            main_memory,
        }
    }

    pub fn access(&mut self, key: CacheKey) {
        let tick = self.clock.tick();

        if let Some(entry) = self.store.get_mut(&key) {
            // Hit
            entry.on_access(tick);
            self.policy.on_hit(key);
            self.metrics.record_hit();
            return;
        }

        // Miss
        self.metrics.record_miss();
        self.policy.on_miss(key);

        if self.capacity == 0 {
            return;
        }

        // If at capacity, ask policy who to evict
        if self.store.len() >= self.capacity {
            if let Some(evict_key) = self.policy.victim() {
                self.store.remove(&evict_key);
                self.policy.remove(evict_key);
                self.metrics.record_eviction();
            }
        }

        // Fetch from main memory and insert into cache
        if let Some(obj) = self.main_memory.fetch(&key) {
            let entry = Entry::new(obj.key, obj.size_in_bytes, tick);
            self.store.insert(key, entry);
            self.policy.insert(key);
        }
    }
}
