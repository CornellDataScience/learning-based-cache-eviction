use std::collections::HashMap;

use crate::core::entry::Entry;
use crate::core::mainmemory::MainMemory;
use crate::core::metrics::Metrics;
use crate::core::policy::{CacheKey, Policy};
use crate::core::time::Clock;
use crate::core::trace::CacheTrace;
use crate::core::trace::CacheEvent;
use crate::core::mainmemory::ENTRY_SIZE;


pub struct Cache<P: Policy, const MM_SIZE: usize> {
    pub capacity: usize,
    pub store: HashMap<CacheKey, Entry<CacheKey, ENTRY_SIZE>>,
    pub policy: P,
    pub metrics: Metrics,
    pub clock: Clock,
    pub main_memory: MainMemory<MM_SIZE>,
    pub cache_trace: CacheTrace,
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
            cache_trace: CacheTrace::new(true), // enabled is always set to true - need to change
            // see trace.rs, i think it doesn't make sense to have trace be either enabled or disabled
        }
    }

    pub fn access(&mut self, key: CacheKey) {
        let tick = self.clock.tick();

        if let Some(entry) = self.store.get_mut(&key) {
            // Hit
            entry.on_access(tick);
            self.policy.on_hit(key);
            self.metrics.record_hit();
            self.cache_trace.record_event(CacheEvent::Hit{key, tick});
            return;
        }

        // Miss
        self.metrics.record_miss();
        self.policy.on_miss(key);
        self.cache_trace.record_event(CacheEvent::Miss{key, tick});

        if self.capacity == 0 {
            return;
        }

        // If at capacity, ask policy who to evict
        if self.store.len() >= self.capacity {
            if let Some(evict_key) = self.policy.victim() {
                if let Some(entry) = self.store.get(&evict_key) {
                    let size_bytes = entry.size_in_bytes;
                    self.cache_trace.record_event(CacheEvent::Evict{key: evict_key, size_bytes, tick});
                }
                self.store.remove(&evict_key);
                self.policy.remove(evict_key);
                self.metrics.record_eviction();
            }
        }

        // Fetch from main memory and insert into cache
        if let Some(entry) = self.main_memory.fetch(&key) {
            let size_bytes = entry.size_in_bytes;
            self.store.insert(key, entry.clone());
            self.policy.insert(key);
            self.cache_trace.record_event(CacheEvent::Insert{key, size_bytes, tick});
        }
    }
}
