use crate::core::policy::CacheKey;

// may also want to track byte size of hits/misses/evictions in the future
#[derive(Debug, Clone)] // allow for auto-formatter printing
pub enum CacheEvent {
    Hit { 
        key: CacheKey,
        tick: u64 
    }, // found the key in cache
    Miss { 
        key: CacheKey,
        tick: u64
    }, // did not locate key in cache
    Insert { 
        key: CacheKey, 
        size_bytes: usize, 
        tick: u64 
    }, // inserted a new key into cache (may cause eviction if cache is full)
    Evict { 
        key: CacheKey, 
        size_bytes: usize, 
        tick: u64
    }, // evicted a key from cache to free up space for new insertions
}

impl CacheEvent {
    pub fn key(&self) -> CacheKey {
        match self {
            CacheEvent::Hit {key, ..} => *key,
            CacheEvent::Miss {key, ..} => *key,
            CacheEvent::Insert {key, ..} => *key,
            CacheEvent::Evict {key, ..} => *key,
        }
    }
}

#[derive(Debug, Default)]
pub struct CacheTrace {
    // i feel like it doesn't make sense to have a cache trace be either enabled or disabled, 
    // we should just either add it to the cache or not add it depending on whether or not we want a trace
    events: Vec<CacheEvent>,
    enabled: bool, // only enabled during testing/debugging
}

impl CacheTrace {
    pub fn new(enabled: bool) -> Self {
        Self {
            events: Vec::new(),
            enabled,
        }
    }

    pub fn record_event(&mut self, event: CacheEvent) {
        if self.enabled {
            self.events.push(event);
        }
    }

    pub fn events(&self) -> &[CacheEvent] {
        &self.events
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // this returns the actual vector
    pub fn get_requests(&self) -> Vec<CacheKey> {
        let mut res: Vec<CacheKey> = Vec::new();
        for event in &self.events {
            res.push(event.key());
        }
        res
    }
}