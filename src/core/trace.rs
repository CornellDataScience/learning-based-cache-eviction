// may also want to track byte size of hits/misses/evictions in the future
#[derive(Debug, Clone)] // allow for auto-formatter printing
pub enum CacheEvent {
    Hit { 
        key: u64,
        tick: u64 
    }, // found the key in cache
    Miss { 
        key: u64,
        tick: u64
    }, // did not locate key in cache
    Insert { 
        key: u64, 
        size_bytes: usize, 
        tick: u64 
    }, // inserted a new key into cache (may cause eviction if cache is full)
    Evict { 
        key: u64, 
        size_bytes: usize, 
        tick: u64
    }, // evicted a key from cache to free up space for new insertions
}

#[derive(Debug, Default)]
pub struct CacheTrace {
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
}