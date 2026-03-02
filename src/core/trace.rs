// may also want to track byte size of hits/misses/evictions in the future
#[derive(Debug)] // allow for auto-formatter printing
pub enum CacheEvent {
    Hit  { key_hash: u64, tick: usize }, // found the key in cache
    Miss { key_hash: u64, tick: usize }, // did not locate key in cache
    Insert { key_hash: u64, size_bytes: usize, tick: usize }, // inserted a new key into cache (may cause eviction if cache is full)
    Evict  { key_hash: u64, size_bytes: usize, tick: usize }, // evicted a key from cache to free up space for new insertions
}

#[derive(Debug)]
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

    pub fn get_events(&self) -> &[CacheEvent] {
        &self.events
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}