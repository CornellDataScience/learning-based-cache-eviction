use crate::core::policy::CacheKey;

#[derive(Debug, Clone, Copy)]
pub struct Request {
    pub key: CacheKey,
}

impl Request {
    pub fn new(key: CacheKey) -> Self {
        Self { key }
    }
}

impl RequestTrace {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    pub fn push(&mut self, request: Request) {
        self.requests.push(request);
    }

    pub fn requests(&self) -> &[Request] {
        &self.requests
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

#[derive(Debug, Default, Clone)]
pub struct RequestTrace {
    requests: Vec<Request>,
}



// Internal cache event log for debugging / analysis
#[derive(Debug, Clone)] 
pub enum CacheEvent {
    Hit { 
        key: CacheKey,
        tick: u64 
    },
    Miss { 
        key: CacheKey,
        tick: u64
    },
    Insert { 
        key: CacheKey, 
        size_bytes: usize, 
        tick: u64 
    },
    Evict { 
        key: CacheKey, 
        size_bytes: usize, 
        tick: u64
    },
}

#[derive(Debug, Default)]
pub struct EventTrace {
    events: Vec<CacheEvent>,
    enabled: bool,
}

impl EventTrace {
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