use std::collections::HashMap;

use crate::core::entry::Entry;
use crate::core::metrics::Metrics;
use crate::core::policy::{CacheKey, Policy};
use crate::core::time::Clock;

pub struct MainMemory<const SIZE: usize> {
    pub mem: HashMap<CacheKey, Entry<CacheKey>>,
}

impl<const SIZE: usize> MainMemory<SIZE> {
    pub fn new() -> Self {
        Self {
            mem: HashMap::new()
        }
    }

    pub fn fetch(&self, key: &CacheKey) -> Option<&mut Entry<CacheKey>>{
        self.mem.get(key)
    }
}