use std::collections::HashMap;

use crate::core::policy::CacheKey;
use crate::core::entry::Entry;

//define entry size 
pub const ENTRY_SIZE : usize = 8;

pub struct MainMemory<const SIZE: usize> {
    pub mem: HashMap<CacheKey, Entry<CacheKey, ENTRY_SIZE>>,
}

impl<const SIZE: usize> MainMemory<SIZE> {
    pub fn new() -> Self {
        Self {
            mem: HashMap::with_capacity(SIZE),
        }
    }

    pub fn fetch(&self, key: &CacheKey) -> Option<&Entry<CacheKey, ENTRY_SIZE>> {
        self.mem.get(key)
    }

    pub fn insert(&mut self, obj: Entry<CacheKey, ENTRY_SIZE>) {
        self.mem.insert(obj.key, obj);
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.mem.contains_key(key)
    }
}