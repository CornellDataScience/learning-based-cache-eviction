use std::collections::HashMap;

use crate::core::policy::CacheKey;

pub struct MemoryObject {
    pub key: CacheKey,
    pub size_in_bytes: usize,
    pub bytes: Vec<u8>,
}

impl MemoryObject {
    pub fn new(key: CacheKey, bytes: Vec<u8>) -> Self {
        let size_in_bytes = bytes.len();
        Self {
            key,
            size_in_bytes,
            bytes,
        }
    }
}

pub struct MainMemory<const SIZE: usize> {
    pub mem: HashMap<CacheKey, MemoryObject>,
}

impl<const SIZE: usize> MainMemory<SIZE> {
    pub fn new() -> Self {
        Self {
            mem: HashMap::with_capacity(SIZE),
        }
    }

    pub fn fetch(&self, key: &CacheKey) -> Option<&MemoryObject> {
        self.mem.get(key)
    }

    pub fn insert(&mut self, obj: MemoryObject) {
        self.mem.insert(obj.key, obj);
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.mem.contains_key(key)
    }
}