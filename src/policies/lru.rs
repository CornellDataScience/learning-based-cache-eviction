use crate::core::policy::{CacheKey, Policy};
use indexmap::IndexSet;

pub struct LruPolicy {
    order: IndexSet<CacheKey>,
}

impl LruPolicy {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: IndexSet::with_capacity(capacity),
        }
    }
}

impl Policy for LruPolicy {
    fn on_hit(&mut self, key: CacheKey) {
        self.order.shift_remove(&key);
        self.order.insert(key);
    }

    fn on_miss(&mut self, _key: CacheKey) {}

    fn insert(&mut self, key: CacheKey) {
        self.order.shift_remove(&key);
        self.order.insert(key);
    }

    fn remove(&mut self, key: CacheKey) {
        self.order.shift_remove(&key);
    }

    fn victim(&mut self) -> Option<CacheKey> {
        self.order.get_index(0).copied()
    }
}
