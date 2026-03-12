use crate::core::policy::{Policy, CacheKey}; //bringing policy trait into scope
use indexmap::IndexSet;

pub struct LruPolicy {
    order: IndexSet<CacheKey>,//only care about key ordering
}

impl LruPolicy {
    //returns a new LRU policy
    pub fn new(capacity: usize) -> Self {
        Self { //initialize LRU policy
            order: IndexSet::with_capacity(capacity),
        }
    }
}

impl Policy for LruPolicy { 
    fn on_hit(&mut self, key: CacheKey) {
        // key accessed, moved to the back(more recent)
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