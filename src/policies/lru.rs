use crate::policy::{Policy, CacheKey}; //bringing policy trait into scope

use indexmap::IndexSet;

pub struct LruPolicy {
    order: IndexSet<CacheKey>,//only care about key ordering
    capacity: usize
}

impl LruPolicy {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: IndexSet::with_capacity(capacity),
            capacity
        }
    }
}

impl Policy for LruPolicy { 
    fn on_hit(&mut self, key: CacheKey){
        // key accessed, moved to the back(more recent)
        self.order.shift_remove(&key);
        self.order.insert(key);
    }

    fn on_miss(&mut self, key: CacheKey) -> Option<CacheKey> {
        // check if we need to remove an element from cache
        if(self.order.len() >= self.capacity){
            // yes, front of list
            self.order.get_index(0)
        }else{
            // still has room
            None
        }
    }

    fn insert(&mut self, key: CacheKey){
        self.order.insert(key);
    }

    fn remove(&mut self, key: CacheKey){
        self.order.shift_remove(&key);
    }
}