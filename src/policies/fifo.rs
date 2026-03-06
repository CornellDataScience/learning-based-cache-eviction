use std::collections::VecDeque;
use crate::policy::{Policy, CacheKey};

/// Tracks insertion order in a VecDeque.
/// - on_hit  : no-op — FIFO never changes a key's position after admission.
/// - on_miss : no-op — policy state is not affected by a miss.
/// - insert  : enqueue key at the back.
/// - remove  : dequeue from the front.
/// - victim  : peek at the front — oldest admitted key, read-only.
pub struct FifoPolicy {
    order: VecDeque<CacheKey>,
    capacity: usize
}

impl FifoPolicy {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::with_capacity(capacity),
            capacity
        }
    }
}

impl Policy for FifoPolicy {
    fn on_hit(&mut self, _key: CacheKey) {
    }

    fn on_miss(&mut self, _key: CacheKey) {
    }

    fn insert(&mut self, key: CacheKey) {
        self.order.push_back(key);
    }

    fn remove(&mut self, key: CacheKey) {
        // evication path: O(1)
        if self.order.front() == Some(&key) {
            self.order.pop_front();
        // explicit remove: O(n)
        } else if let Some(pos) = self.order.iter().position(|&k| k == key) {
            self.order.remove(pos);
        }
    }

    fn victim(&self) -> Option<CacheKey> {
        self.order.front().copied()
    }
}