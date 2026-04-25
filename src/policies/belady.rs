use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::policy::{CacheKey, Policy};
use crate::core::trace::RequestTrace;

/// Belady's optimal eviction policy.
///
/// On each eviction decision, evicts the cached key whose next future access
/// is furthest away (or never, treated as usize::MAX).
///
/// Requires the full request trace at construction time. Maintains its own
/// `future_positions` map in sync with the trace replay: each call to
/// `on_hit` or `on_miss` pops the current request's index off the front of
/// that key's queue, so that `victim()` always sees only future positions.
pub struct BeladyPolicy {
    /// Keys currently resident in cache.
    cached_keys: HashSet<CacheKey>,
    /// Remaining future positions for every key in the trace.
    /// Invariant: by the time `victim()` is called, the current request's
    /// index has already been popped (done inside `on_hit`/`on_miss`).
    future_positions: HashMap<CacheKey, VecDeque<usize>>,
}

impl BeladyPolicy {
    /// Build a Belady policy from a request trace.
    /// `future_positions` is pre-populated with every occurrence of each key.
    pub fn new(trace: &RequestTrace) -> Self {
        let mut future_positions: HashMap<CacheKey, VecDeque<usize>> = HashMap::new();
        for (i, req) in trace.requests().iter().enumerate() {
            future_positions.entry(req.key).or_default().push_back(i);
        }
        Self {
            cached_keys: HashSet::new(),
            future_positions,
        }
    }

    /// Next future access index for `key`, or `usize::MAX` if it won't be
    /// accessed again. Used by `victim()` to compare candidates.
    fn next_use(&self, key: CacheKey) -> usize {
        self.future_positions
            .get(&key)
            .and_then(|q| q.front().copied())
            .unwrap_or(usize::MAX)
    }
}

impl Policy for BeladyPolicy {
    /// Cache hit: pop the current request index off this key's future queue.
    fn on_hit(&mut self, key: CacheKey) {
        if let Some(q) = self.future_positions.get_mut(&key) {
            q.pop_front();
        }
    }

    /// Cache miss: pop the current request index off this key's future queue.
    /// Must happen before `victim()` is called so that `victim()` sees only
    /// strictly future positions.
    fn on_miss(&mut self, key: CacheKey) {
        if let Some(q) = self.future_positions.get_mut(&key) {
            q.pop_front();
        }
    }

    fn insert(&mut self, key: CacheKey) {
        self.cached_keys.insert(key);
    }

    fn remove(&mut self, key: CacheKey) {
        self.cached_keys.remove(&key);
    }

    /// Return the cached key with the furthest next future use.
    /// Ties are broken arbitrarily (both candidates are equally bad choices).
    fn victim(&mut self) -> Option<CacheKey> {
        self.cached_keys
            .iter()
            .copied()
            .max_by_key(|&key| self.next_use(key))
    }
}
