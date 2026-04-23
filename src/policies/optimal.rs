use std::collections::{HashMap, VecDeque};

use crate::core::policy::{CacheKey, Policy};
use crate::core::trace::RequestTrace;

pub struct OptimalPolicy {
    future_positions: HashMap<CacheKey, VecDeque<usize>>,
    residents: Vec<CacheKey>,
    request_index: usize,
}

impl OptimalPolicy {
    pub fn from_trace(trace: &RequestTrace, capacity: usize) -> Self {
        let mut future_positions: HashMap<CacheKey, VecDeque<usize>> = HashMap::new();
        for (index, request) in trace.requests().iter().enumerate() {
            future_positions
                .entry(request.key)
                .or_default()
                .push_back(index);
        }

        Self {
            future_positions,
            residents: Vec::with_capacity(capacity),
            request_index: 0,
        }
    }

    fn advance_request(&mut self, key: CacheKey) {
        if let Some(positions) = self.future_positions.get_mut(&key)
            && positions.front() == Some(&self.request_index)
        {
            positions.pop_front();
        }
        self.request_index = self.request_index.saturating_add(1);
    }
}

impl Policy for OptimalPolicy {
    fn on_hit(&mut self, key: CacheKey) {
        self.advance_request(key);
    }

    fn on_miss(&mut self, key: CacheKey) {
        self.advance_request(key);
    }

    fn insert(&mut self, key: CacheKey) {
        if !self.residents.contains(&key) {
            self.residents.push(key);
        }
    }

    fn remove(&mut self, key: CacheKey) {
        if let Some(position) = self.residents.iter().position(|resident| *resident == key) {
            self.residents.swap_remove(position);
        }
    }

    fn victim(&mut self) -> Option<CacheKey> {
        self.residents.iter().copied().max_by_key(|key| {
            self.future_positions
                .get(key)
                .and_then(|positions| positions.front().copied())
                .unwrap_or(usize::MAX)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trace::{Request, RequestTrace};

    #[test]
    fn evicts_item_with_farthest_next_use() {
        let mut trace = RequestTrace::new();
        for key in [0, 1, 2, 0, 1, 2] {
            trace.push(Request::new(key));
        }

        let mut policy = OptimalPolicy::from_trace(&trace, 2);
        policy.on_miss(0);
        policy.insert(0);
        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);

        assert_eq!(policy.victim(), Some(1));
    }
}
