use std::collections::HashMap;

use crate::core::policy::CacheKey;
use crate::core::trace::CacheTrace;

#[derive(Debug, Clone)]
pub struct Summary {
    pub num_requests: usize,
    pub num_unique_keys: usize,
    pub most_frequent_key: u64,
    pub most_frequent_key_count: usize,
}

impl Summary {
    pub fn summarize(trace: &CacheTrace) -> Self {
        let requests = trace.get_requests();
        let num_requests = requests.len();

        // Count occurrences of each key
        let mut counts: HashMap<CacheKey, usize> = HashMap::new();
        for req in requests {
            *counts.entry(req).or_insert(0) += 1;
        }

        let num_unique_keys = counts.len();

        // Find the most frequent key
        let (most_frequent_key, most_frequent_key_count) = counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or((0u64, 0));

        Summary {
            num_requests,
            num_unique_keys,
            most_frequent_key,
            most_frequent_key_count,
        }
    }
}