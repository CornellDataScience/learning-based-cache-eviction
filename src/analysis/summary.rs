use std::collections::HashMap;

use crate::core::policy::CacheKey;
use crate::core::trace::RequestTrace;

#[derive(Debug, Clone)]
pub struct Summary {
    pub num_requests: usize,
    pub num_unique_keys: usize,
    pub most_frequent_key: Option<CacheKey>,
    pub most_frequent_key_count: usize,
}

impl Summary {
    pub fn summarize(trace: &RequestTrace) -> Self {
        let requests = trace.requests();
        let num_requests = requests.len();

        let mut counts: HashMap<CacheKey, usize> = HashMap::new();
        for req in requests {
            *counts.entry(req.key).or_insert(0) += 1;
        }

        let num_unique_keys = counts.len();

        let (most_frequent_key, most_frequent_key_count) = counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(key, count)| (Some(key), count))
            .unwrap_or((None, 0));

        Summary {
            num_requests,
            num_unique_keys,
            most_frequent_key,
            most_frequent_key_count,
        }
    }
}