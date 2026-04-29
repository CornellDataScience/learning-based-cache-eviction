use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use crate::core::policy::CacheKey;
use crate::data::pairwise_samples::{
    PairwiseSample, compare_next_accesses, pairwise_label_from_ordering,
};
use crate::policies::learnedpolicy::PendingDecision;

use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct ReplayBuffer {
    buffer: VecDeque<PairwiseSample>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    pub fn push(&mut self, example: PairwiseSample) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(example);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn iter(&self) -> impl Iterator<Item = &PairwiseSample> {
        self.buffer.iter()
    }

    pub fn drain_all(&mut self) -> Vec<PairwiseSample> {
        self.buffer.drain(..).collect()
    }

    pub fn sample(&self, n: usize) -> Vec<PairwiseSample> {
        let mut rng = thread_rng();

        if self.buffer.len() <= n {
            let mut samples = Vec::with_capacity(self.buffer.len());

            for item in self.buffer.iter() {
                samples.push(item.clone());
            }

            return samples;
        }

        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(&mut rng);

        let mut samples = Vec::with_capacity(n);

        for i in indices.into_iter().take(n) {
            samples.push(self.buffer[i].clone());
        }

        samples
    }
}

/// Convert a matured `PendingDecision` into pairwise training examples.
///
/// Pairs are drawn from `shortlist_candidates`. Labels follow the same
/// convention as the offline pipeline: y=1 means key0 should be evicted
/// first (its next access after `decision_tick` is farther away or never).
///
/// `future_access_ticks` maps each key to the ordered list of access ticks
/// recorded since the decision was made. Only ticks strictly greater than
/// `decision.decision_tick` are considered.
pub fn mature_decision(
    decision: &PendingDecision,
    future_access_ticks: &HashMap<CacheKey, VecDeque<u64>>,
    skip_ties: bool,
    add_swapped_pairs: bool,
) -> Vec<PairwiseSample> {
    let mut examples = Vec::new();
    let candidates = &decision.shortlist_candidates;

    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let key0 = candidates[i];
            let key1 = candidates[j];

            let next0 = first_access_after(key0, decision.decision_tick, future_access_ticks);
            let next1 = first_access_after(key1, decision.decision_tick, future_access_ticks);
            let ordering = compare_next_accesses(next0, next1);

            if ordering == Ordering::Equal && skip_ties {
                continue;
            }

            let y01 = pairwise_label_from_ordering(ordering);
            let request_index = usize::try_from(decision.decision_id).unwrap_or(usize::MAX);

            if let Some(x01) = decision.pair_features(key0, key1) {
                examples.push(PairwiseSample {
                    trace_name: "online_learning".to_string(),
                    cache_size: decision.resident_candidates.len(),
                    request_index,
                    tick: decision.decision_tick,
                    key0,
                    key1,
                    x: x01.to_vec(),
                    y: y01,
                });
            }

            if add_swapped_pairs {
                if let Some(x10) = decision.pair_features(key1, key0) {
                    examples.push(PairwiseSample {
                        trace_name: "online_learning".to_string(),
                        cache_size: decision.resident_candidates.len(),
                        request_index,
                        tick: decision.decision_tick,
                        key0: key1,
                        key1: key0,
                        x: x10.to_vec(),
                        y: 1 - y01,
                    });
                }
            }
        }
    }

    examples
}

fn first_access_after(
    key: CacheKey,
    after_tick: u64,
    future_access_ticks: &HashMap<CacheKey, VecDeque<u64>>,
) -> Option<u64> {
    future_access_ticks
        .get(&key)
        .and_then(|ticks| ticks.iter().find(|&&t| t > after_tick).copied())
}
