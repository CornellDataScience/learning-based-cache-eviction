use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use crate::core::cache::Cache;
use crate::core::entry::Entry;
use crate::core::policy::{CacheKey, Policy};
use crate::core::trace::RequestTrace;

/// One pairwise ranking sample.
///
/// Convention:
/// - x = phi(key0) - phi(key1)
/// - y = 1 if key0 should be evicted before key1
/// - y = 0 otherwise
#[derive(Clone, Debug)]
pub struct PairwiseSample {
    pub trace_name: String,
    pub cache_size: usize,
    pub request_index: usize,
    pub tick: u64,
    pub key0: CacheKey,
    pub key1: CacheKey,
    pub x: Vec<f32>,
    pub y: u8,
}

/// Config for pairwise dataset generation.
#[derive(Clone, Debug)]
pub struct PairwiseDatasetConfig {
    pub add_swapped_pairs: bool,
    pub skip_ties: bool,
    pub decay_factors: Vec<f32>,
}

impl Default for PairwiseDatasetConfig {
    fn default() -> Self {
        Self {
            add_swapped_pairs: true,
            skip_ties: true,
            decay_factors: vec![0.5, 0.8, 0.95],
        }
    }
}

#[derive(Clone, Debug)]
struct FeatureState {
    first_request_tick: u64,
    last_request_tick: u64,
    total_request_count: u64,

    last_interarrival: f32,
    avg_interarrival: f32,
    gap_count: u64,

    decay_counters: Vec<f32>,
}

impl FeatureState {
    fn new(now_tick: u64, decay_dims: usize) -> Self {
        Self {
            first_request_tick: now_tick,
            last_request_tick: now_tick,
            total_request_count: 1,
            last_interarrival: 0.0,
            avg_interarrival: 0.0,
            gap_count: 0,
            decay_counters: vec![1.0; decay_dims],
        }
    }

    fn update_on_request(&mut self, now_tick: u64, decay_factors: &[f32]) {
        let delta = now_tick.saturating_sub(self.last_request_tick) as f32;

        self.last_interarrival = delta;
        self.avg_interarrival = if self.gap_count == 0 {
            delta
        } else {
            (self.avg_interarrival * self.gap_count as f32 + delta) / (self.gap_count as f32 + 1.0)
        };
        self.gap_count += 1;

        for (counter, &alpha) in self.decay_counters.iter_mut().zip(decay_factors.iter()) {
            *counter = (*counter) * alpha.powf(delta) + 1.0;
        }

        self.total_request_count = self.total_request_count.saturating_add(1);
        self.last_request_tick = now_tick;
    }
}

pub struct PairwiseDatasetGenerator;

impl PairwiseDatasetGenerator {
    pub fn generate<P: Policy, const MM_SIZE: usize>(
        trace_name: &str,
        trace: &RequestTrace,
        cache: &mut Cache<P, MM_SIZE>,
        config: &PairwiseDatasetConfig,
    ) -> Vec<PairwiseSample> {
        assert!(
            !config.decay_factors.is_empty(),
            "decay_factors must not be empty"
        );
        assert!(
            cache.capacity > 0,
            "cache.capacity must be > 0 to generate eviction data"
        );
        assert!(
            cache.store.is_empty(),
            "dataset generation expects an empty cache.store at start"
        );
        assert!(
            cache.clock.get_tick() == 0,
            "dataset generation expects clock tick 0 at start"
        );
        assert!(
            cache.metrics.request_count == 0,
            "dataset generation expects fresh metrics at start"
        );

        let mut dataset = Vec::new();
        let mut feature_state: HashMap<CacheKey, FeatureState> = HashMap::new();
        let mut future_positions = Self::build_future_positions(trace);

        for (request_index, req) in trace.requests().iter().enumerate() {
            let key = req.key;

            if let Some(q) = future_positions.get_mut(&key) {
                if q.front() == Some(&request_index) {
                    q.pop_front();
                }
            }

            let decision_tick = cache.clock.get_tick().saturating_add(1);

            let is_hit = cache.store.contains_key(&key);
            let is_full = cache.store.len() >= cache.capacity;
            let eviction_decision = !is_hit && is_full;

            if eviction_decision {
                let candidates: Vec<CacheKey> = cache.store.keys().copied().collect();

                for i in 0..candidates.len() {
                    for j in (i + 1)..candidates.len() {
                        let key0 = candidates[i];
                        let key1 = candidates[j];

                        let cmp =
                            Self::compare_candidates_by_future_use(key0, key1, &future_positions);

                        if cmp == Ordering::Equal && config.skip_ties {
                            continue;
                        }

                        let entry0 = cache
                            .store
                            .get(&key0)
                            .expect("candidate key missing from cache.store");
                        let entry1 = cache
                            .store
                            .get(&key1)
                            .expect("candidate key missing from cache.store");

                        let hist0 = feature_state.get(&key0);
                        let hist1 = feature_state.get(&key1);

                        let phi0 = Self::extract_features(
                            entry0,
                            hist0,
                            decision_tick,
                            config.decay_factors.len(),
                        );
                        let phi1 = Self::extract_features(
                            entry1,
                            hist1,
                            decision_tick,
                            config.decay_factors.len(),
                        );

                        let x01 = Self::subtract_features(&phi0, &phi1);
                        let y01 = match cmp {
                            Ordering::Greater => 1,
                            Ordering::Less => 0,
                            Ordering::Equal => 0,
                        };

                        dataset.push(PairwiseSample {
                            trace_name: trace_name.to_string(),
                            cache_size: cache.capacity,
                            request_index,
                            tick: decision_tick,
                            key0,
                            key1,
                            x: x01,
                            y: y01,
                        });

                        if config.add_swapped_pairs {
                            let x10 = Self::subtract_features(&phi1, &phi0);
                            let y10 = 1 - y01;

                            dataset.push(PairwiseSample {
                                trace_name: trace_name.to_string(),
                                cache_size: cache.capacity,
                                request_index,
                                tick: decision_tick,
                                key0: key1,
                                key1: key0,
                                x: x10,
                                y: y10,
                            });
                        }
                    }
                }
            }

            cache.access(key);

            let now_tick = cache.clock.get_tick();
            match feature_state.get_mut(&key) {
                Some(fs) => fs.update_on_request(now_tick, &config.decay_factors),
                None => {
                    feature_state
                        .insert(key, FeatureState::new(now_tick, config.decay_factors.len()));
                }
            }
        }

        dataset
    }

    fn build_future_positions(trace: &RequestTrace) -> HashMap<CacheKey, VecDeque<usize>> {
        let mut positions: HashMap<CacheKey, VecDeque<usize>> = HashMap::new();

        for (i, req) in trace.requests().iter().enumerate() {
            positions.entry(req.key).or_default().push_back(i);
        }

        positions
    }

    fn compare_candidates_by_future_use(
        key0: CacheKey,
        key1: CacheKey,
        future_positions: &HashMap<CacheKey, VecDeque<usize>>,
    ) -> Ordering {
        let next0 = future_positions
            .get(&key0)
            .and_then(|q| q.front().copied())
            .unwrap_or(usize::MAX);

        let next1 = future_positions
            .get(&key1)
            .and_then(|q| q.front().copied())
            .unwrap_or(usize::MAX);

        next0.cmp(&next1)
    }

    fn subtract_features(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "feature vectors must have same length");
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    /// Feature order:
    ///  0. resident age since insertion
    ///  1. resident time since last access
    ///  2. resident access_count
    ///  3. resident frequency
    ///  4. global age since first-ever request
    ///  5. global time since last request
    ///  6. global total request count
    ///  7. last interarrival
    ///  8. avg interarrival
    ///  9. gap count
    /// 10.. decayed frequency counters
    fn extract_features(
        entry: &Entry<CacheKey>,
        history: Option<&FeatureState>,
        decision_tick: u64,
        decay_dims: usize,
    ) -> Vec<f32> {
        let resident_age = decision_tick.saturating_sub(entry.insertion_tick) as f32;
        let resident_time_since_last = decision_tick.saturating_sub(entry.last_access_tick) as f32;

        let mut v = vec![
            resident_age,
            resident_time_since_last,
            entry.access_count as f32,
            entry.frequency(decision_tick) as f32,
        ];

        if let Some(h) = history {
            v.push(decision_tick.saturating_sub(h.first_request_tick) as f32);
            v.push(decision_tick.saturating_sub(h.last_request_tick) as f32);
            v.push(h.total_request_count as f32);
            v.push(h.last_interarrival);
            v.push(h.avg_interarrival);
            v.push(h.gap_count as f32);
            v.extend(h.decay_counters.iter().copied());
        } else {
            v.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            v.extend(std::iter::repeat(0.0).take(decay_dims));
        }

        v
    }
}
