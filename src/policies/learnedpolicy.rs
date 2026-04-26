use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::core::policy::{CacheKey, Policy};
use crate::deployed::{EvictionMLPNormalized, FEATURE_DIM};
use crate::policies::label_maturation::{ReplayBuffer, mature_decision};

use crate::data::pairwise_csv_writer::PairwiseCsvWriter;

type MyBackend = burn_ndarray::NdArray<f32>;

const DEFAULT_DECAY_FACTORS: [f32; 3] = [0.5, 0.8, 0.95];
const DEFAULT_SHORTLIST_K: usize = 4;
const DEFAULT_MATURITY_WINDOW: u64 = 100;
const DEFAULT_REPLAY_BUFFER_CAPACITY: usize = 10_000;
const RETRAIN_EVERY: u64 = 100;
const MIN_BUFFER_SIZE: usize = 500;
const RETRAIN_SAMPLE_SIZE: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionSource {
    Model,
    Fallback,
    SingleCandidate,
}

#[derive(Clone, Debug)]
pub struct CandidateSnapshot {
    pub key: CacheKey,
    pub features: [f32; FEATURE_DIM],
}

#[derive(Clone, Debug)]
pub struct PendingDecision {
    pub decision_id: u64,
    pub model_version: u64,
    pub source: DecisionSource,
    pub decision_tick: u64,
    pub miss_key: CacheKey,
    pub resident_candidates: Vec<CandidateSnapshot>,
    pub shortlist_candidates: Vec<CacheKey>,
    pub chosen_victim: CacheKey,
}

impl PendingDecision {
    pub fn candidate(&self, key: CacheKey) -> Option<&CandidateSnapshot> {
        self.resident_candidates
            .iter()
            .find(|candidate| candidate.key == key)
    }

    pub fn pair_features(&self, key0: CacheKey, key1: CacheKey) -> Option<[f32; FEATURE_DIM]> {
        let phi0 = self.candidate(key0)?.features;
        let phi1 = self.candidate(key1)?.features;

        let mut diff = [0.0; FEATURE_DIM];
        for idx in 0..FEATURE_DIM {
            diff[idx] = phi0[idx] - phi1[idx];
        }

        Some(diff)
    }
}

#[derive(Clone, Debug)]
struct ResidentState {
    insertion_tick: u64,
    last_access_tick: u64,
    access_count: u64,
}

impl ResidentState {
    fn new(now_tick: u64) -> Self {
        Self {
            insertion_tick: now_tick,
            last_access_tick: now_tick,
            access_count: 1,
        }
    }

    fn on_access(&mut self, now_tick: u64) {
        self.last_access_tick = now_tick;
        self.access_count = self.access_count.saturating_add(1);
    }

    fn frequency(&self, now_tick: u64) -> f32 {
        let age = now_tick
            .saturating_sub(self.insertion_tick)
            .saturating_add(1) as f32;
        self.access_count as f32 / age
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

    fn observe(&mut self, now_tick: u64, decay_factors: &[f32]) {
        let delta = now_tick.saturating_sub(self.last_request_tick) as f32;

        self.last_interarrival = delta;
        self.avg_interarrival = if self.gap_count == 0 {
            delta
        } else {
            (self.avg_interarrival * self.gap_count as f32 + delta) / (self.gap_count as f32 + 1.0)
        };
        self.gap_count = self.gap_count.saturating_add(1);

        for (counter, alpha) in self
            .decay_counters
            .iter_mut()
            .zip(decay_factors.iter().copied())
        {
            *counter = *counter * alpha.powf(delta) + 1.0;
        }

        self.total_request_count = self.total_request_count.saturating_add(1);
        self.last_request_tick = now_tick;
    }
}

#[derive(Clone, Debug)]
struct PendingModelSwap {
    path: PathBuf,
}

pub struct LearnedPolicy {
    model: Option<EvictionMLPNormalized<MyBackend>>,
    model_path: PathBuf,
    shortlist_k: usize,
    debug: bool,
    tick: u64,
    pending_miss: Option<CacheKey>,
    residents: HashMap<CacheKey, ResidentState>,
    history: HashMap<CacheKey, FeatureState>,
    decay_factors: Vec<f32>,
    model_version: u64,
    next_decision_id: u64,
    pending_decisions: VecDeque<PendingDecision>,
    // Step 2: label maturation
    maturity_window: u64,
    future_access_ticks: HashMap<CacheKey, VecDeque<u64>>,
    replay_buffer: ReplayBuffer,
    pending_swap: Option<PendingModelSwap>,
    swap_checkpoint_interval: Option<u64>,
}

impl LearnedPolicy {
    pub fn new() -> Self {
        Self::from_path("eviction_mlp.pt")
    }

    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self::with_decay_factors(path, DEFAULT_DECAY_FACTORS.to_vec())
    }

    pub fn with_decay_factors(path: impl AsRef<Path>, decay_factors: Vec<f32>) -> Self {
        Self::with_config(path, decay_factors, DEFAULT_SHORTLIST_K)
    }

    pub fn with_shortlist_k(path: impl AsRef<Path>, shortlist_k: usize) -> Self {
        Self::with_config(path, DEFAULT_DECAY_FACTORS.to_vec(), shortlist_k)
    }

    pub fn with_config(
        path: impl AsRef<Path>,
        decay_factors: Vec<f32>,
        shortlist_k: usize,
    ) -> Self {
        Self::with_config_and_debug(path, decay_factors, shortlist_k, false)
    }

    pub fn with_config_and_debug(
        path: impl AsRef<Path>,
        decay_factors: Vec<f32>,
        shortlist_k: usize,
        debug: bool,
    ) -> Self {
        let device = Default::default();
        let path_buf = path.as_ref().to_path_buf();
        let model = match EvictionMLPNormalized::<MyBackend>::load(&path_buf, &device) {
            Ok(model) => Some(model),
            Err(err) => {
                eprintln!("{err}");
                None
            }
        };

        Self {
            model,
            model_path: path_buf,
            shortlist_k: shortlist_k.max(1),
            debug,
            tick: 0,
            pending_miss: None,
            residents: HashMap::new(),
            history: HashMap::new(),
            decay_factors,
            model_version: 0,
            next_decision_id: 0,
            pending_decisions: VecDeque::new(),
            maturity_window: DEFAULT_MATURITY_WINDOW,
            future_access_ticks: HashMap::new(),
            replay_buffer: ReplayBuffer::new(DEFAULT_REPLAY_BUFFER_CAPACITY),
            pending_swap: None,
            swap_checkpoint_interval: None,
        }
    }

    pub fn without_model() -> Self {
        Self::without_model_with_shortlist_k(DEFAULT_SHORTLIST_K)
    }

    pub fn without_model_with_shortlist_k(shortlist_k: usize) -> Self {
        Self {
            model: None,
            model_path: PathBuf::from("eviction_mlp.pt"),
            shortlist_k: shortlist_k.max(1),
            debug: false,
            tick: 0,
            pending_miss: None,
            residents: HashMap::new(),
            history: HashMap::new(),
            decay_factors: DEFAULT_DECAY_FACTORS.to_vec(),
            model_version: 0,
            next_decision_id: 0,
            pending_decisions: VecDeque::new(),
            maturity_window: DEFAULT_MATURITY_WINDOW,
            future_access_ticks: HashMap::new(),
            replay_buffer: ReplayBuffer::new(DEFAULT_REPLAY_BUFFER_CAPACITY),
            pending_swap: None,
            swap_checkpoint_interval: None,
        }
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn shortlist_k(&self) -> usize {
        self.shortlist_k
    }

    pub fn debug_enabled(&self) -> bool {
        self.debug
    }

    pub fn model_version(&self) -> u64 {
        self.model_version
    }

    pub fn pending_decision_count(&self) -> usize {
        self.pending_decisions.len()
    }

    pub fn pending_decisions(&self) -> &VecDeque<PendingDecision> {
        &self.pending_decisions
    }

    pub fn pop_oldest_pending_decision(&mut self) -> Option<PendingDecision> {
        self.pending_decisions.pop_front()
    }

    pub fn maturity_window(&self) -> u64 {
        self.maturity_window
    }

    pub fn set_maturity_window(&mut self, window: u64) {
        self.maturity_window = window;
    }

    pub fn replay_buffer(&self) -> &ReplayBuffer {
        &self.replay_buffer
    }

    pub fn replay_buffer_mut(&mut self) -> &mut ReplayBuffer {
        &mut self.replay_buffer
    }

    pub fn request_model_swap(&mut self, path: impl AsRef<Path>) {
        self.pending_swap = Some(PendingModelSwap {
            path: path.as_ref().to_path_buf(),
        });
    }

    pub fn set_swap_checkpoint_interval(&mut self, interval: Option<u64>) {
        self.swap_checkpoint_interval = interval.map(|value| value.max(1));
    }

    pub fn pending_model_swap_path(&self) -> Option<&Path> {
        self.pending_swap.as_ref().map(|swap| swap.path.as_path())
    }

    fn should_apply_swap_now(&self) -> bool {
        match self.swap_checkpoint_interval {
            Some(interval) => self.tick % interval == 0,
            None => true,
        }
    }

    fn apply_pending_swap_if_ready(&mut self) {
        if self.pending_swap.is_none() || !self.should_apply_swap_now() {
            return;
        }

        let pending = self
            .pending_swap
            .take()
            .expect("pending model swap must exist when applying");
        let device = Default::default();
        match EvictionMLPNormalized::<MyBackend>::load(&pending.path, &device) {
            Ok(model) => {
                self.model = Some(model);
                self.model_path = pending.path;
                self.model_version = self.model_version.saturating_add(1);
            }
            Err(err) => {
                eprintln!("{err}");
            }
        }
    }

    fn record_request(&mut self, key: CacheKey) {
        match self.history.get_mut(&key) {
            Some(state) => state.observe(self.tick, &self.decay_factors),
            None => {
                self.history
                    .insert(key, FeatureState::new(self.tick, self.decay_factors.len()));
            }
        }
    }

    fn extract_features(&self, key: CacheKey) -> [f32; FEATURE_DIM] {
        let resident = self
            .residents
            .get(&key)
            .expect("resident key missing from learned policy");

        let resident_age = self.tick.saturating_sub(resident.insertion_tick) as f32;
        let resident_time_since_last = self.tick.saturating_sub(resident.last_access_tick) as f32;

        let mut values = Vec::with_capacity(FEATURE_DIM);
        values.push(resident_age);
        values.push(resident_time_since_last);
        values.push(resident.access_count as f32);
        values.push(resident.frequency(self.tick));

        if let Some(history) = self.history.get(&key) {
            values.push(self.tick.saturating_sub(history.first_request_tick) as f32);
            values.push(self.tick.saturating_sub(history.last_request_tick) as f32);
            values.push(history.total_request_count as f32);
            values.push(history.last_interarrival);
            values.push(history.avg_interarrival);
            values.push(history.gap_count as f32);
            values.extend(history.decay_counters.iter().copied());
        } else {
            values.extend(std::iter::repeat_n(0.0, 6 + self.decay_factors.len()));
        }

        values
            .try_into()
            .expect("learned policy generated an unexpected feature dimension")
    }

    fn pair_features(&self, key0: CacheKey, key1: CacheKey) -> [f32; FEATURE_DIM] {
        let phi0 = self.extract_features(key0);
        let phi1 = self.extract_features(key1);

        let mut diff = [0.0; FEATURE_DIM];
        for idx in 0..FEATURE_DIM {
            diff[idx] = phi0[idx] - phi1[idx];
        }
        diff
    }

    fn fallback_victim(&self) -> Option<CacheKey> {
        self.residents
            .iter()
            .max_by_key(|(_, state)| {
                (
                    self.tick.saturating_sub(state.last_access_tick),
                    self.tick.saturating_sub(state.insertion_tick),
                )
            })
            .map(|(&key, _)| key)
    }

    fn resident_keys_by_priority(&self) -> Vec<CacheKey> {
        let mut keys: Vec<CacheKey> = self.residents.keys().copied().collect();
        keys.sort_by_key(|key| {
            let state = self
                .residents
                .get(key)
                .expect("resident missing while ordering learned-policy candidates");
            (state.last_access_tick, state.insertion_tick, *key)
        });
        keys
    }

    fn build_pending_decision(
        &self,
        chosen_victim: CacheKey,
        shortlist_candidates: &[CacheKey],
        source: DecisionSource,
        decision_id: u64,
    ) -> PendingDecision {
        let miss_key = self
            .pending_miss
            .expect("pending miss must exist when recording a learned-policy decision");

        let resident_candidates = self
            .resident_keys_by_priority()
            .into_iter()
            .map(|key| CandidateSnapshot {
                key,
                features: self.extract_features(key),
            })
            .collect();

        PendingDecision {
            decision_id,
            model_version: self.model_version,
            source,
            decision_tick: self.tick,
            miss_key,
            resident_candidates,
            shortlist_candidates: shortlist_candidates.to_vec(),
            chosen_victim,
        }
    }

    fn record_pending_decision(
        &mut self,
        chosen_victim: CacheKey,
        shortlist_candidates: &[CacheKey],
        source: DecisionSource,
    ) {
        let decision_id = self.next_decision_id;
        self.next_decision_id = self.next_decision_id.saturating_add(1);

        let snapshot =
            self.build_pending_decision(chosen_victim, shortlist_candidates, source, decision_id);
        self.pending_decisions.push_back(snapshot);

        // Watch shortlist keys so we can record their future accesses for labeling.
        for &key in shortlist_candidates {
            self.future_access_ticks.entry(key).or_default();
        }
    }

    fn record_future_access(&mut self, key: CacheKey) {
        if let Some(ticks) = self.future_access_ticks.get_mut(&key) {
            ticks.push_back(self.tick);
        }
    }

    /// Drain and label all pending decisions that have waited at least
    /// `maturity_window` ticks. Labeled examples are pushed to the replay buffer.
    pub fn try_mature_pending_decisions(&mut self) {
        while let Some(front) = self.pending_decisions.front() {
            if self.tick.saturating_sub(front.decision_tick) < self.maturity_window {
                break;
            }
            let decision = self.pending_decisions.pop_front().unwrap();
            let examples = mature_decision(&decision, &self.future_access_ticks, true, true);
            for example in examples {
                self.replay_buffer.push(example);
            }
            self.prune_future_access_ticks();
        }
    }

    /// Remove future-access tracking for keys no longer referenced by any pending decision.
    fn prune_future_access_ticks(&mut self) {
        let still_needed: HashSet<CacheKey> = self
            .pending_decisions
            .iter()
            .flat_map(|d| d.shortlist_candidates.iter().copied())
            .collect();
        self.future_access_ticks
            .retain(|k, _| still_needed.contains(k));
    }

    fn shortlist_candidates(&self) -> Vec<CacheKey> {
        let mut keys = self.resident_keys_by_priority();
        keys.truncate(self.shortlist_k.min(keys.len()));
        keys
    }

    fn maybe_retrain(&mut self) {

        if self.replay_buffer.len() < MIN_BUFFER_SIZE {
            return;
        }

        if self.next_decision_id % RETRAIN_EVERY != 0 {
            return;
        }

        let data = self.replay_buffer.sample(RETRAIN_SAMPLE_SIZE);

        let data_path_string = "data/online_training.csv";

        let data_path = PathBuf::from(data_path_string);

        if let Err(err) = PairwiseCsvWriter::write_to_path(
            &data_path,
            &data,
            self.decay_factors.len(),
        ) {
            println!("failed to write training csv");
            return;
        }

        let output = std::process::Command::new("python3")
            .current_dir("pytorch_model")
            .arg("model.py")
            .arg("--train-csv")
            .arg(data_path_string)
            .arg("--val-csv")
            .arg(data_path_string)
            .arg("--epochs")
            .arg("5")
            .arg("--init-checkpoint")
            .arg("eviction_mlp.pt")
            .output();

        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("pytorch_model")
            .join("eviction_mlp.pt");

        match output {
            Ok(out) if out.status.success() => {
                self.request_model_swap(model_path);
            }
            Ok(out) => {
                println!("training failed");
            }
            Err(err) => {
                println!("training process failed to start");
            }
        }

    }

}

impl Policy for LearnedPolicy {
    fn on_hit(&mut self, key: CacheKey) {
        self.tick = self.tick.saturating_add(1);
        self.pending_miss = None;

        if let Some(resident) = self.residents.get_mut(&key) {
            resident.on_access(self.tick);
        }
        self.record_request(key);
        self.record_future_access(key);
        self.try_mature_pending_decisions();
        self.maybe_retrain();
        self.apply_pending_swap_if_ready();
    }

    fn on_miss(&mut self, key: CacheKey) {
        self.tick = self.tick.saturating_add(1);
        self.pending_miss = Some(key);
        self.try_mature_pending_decisions();
        self.maybe_retrain();
    }

    fn insert(&mut self, key: CacheKey) {
        self.residents.insert(key, ResidentState::new(self.tick));
        self.record_request(key);

        if self.pending_miss == Some(key) {
            self.pending_miss = None;
        }

        self.apply_pending_swap_if_ready();
    }

    fn remove(&mut self, key: CacheKey) {
        self.residents.remove(&key);
    }

    fn victim(&mut self) -> Option<CacheKey> {
        let keys = self.shortlist_candidates();
        if keys.is_empty() {
            return None;
        }
        let (victim, source) = if keys.len() == 1 {
            (keys[0], DecisionSource::SingleCandidate)
        } else if let Some(model) = &self.model {
            if self.debug {
                eprintln!("[learned] shortlist={keys:?}");
            }

            let mut scores: HashMap<CacheKey, usize> =
                keys.iter().copied().map(|key| (key, 0)).collect();

            for i in 0..keys.len() {
                for j in (i + 1)..keys.len() {
                    let key0 = keys[i];
                    let key1 = keys[j];
                    let features = self.pair_features(key0, key1);
                    let logit = model.predict_pair(&features);

                    let winner = if logit > 0.0 { key0 } else { key1 };
                    if self.debug {
                        eprintln!(
                            "[learned] pair key0={} key1={} logit={:.5} winner={}",
                            key0, key1, logit, winner
                        );
                    }
                    *scores
                        .get_mut(&winner)
                        .expect("winner must exist in learned policy scores") += 1;
                }
            }

            let victim = scores
                .into_iter()
                .max_by_key(|(key, score)| {
                    let state = self
                        .residents
                        .get(key)
                        .expect("resident missing while selecting learned-policy victim");
                    (
                        *score,
                        self.tick.saturating_sub(state.last_access_tick),
                        self.tick.saturating_sub(state.insertion_tick),
                    )
                })
                .map(|(key, _)| key)
                .or_else(|| self.fallback_victim())?;

            (victim, DecisionSource::Model)
        } else {
            (self.fallback_victim()?, DecisionSource::Fallback)
        };

        if self.pending_miss.is_some() {
            self.record_pending_decision(victim, &keys, source);
        }

        if self.debug {
            eprintln!("[learned] victim={victim:?} source={source:?}");
        }

        Some(victim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_behaves_like_lru_for_simple_sequence() {
        let mut policy = LearnedPolicy::without_model();

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);
        policy.on_hit(1);

        assert_eq!(policy.victim(), Some(2));
    }

    #[test]
    fn miss_does_not_pollute_candidate_history_before_insert() {
        let mut policy = LearnedPolicy::without_model();

        policy.on_miss(10);
        assert!(!policy.history.contains_key(&10));

        policy.insert(10);
        assert!(policy.history.contains_key(&10));
        assert!(policy.residents.contains_key(&10));
    }

    #[test]
    fn shortlist_uses_least_recently_used_candidates() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);
        policy.on_miss(3);
        policy.insert(3);
        policy.on_hit(1);
        policy.on_hit(3);

        assert_eq!(policy.shortlist_candidates(), vec![2, 1]);
        assert_eq!(policy.victim(), Some(2));
    }

    #[test]
    fn full_cache_miss_records_pending_decision_snapshot() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);
        policy.on_hit(1);

        policy.on_miss(3);
        let victim = policy.victim();

        assert_eq!(victim, Some(2));
        assert_eq!(policy.pending_decision_count(), 1);

        let snapshot = policy
            .pending_decisions()
            .back()
            .expect("expected pending decision snapshot");

        assert_eq!(snapshot.decision_id, 0);
        assert_eq!(snapshot.model_version, 0);
        assert_eq!(snapshot.decision_tick, 4);
        assert_eq!(snapshot.miss_key, 3);
        assert_eq!(snapshot.chosen_victim, 2);
        assert_eq!(snapshot.source, DecisionSource::Fallback);
        assert_eq!(snapshot.shortlist_candidates, vec![2, 1]);

        let resident_keys: Vec<CacheKey> = snapshot
            .resident_candidates
            .iter()
            .map(|candidate| candidate.key)
            .collect();
        assert_eq!(resident_keys, vec![2, 1]);
    }

    #[test]
    fn under_capacity_insert_does_not_record_pending_decision() {
        let mut policy = LearnedPolicy::without_model();

        policy.on_miss(10);
        policy.insert(10);

        assert_eq!(policy.pending_decision_count(), 0);
    }

    #[test]
    fn popping_pending_decision_returns_oldest_snapshot() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);

        policy.on_miss(3);
        let _ = policy.victim();

        let snapshot = policy
            .pop_oldest_pending_decision()
            .expect("expected pending decision");
        assert_eq!(snapshot.decision_id, 0);
        assert_eq!(policy.pending_decision_count(), 0);
    }

    #[test]
    fn pending_decision_can_reconstruct_pair_features() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);
        policy.on_hit(1);

        policy.on_miss(3);
        let _ = policy.victim();

        let snapshot = policy
            .pending_decisions()
            .back()
            .expect("expected pending decision snapshot");

        let from_snapshot = snapshot
            .pair_features(2, 1)
            .expect("expected pairwise features from snapshot");
        let from_policy = policy.pair_features(2, 1);

        assert_eq!(from_snapshot, from_policy);
        assert!(snapshot.pair_features(2, 999).is_none());
    }

    #[test]
    fn matured_decisions_are_labeled_into_replay_buffer() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);
        policy.set_maturity_window(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);
        policy.on_hit(1);

        policy.on_miss(3);
        assert_eq!(policy.victim(), Some(2));
        assert_eq!(policy.pending_decision_count(), 1);
        assert!(policy.replay_buffer().is_empty());

        // key 1 is reused before key 2, so key 2 should be evicted first.
        policy.on_hit(1);
        policy.on_miss(4);

        assert_eq!(policy.pending_decision_count(), 0);
        let samples: Vec<_> = policy.replay_buffer().iter().collect();
        assert_eq!(samples.len(), 2);

        let forward = samples
            .iter()
            .find(|sample| sample.key0 == 2 && sample.key1 == 1)
            .expect("expected forward pairwise sample");
        assert_eq!(forward.y, 1);

        let swapped = samples
            .iter()
            .find(|sample| sample.key0 == 1 && sample.key1 == 2)
            .expect("expected swapped pairwise sample");
        assert_eq!(swapped.y, 0);
    }

    #[test]
    fn model_swap_is_not_applied_during_victim_decision() {
        let mut policy = LearnedPolicy::without_model_with_shortlist_k(2);

        policy.on_miss(1);
        policy.insert(1);
        policy.on_miss(2);
        policy.insert(2);

        policy.set_swap_checkpoint_interval(Some(100));
        policy.request_model_swap("eviction_mlp.pt");

        assert!(policy.pending_model_swap_path().is_some());
        let _ = policy.victim();
        assert!(policy.pending_model_swap_path().is_some());
    }

    #[test]
    fn model_swap_applies_only_on_checkpoint_boundaries() {
        let mut policy = LearnedPolicy::without_model();

        policy.on_miss(1);
        policy.insert(1);

        policy.set_swap_checkpoint_interval(Some(4));
        policy.request_model_swap("definitely_missing_model_checkpoint.pt");

        policy.on_hit(1);
        assert!(policy.pending_model_swap_path().is_some());

        policy.on_hit(1);
        assert!(policy.pending_model_swap_path().is_some());

        policy.on_hit(1);
        assert!(policy.pending_model_swap_path().is_none());
    }
}
