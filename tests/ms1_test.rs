use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use lbce::core::cache::Cache;
use lbce::core::mainmemory::{MainMemory, MemoryObject};
use lbce::core::replay_engine::replay_trace;
use lbce::core::trace::{CacheEvent, Request, RequestTrace};
use lbce::policies::fifo::FifoPolicy;
use lbce::policies::naivelru::LruPolicy;
use lbce::workloads::bursty::BurstyWorkload;
use lbce::analysis::io::{
    load_request_trace_csv,
    write_request_trace_csv,
};
use lbce::workloads::looping::LoopingWorkload;
use lbce::workloads::phase::PhaseWorkload;
use lbce::analysis::summary::Summary;
use lbce::workloads::workload::collect_requests;
use lbce::workloads::zipf::ZipfWorkload;

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{name}_{nanos}.csv"))
}

fn populated_main_memory<const N: usize>(keys: &[u64]) -> MainMemory<N> {
    let mut mm = MainMemory::new();
    for &key in keys {
        mm.insert(MemoryObject::new(key, vec![0u8; 1]));
    }
    mm
}

fn fifo_cache(capacity: usize, keys: &[u64]) -> Cache<FifoPolicy, 1024> {
    let mm = populated_main_memory::<1024>(keys);
    Cache::new(capacity, FifoPolicy::new(capacity), mm)
}

fn lru_cache(capacity: usize, keys: &[u64]) -> Cache<LruPolicy, 1024> {
    let mm = populated_main_memory::<1024>(keys);
    Cache::new(capacity, LruPolicy::new(capacity), mm)
}

#[test]
fn looping_workload_generates_expected_sequence() {
    let mut workload = LoopingWorkload::new(vec![10, 20, 30], 8);
    let trace = collect_requests(&mut workload);

    let keys: Vec<u64> = trace.requests().iter().map(|r| r.key).collect();
    assert_eq!(keys, vec![10, 20, 30, 10, 20, 30, 10, 20]);
    assert_eq!(trace.len(), 8);
}

#[test]
fn phase_workload_switches_key_ranges_correctly() {
    let mut workload = PhaseWorkload::new(2, 3, 4);
    let trace = collect_requests(&mut workload);

    let keys: Vec<u64> = trace.requests().iter().map(|r| r.key).collect();

    // Phase 0: keys in [0,1,2], repeated for 4 requests
    assert_eq!(&keys[0..4], &[0, 1, 2, 0]);

    // Phase 1: keys in [3,4,5], repeated for 4 requests
    assert_eq!(&keys[4..8], &[3, 4, 5, 3]);
}

#[test]
fn bursty_workload_has_quiet_then_burst_pattern() {
    let mut workload = BurstyWorkload::new(2, 3, 2, 4);
    let trace = collect_requests(&mut workload);

    let keys: Vec<u64> = trace.requests().iter().map(|r| r.key).collect();

    // cycle 1: quiet(0,1,2), burst(4,4)
    // cycle 2: quiet(0,1,2), burst(4,4)
    assert_eq!(keys, vec![0, 1, 2, 4, 4, 0, 1, 2, 4, 4]);
}

#[test]
fn zipf_workload_is_reproducible_and_uses_only_given_keys() {
    let keys = vec![100, 200, 300, 400];
    let mut w1 = ZipfWorkload::new(keys.clone(), 50, 1.2, 12345);
    let mut w2 = ZipfWorkload::new(keys.clone(), 50, 1.2, 12345);

    let t1 = collect_requests(&mut w1);
    let t2 = collect_requests(&mut w2);

    let s1: Vec<u64> = t1.requests().iter().map(|r| r.key).collect();
    let s2: Vec<u64> = t2.requests().iter().map(|r| r.key).collect();

    assert_eq!(s1, s2);
    assert_eq!(s1.len(), 50);
    assert!(s1.iter().all(|k| keys.contains(k)));
}

#[test]
fn summary_reports_requests_uniques_and_hottest_key() {
    let mut trace = RequestTrace::new();
    for key in [7, 9, 7, 5, 7, 9] {
        trace.push(Request::new(key));
    }

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 6);
    assert_eq!(summary.num_unique_keys, 3);
    assert_eq!(summary.most_frequent_key, Some(7));
    assert_eq!(summary.most_frequent_key_count, 3);
}

#[test]
fn replay_trace_fifo_returns_expected_metrics() {
    let mut trace = RequestTrace::new();
    for key in [1, 2, 1, 3] {
        trace.push(Request::new(key));
    }

    let mut cache = fifo_cache(2, &[1, 2, 3]);
    let result = replay_trace(&trace, &mut cache);

    assert_eq!(result.request_count, 4);
    assert_eq!(result.hit_count, 1);
    assert_eq!(result.eviction_count, 1);
    assert!((result.hit_rate - 0.25).abs() < 1e-12);
    assert!((result.miss_rate - 0.75).abs() < 1e-12);

    // FIFO on [1,2,1,3] with cap 2 evicts 1
    assert!(!cache.store.contains_key(&1));
    assert!(cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn replay_trace_lru_returns_expected_metrics() {
    let mut trace = RequestTrace::new();
    for key in [1, 2, 1, 3] {
        trace.push(Request::new(key));
    }

    let mut cache = lru_cache(2, &[1, 2, 3]);
    let result = replay_trace(&trace, &mut cache);

    assert_eq!(result.request_count, 4);
    assert_eq!(result.hit_count, 1);
    assert_eq!(result.eviction_count, 1);
    assert!((result.hit_rate - 0.25).abs() < 1e-12);
    assert!((result.miss_rate - 0.75).abs() < 1e-12);

    // LRU on [1,2,1,3] with cap 2 evicts 2
    assert!(cache.store.contains_key(&1));
    assert!(!cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn replay_trace_populates_cache_event_trace() {
    let mut trace = RequestTrace::new();
    for key in [1, 2, 1, 3] {
        trace.push(Request::new(key));
    }

    let mut cache = fifo_cache(2, &[1, 2, 3]);
    let _ = replay_trace(&trace, &mut cache);

    let events = cache.event_trace.events();
    assert!(!events.is_empty());

    let hit_count = events.iter().filter(|e| matches!(e, CacheEvent::Hit { .. })).count();
    let miss_count = events.iter().filter(|e| matches!(e, CacheEvent::Miss { .. })).count();
    let insert_count = events.iter().filter(|e| matches!(e, CacheEvent::Insert { .. })).count();
    let evict_count = events.iter().filter(|e| matches!(e, CacheEvent::Evict { .. })).count();

    assert_eq!(hit_count, 1);
    assert_eq!(miss_count, 3);
    assert_eq!(insert_count, 3);
    assert_eq!(evict_count, 1);
}

#[test]
fn write_and_load_request_trace_csv_works() {
    let mut trace = RequestTrace::new();
    for key in [11, 22, 11, 33] {
        trace.push(Request::new(key));
    }

    let path = unique_temp_path("request_trace_roundtrip");
    write_request_trace_csv(path.to_str().unwrap(), &trace);

    // NOTE: load_request_trace_csv hashes string keys from CSV.
    // So this is not a numeric roundtrip test; it is a shape/count consistency test.
    let loaded = load_request_trace_csv(path.to_str().unwrap());

    assert_eq!(loaded.len(), 4);

    let loaded_keys: Vec<u64> = loaded.requests().iter().map(|r| r.key).collect();
    assert_eq!(loaded_keys[0], loaded_keys[2]); // "11" hashes the same both times
    assert_ne!(loaded_keys[0], loaded_keys[1]); // "11" vs "22"

    fs::remove_file(path).unwrap();
}

#[test]
fn collect_requests_and_replay_work_together() {
    let mut workload = LoopingWorkload::new(vec![1, 2], 6);
    let trace = collect_requests(&mut workload);

    let mut cache = fifo_cache(2, &[1, 2]);
    let result = replay_trace(&trace, &mut cache);

    // Request sequence: 1,2,1,2,1,2
    // First two misses, next four hits
    assert_eq!(result.request_count, 6);
    assert_eq!(result.hit_count, 4);
    assert_eq!(result.eviction_count, 0);
    assert!((result.hit_rate - (4.0 / 6.0)).abs() < 1e-12);
}