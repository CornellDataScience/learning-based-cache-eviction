use lbce::analysis::summary::Summary;
use lbce::core::trace::{CacheEvent, CacheTrace};

#[test]
fn test_summary_empty_trace() {
    let trace = CacheTrace::new(true);
    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 0);
    assert_eq!(summary.num_unique_keys, 0);
    assert_eq!(summary.most_frequent_key, 0);
    assert_eq!(summary.most_frequent_key_count, 0);
}

#[test]
fn test_summary_single_request() {
    let mut trace = CacheTrace::new(true);
    trace.record_event(CacheEvent::Hit { key: 1, tick: 1 });

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 1);
    assert_eq!(summary.num_unique_keys, 1);
    assert_eq!(summary.most_frequent_key, 1);
    assert_eq!(summary.most_frequent_key_count, 1);
}

#[test]
fn test_summary_multiple_unique_keys() {
    let mut trace = CacheTrace::new(true);
    trace.record_event(CacheEvent::Hit { key: 1, tick: 1 });
    trace.record_event(CacheEvent::Miss { key: 2, tick: 2 });
    trace.record_event(CacheEvent::Hit { key: 3, tick: 3 });

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 3);
    assert_eq!(summary.num_unique_keys, 3);
}

#[test]
fn test_summary_most_frequent_key() {
    let mut trace = CacheTrace::new(true);
    trace.record_event(CacheEvent::Hit { key: 1, tick: 1 });
    trace.record_event(CacheEvent::Hit { key: 1, tick: 2 });
    trace.record_event(CacheEvent::Hit { key: 1, tick: 3 });
    trace.record_event(CacheEvent::Miss { key: 2, tick: 4 });

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 4);
    assert_eq!(summary.num_unique_keys, 2);
    assert_eq!(summary.most_frequent_key, 1);
    assert_eq!(summary.most_frequent_key_count, 3);
}

#[test]
fn test_summary_most_frequent_key_tie() {
    let mut trace = CacheTrace::new(true);
    trace.record_event(CacheEvent::Hit { key: 1, tick: 1 });
    trace.record_event(CacheEvent::Hit { key: 2, tick: 2 });
    trace.record_event(CacheEvent::Hit { key: 1, tick: 3 });
    trace.record_event(CacheEvent::Hit { key: 2, tick: 4 });

    let summary = Summary::summarize(&trace);
    assert_eq!(summary.most_frequent_key_count, 2);
    assert!([1, 2].contains(&summary.most_frequent_key));
}

#[test]
fn test_summary_with_insert_evict_events() {
    let mut trace = CacheTrace::new(true);
    trace.record_event(CacheEvent::Hit { key: 1, tick: 1 });
    trace.record_event(CacheEvent::Miss { key: 2, tick: 2 });
    trace.record_event(CacheEvent::Insert {
        key: 3,
        size_bytes: 64,
        tick: 3,
    });
    trace.record_event(CacheEvent::Evict {
        key: 4,
        size_bytes: 64,
        tick: 4,
    });

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 4);
    assert_eq!(summary.num_unique_keys, 4);
}

#[test]
fn test_summary_repeated_accesses() {
    let mut trace = CacheTrace::new(true);

    for i in 0..10 {
        if i % 2 == 0 {
            trace.record_event(CacheEvent::Hit {
                key: 1,
                tick: i as u64,
            });
        } else {
            trace.record_event(CacheEvent::Hit {
                key: 2,
                tick: i as u64,
            });
        }
    }

    let summary = Summary::summarize(&trace);

    assert_eq!(summary.num_requests, 10);
    assert_eq!(summary.num_unique_keys, 2);
    assert_eq!(summary.most_frequent_key_count, 5);
}
