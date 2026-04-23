use lbce::workloads::looping::LoopingWorkload;
use lbce::workloads::workload::Workload;
use lbce::workloads::zipf::ZipfWorkload;
use std::collections::HashMap;

// LOOPING TESTS
#[test]
fn test_looping_workload_completion() {
    let keys = vec![10, 20, 30];
    let mut w = LoopingWorkload::new(keys, 15);
    let mut count = 0;
    while let Some(_) = w.next_request() {
        count += 1;
    }
    assert_eq!(count, 15);
    assert!(w.is_complete());
    assert_eq!(w.next_request(), None);
}

#[test]
fn test_looping_workload_cycle() {
    let keys = vec![1, 2, 3];
    let mut w = LoopingWorkload::new(keys.clone(), 10);
    let mut results = Vec::new();
    let cycle_num = 2;
    let mut current_cycle = 0;
    while let Some(k) = w.next_request() {
        results.push(k);
        if results.len() == keys.len() * (current_cycle + 1) {
            current_cycle += 1;
        }
        if current_cycle == cycle_num {
            break;
        }
    }
    assert_eq!(results, vec![1, 2, 3, 1, 2, 3]);
}

// ZIPF TESTS
#[test]
fn test_zipf_workload_completion() {
    let keys = vec![10, 20, 30];
    let mut w = ZipfWorkload::new(keys, 15);
    let mut count = 0;
    while let Some(_) = w.next_request() {
        count += 1;
    }
    assert_eq!(count, 15);
    assert!(w.is_complete());
    assert_eq!(w.next_request(), None);
}

#[test]
fn test_zipf_distribution_skew() {
    let keys = vec![1, 2, 3, 4, 5];
    let total_requests = 10000;
    let mut w = ZipfWorkload::new(keys.clone(), total_requests);

    let mut counts = HashMap::new();
    // update hashmap with counts as you see each key
    while let Some(k) = w.next_request() {
        *counts.entry(k).or_insert(0) += 1;
    }
    // assert that each subsequent count must be smaller than the previous
    let mut prev_count: usize = usize::MAX;
    for &k in &keys {
        // scuffed, prevents rust from throwing an error bc option incompatible w/ usize
        let current_count = *counts.get(&k).unwrap_or(&0);
        assert!(current_count < prev_count);
        prev_count = current_count;
    }
}
