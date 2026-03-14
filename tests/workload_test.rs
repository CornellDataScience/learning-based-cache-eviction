use lbce::workloads::phase::PhaseWorkload;
use lbce::workloads::bursty::BurstyWorkload;
use lbce::workloads::workload::Workload;

// ── PHASE TESTS ────────────────────────────────────────────────────────────────

#[test]
fn test_phase_workload_completion() {
    // 3 phases * 10 requests each = 30 total
    let mut w = PhaseWorkload::new(3, 4, 10);
    let mut count = 0;
    while let Some(_) = w.next_request() {
        count += 1;
    }
    assert_eq!(count, 30);
    assert!(w.is_complete());
    assert_eq!(w.next_request(), None);
}

#[test]
fn test_phase_workload_disjoint_key_spaces() {
    // 3 phases, 3 keys each, 6 requests per phase
    // phase 0 keys: 0,1,2
    // phase 1 keys: 3,4,5
    // phase 2 keys: 6,7,8
    let mut w = PhaseWorkload::new(3, 3, 6);
    let mut phase_keys: Vec<Vec<u64>> = vec![Vec::new(); 3];
    let mut request_num = 0;

    while let Some(k) = w.next_request() {
        let phase = request_num / 6; // 6 requests per phase
        phase_keys[phase].push(k);
        request_num += 1;
    }

    // verify each phase only contains its own keys and no overlap between phases
    let phase0: std::collections::HashSet<u64> = phase_keys[0].iter().cloned().collect();
    let phase1: std::collections::HashSet<u64> = phase_keys[1].iter().cloned().collect();
    let phase2: std::collections::HashSet<u64> = phase_keys[2].iter().cloned().collect();

    assert!(phase0.is_disjoint(&phase1), "phase 0 and phase 1 share keys");
    assert!(phase1.is_disjoint(&phase2), "phase 1 and phase 2 share keys");
    assert!(phase0.is_disjoint(&phase2), "phase 0 and phase 2 share keys");
}

#[test]
fn test_phase_workload_key_range() {
    // phase 0 should only produce keys 0..keys_per_phase
    // phase 1 should only produce keys keys_per_phase..2*keys_per_phase
    let keys_per_phase = 4;
    let mut w = PhaseWorkload::new(2, keys_per_phase, 8);
    let mut phase0_keys = Vec::new();
    let mut phase1_keys = Vec::new();

    for i in 0..16 {
        if let Some(k) = w.next_request() {
            if i < 8 {
                phase0_keys.push(k);
            } else {
                phase1_keys.push(k);
            }
        }
    }

    // phase 0 keys must all be in range [0, keys_per_phase)
    for k in &phase0_keys {
        assert!(*k < keys_per_phase, "phase 0 produced out-of-range key {}", k);
    }
    // phase 1 keys must all be in range [keys_per_phase, 2*keys_per_phase)
    for k in &phase1_keys {
        assert!(
            *k >= keys_per_phase && *k < 2 * keys_per_phase,
            "phase 1 produced out-of-range key {}",
            k
        );
    }
}

// ── BURSTY TESTS ───────────────────────────────────────────────────────────────

#[test]
fn test_bursty_workload_completion() {
    // 2 cycles * (6 quiet + 4 burst) = 20 total
    let mut w = BurstyWorkload::new(2, 6, 4, 5);
    let mut count = 0;
    while let Some(_) = w.next_request() {
        count += 1;
    }
    assert_eq!(count, 20);
    assert!(w.is_complete());
    assert_eq!(w.next_request(), None);
}

#[test]
fn test_bursty_burst_key_is_outside_background() {
    // background keys are 0..background_keys
    // burst key should be exactly background_keys (just outside the range)
    let background_keys = 5;
    let burst_key = background_keys; // expected: 5
    let mut w = BurstyWorkload::new(1, 6, 4, background_keys);
    let mut all_keys: Vec<u64> = Vec::new();

    while let Some(k) = w.next_request() {
        all_keys.push(k);
    }

    // burst period is the last 4 requests (quiet=6, burst=4)
    let burst_requests = &all_keys[6..];
    for k in burst_requests {
        assert_eq!(*k, burst_key, "burst period contained unexpected key {}", k);
    }

    // quiet period should never contain the burst key
    let quiet_requests = &all_keys[..6];
    for k in quiet_requests {
        assert_ne!(*k, burst_key, "quiet period contained burst key");
    }
}

#[test]
fn test_bursty_quiet_period_stays_in_background_range() {
    let background_keys: u64 = 5;
    let mut w = BurstyWorkload::new(2, 6, 4, background_keys);
    let mut request_num = 0;
    let cycle_length = 10; // 6 quiet + 4 burst

    while let Some(k) = w.next_request() {
        let position_in_cycle = request_num % cycle_length;
        if position_in_cycle < 6 {
            // quiet period — key must be within background range
            assert!(
                k < background_keys,
                "quiet period produced out-of-range key {}",
                k
            );
        }
        request_num += 1;
    }
}

#[test]
fn test_bursty_burst_period_is_single_key() {
    let background_keys: u64 = 5;
    let mut w = BurstyWorkload::new(3, 6, 4, background_keys);
    let mut request_num = 0;
    let cycle_length = 10;

    while let Some(k) = w.next_request() {
        let position_in_cycle = request_num % cycle_length;
        if position_in_cycle >= 6 {
            // burst period — all keys must be identical (the burst key)
            assert_eq!(k, background_keys, "burst period had unexpected key {}", k);
        }
        request_num += 1;
    }
}