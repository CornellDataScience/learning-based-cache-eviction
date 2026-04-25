use lbce::core::policy::Policy;
use lbce::core::trace::{Request, RequestTrace};
use lbce::policies::belady::BeladyPolicy;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_trace(keys: &[u64]) -> RequestTrace {
    let mut t = RequestTrace::new();
    for &k in keys {
        t.push(Request::new(k));
    }
    t
}

/// Build a policy, pre-load `resident` keys into cached_keys via insert(),
/// then simulate that we are at request `at_index` by popping each key's
/// future queue up through (but not including) `at_index`.
///
/// `at_index` is the index of the *incoming* miss — i.e., the current
/// request's index has already been popped by on_miss() before victim() runs.
fn setup(
    trace: &RequestTrace,
    resident: &[u64],
    at_index: usize,
) -> BeladyPolicy {
    let mut p = BeladyPolicy::new(trace);

    // Advance each resident key's queue to just before at_index,
    // mirroring what on_hit / on_miss would have done for prior requests.
    for &key in resident {
        // Pop entries whose index < at_index (those requests have passed).
        // The entry AT at_index is the current one; on_miss() will pop it.
        // But since we're calling victim() directly in tests, we manually
        // advance to the correct state.
        for (i, req) in trace.requests().iter().enumerate() {
            if i >= at_index {
                break;
            }
            if req.key == key {
                p.on_hit(key); // advance the queue (pop front)
            }
        }
        p.insert(key);
    }

    p
}

// ── victim() selection ────────────────────────────────────────────────────────

/// victim() returns None when the cache is empty.
#[test]
fn victim_none_when_empty() {
    let trace = make_trace(&[1, 2, 3]);
    let mut p = BeladyPolicy::new(&trace);
    assert_eq!(p.victim(), None);
}

/// Among two keys, evicts the one with the later next use.
///
/// Trace:  A B C A B   (indices 0-4)
/// At index 2 (request C), cache holds A and B.
/// A's next use = 3, B's next use = 4  →  victim = B.
#[test]
fn victim_evicts_furthest_next_use() {
    //            0  1  2  3  4
    let trace = make_trace(&[1, 2, 0, 1, 2]);
    // At index 2: cache holds keys 1 (next=3) and 2 (next=4).
    let mut p = setup(&trace, &[1, 2], 2);
    // Simulate on_miss for the incoming key (0) at index 2.
    p.on_miss(0);
    assert_eq!(p.victim(), Some(2)); // 2 has next use at index 4 > 3
}

/// A key with no future use (never accessed again) is always the victim.
///
/// Trace:  A B C A   (indices 0-3)
/// At index 2 (request C), cache holds A and B.
/// A's next use = 3, B has no more uses → victim = B (next = MAX).
#[test]
fn victim_prefers_key_with_no_future_use() {
    //            0  1  2  3
    let trace = make_trace(&[1, 2, 0, 1]);
    // At index 2: cache holds keys 1 (next=3) and 2 (no future use).
    let mut p = setup(&trace, &[1, 2], 2);
    p.on_miss(0);
    assert_eq!(p.victim(), Some(2));
}

/// When all candidates have the same next-use index, victim() still returns
/// some key (tie-breaking is arbitrary but must not panic or return None).
#[test]
fn victim_handles_tie() {
    //            0  1  2  3  4
    let trace = make_trace(&[1, 2, 0, 1, 2]);
    // At index 4 (request 2 again), cache holds 1 and 0.
    // Key 1: last seen at index 3, no future use → MAX.
    // Key 0: last seen at index 2, no future use → MAX.
    let mut p = setup(&trace, &[1, 0], 4);
    p.on_miss(2);
    let v = p.victim();
    assert!(v == Some(1) || v == Some(0), "expected one of the tied candidates");
}

// ── insert / remove ───────────────────────────────────────────────────────────

/// After remove(), the key is no longer a candidate.
#[test]
fn remove_excludes_key_from_victim() {
    //            0  1  2  3  4
    let trace = make_trace(&[1, 2, 0, 1, 2]);
    let mut p = setup(&trace, &[1, 2], 2);
    p.on_miss(0);
    p.remove(2); // only key 1 remains
    assert_eq!(p.victim(), Some(1));
}

/// victim() returns None once all keys are removed.
#[test]
fn victim_none_after_all_removed() {
    let trace = make_trace(&[1, 2, 3]);
    let mut p = BeladyPolicy::new(&trace);
    p.insert(1);
    p.insert(2);
    p.remove(1);
    p.remove(2);
    assert_eq!(p.victim(), None);
}

/// Inserting a key makes it a candidate for eviction.
#[test]
fn insert_makes_key_a_candidate() {
    let trace = make_trace(&[1]);
    let mut p = BeladyPolicy::new(&trace);
    p.on_miss(1);      // pop index 0 from key 1's queue
    p.insert(1);
    assert_eq!(p.victim(), Some(1));
}

// ── future-position advancement ───────────────────────────────────────────────

/// on_hit() advances the key's queue so victim() uses the *next* future index.
///
/// Trace:  A A B   (indices 0-2)
/// Simulate: request 0 (hit on A after prior insert), then at index 2 (B
/// incoming miss, cache holds A).
/// After the hit at 0, A's next use = 1, then after the hit at 1, A's next
/// use = MAX. So victim at index 2 → A.
#[test]
fn on_hit_advances_future_queue() {
    //            0  1  2
    let trace = make_trace(&[1, 1, 2]);
    let mut p = BeladyPolicy::new(&trace);
    // Request 0: miss, insert A.
    p.on_miss(1);
    p.insert(1);
    // Request 1: hit on A — pops index 1, leaving A with no future use.
    p.on_hit(1);
    // Request 2: miss on B, cache full (capacity-1 check is caller's job;
    // we just need victim() to return A as the only candidate).
    p.on_miss(2);
    p.insert(2);
    // After on_miss for B and then removing B, only A is left.
    p.remove(2);
    assert_eq!(p.victim(), Some(1)); // A has no future use → MAX
}

/// on_miss() advances the queue so victim() doesn't count the current request.
///
/// Trace:  C A B C   (indices 0-3)
/// At index 2 (request B), cache holds A (next=MAX) and C (next=3).
/// on_miss(B) pops nothing from B's queue (B has future at 2, which was
/// the "current" one already consumed by setup).
/// victim → A (next=MAX > 3).
#[test]
fn on_miss_does_not_affect_resident_keys() {
    //            0  1  2  3
    let trace = make_trace(&[3, 1, 2, 3]);
    // At index 2: cache holds key 1 (no future use) and key 3 (next=3).
    let mut p = setup(&trace, &[1, 3], 2);
    p.on_miss(2);
    assert_eq!(p.victim(), Some(1)); // 1 has next=MAX > 3
}

// ── three-way selection ───────────────────────────────────────────────────────

/// With three cached keys, victim() picks the one furthest in the future.
///
/// Trace:  A B C D B C A   (indices 0-6)
/// At index 3 (request D miss), cache holds A (next=6), B (next=4), C (next=5).
/// Furthest = A (index 6)  →  victim = A.
#[test]
fn victim_correct_with_three_candidates() {
    //            0  1  2  3  4  5  6
    let trace = make_trace(&[1, 2, 3, 4, 2, 3, 1]);
    let mut p = setup(&trace, &[1, 2, 3], 3);
    p.on_miss(4);
    assert_eq!(p.victim(), Some(1)); // A next at 6, B at 4, C at 5
}
