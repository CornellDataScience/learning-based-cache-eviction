use lbce::core::policy::{CacheKey, Policy};
use lbce::policies::fifo::FifoPolicy;

fn filled(capacity: usize, keys: &[CacheKey]) -> FifoPolicy {
    let mut p = FifoPolicy::new(capacity);
    for &k in keys {
        p.insert(k);
    }
    p
}

#[test]
fn no_victim_when_under_capacity() {
    let mut p = filled(3, &[1, 2]);
    assert_eq!(p.on_miss(0), None);
}

#[test]
fn no_victim_when_empty() {
    let mut p = FifoPolicy::new(3);
    assert_eq!(p.on_miss(0), None);
}

#[test]
fn victim_is_oldest_at_capacity() {
    let mut p = filled(3, &[10, 20, 30]);
    assert_eq!(p.on_miss(0), Some(10));
}

#[test]
fn on_miss_does_not_mutate_order() {
    let mut p = filled(2, &[1, 2]);
    let _ = p.on_miss(0);
    let _ = p.on_miss(0);
    assert_eq!(p.on_miss(0), Some(1));
}

#[test]
fn remove_front_then_victim_updates() {
    let mut p = filled(3, &[1, 2, 3]);
    p.remove(1);
    p.insert(4);
    assert_eq!(p.on_miss(0), Some(2));
}

#[test]
fn eviction_cycle_respects_fifo_order() {
    let mut p = filled(3, &[1, 2, 3]);

    let victim = p.on_miss(4);
    assert_eq!(victim, Some(1));
    p.remove(victim.unwrap());
    p.insert(4);

    let victim = p.on_miss(5);
    assert_eq!(victim, Some(2));
    p.remove(victim.unwrap());
    p.insert(5);

    assert_eq!(p.order.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);
}

#[test]
fn on_hit_does_not_change_order() {
    let mut p = filled(3, &[1, 2, 3]);
    p.on_hit(3);
    p.on_hit(1);
    assert_eq!(p.order.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
}
